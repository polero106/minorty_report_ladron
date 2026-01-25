"""
Data Loader - Carga datos directamente desde Neo4j sin normalización
Reemplaza etl_policial.py para trabajar con el nuevo city_generator.py
"""

import torch
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData
import numpy as np


class MadridDataLoader:
    """
    Carga datos desde Neo4j para el sistema Minority Report.
    Trabaja con coordenadas REALES de Madrid (no normalizadas).
    """
    
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.data = HeteroData()
        
        # Mapeos de ID Neo4j -> Índice PyTorch
        self.node_map = {'Persona': {}, 'Ubicacion': {}, 'Warning': {}}

    def load_nodes(self):
        """Carga nodos con coordenadas reales (lat, lon)"""
        print("   -> Cargando nodos desde Neo4j (coordenadas reales)...")
        
        # 1. PERSONAS (Features: Risk Seed + Coordenadas Reales)
        # Vector: [risk, lat_real, lon_real] -> Dim 3
        query_p = """
        MATCH (n:Persona) 
        RETURN elementId(n) as id, 
               n.risk_seed as risk, 
               n.lat as lat, 
               n.lon as lon
        """
        self._process_node_type(query_p, 'Persona', 'risk')

        # 2. UBICACIONES (Features: Peligrosidad + Coordenadas Reales)
        # Vector: [danger, lat_real, lon_real] -> Dim 3
        query_u = """
        MATCH (n:Ubicacion) 
        RETURN elementId(n) as id, 
               n.peligrosidad as danger, 
               n.lat as lat, 
               n.lon as lon
        """
        self._process_node_type(query_u, 'Ubicacion', 'danger')

        # 3. WARNINGS (Features: Gravedad + Coordenadas Reales)
        # Vector: [gravity, lat_real, lon_real] -> Dim 3
        query_w = """
        MATCH (n:Warning) 
        RETURN elementId(n) as id, 
               n.gravedad as gravity, 
               n.lat as lat, 
               n.lon as lon
        """
        self._process_node_type(query_w, 'Warning', 'gravity')

    def _process_node_type(self, query, label, value_field):
        """Procesa un tipo de nodo sin normalización"""
        features = []
        
        with self.driver.session() as session:
            result = session.run(query)
            idx = 0
            for record in result:
                neo4j_id = record['id']
                
                # Extraer valores
                value = record[value_field] if record[value_field] is not None else 0.0
                lat = record['lat'] if record['lat'] is not None else 40.416775  # Centro Madrid
                lon = record['lon'] if record['lon'] is not None else -3.703790
                
                # Vector de características [value, lat, lon] - SIN NORMALIZACIÓN
                feat_vec = [float(value), float(lat), float(lon)]
                
                features.append(feat_vec)
                self.node_map[label][neo4j_id] = idx
                idx += 1
        
        # Asignar a PyTorch Geometric
        if features:
            self.data[label].x = torch.tensor(features, dtype=torch.float)
            self.data[label].num_nodes = len(features)
            self.data[label].original_ids = list(self.node_map[label].keys())
            
            print(f"      - {label}: {len(features)} nodos cargados (coordenadas reales)")

    def load_edges(self):
        """Carga relaciones entre nodos"""
        print("   -> Cargando relaciones...")
        
        edge_types = [
            ('Persona', 'VIVE_EN', 'Ubicacion', 'RESIDE_EN'),
            ('Persona', 'COMETIO', 'Warning', 'COMETIDO_POR'),
            ('Warning', 'OCURRIO_EN', 'Ubicacion', 'ESCENARIO_DE')
        ]

        with self.driver.session() as session:
            for src_label, rel_type, dst_label, rev_rel_type in edge_types:
                query = f"""
                MATCH (s:{src_label})-[r:{rel_type}]->(t:{dst_label})
                RETURN elementId(s) as src, elementId(t) as dst
                """
                src_indices = []
                dst_indices = []
                
                result = session.run(query)
                for record in result:
                    s_id, t_id = record['src'], record['dst']
                    
                    if s_id in self.node_map[src_label] and t_id in self.node_map[dst_label]:
                        src_indices.append(self.node_map[src_label][s_id])
                        dst_indices.append(self.node_map[dst_label][t_id])

                if src_indices:
                    # Relación Original
                    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
                    self.data[src_label, rel_type, dst_label].edge_index = edge_index
                    
                    # Relación Inversa
                    rev_edge_index = torch.tensor([dst_indices, src_indices], dtype=torch.long)
                    self.data[dst_label, rev_rel_type, src_label].edge_index = rev_edge_index
                    
                    print(f"      - {rel_type} / {rev_rel_type}: {len(src_indices)} conexiones")

    def get_data(self):
        """Retorna el grafo heterogéneo"""
        return self.data
    
    def close(self):
        """Cierra la conexión"""
        self.driver.close()

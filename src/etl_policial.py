
import torch
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData
import numpy as np

# Constantes de Normalización (Límites aproximados de Madrid)
LAT_MIN, LAT_MAX = 40.30, 40.55
LON_MIN, LON_MAX = -3.85, -3.50

class PoliceETL:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.data = HeteroData()
        
        # Mapeos de ID Neo4j -> Índice PyTorch
        self.node_map = {'Persona': {}, 'Ubicacion': {}, 'Warning': {}}

    def _normalize(self, val, min_val, max_val):
        """MinEx Normalization to [0, 1] range"""
        if val is None: return 0.0
        return (float(val) - min_val) / (max_val - min_val)

    def load_nodes(self):
        print("   -> Extrayendo nodos y coordenadas reales (Lat/Lon)...")
        
        # 1. PERSONAS (Features: Risk Seed + Coordenadas Normalizadas)
        # Vector: [risk, lat_norm, lon_norm] -> Dim 3
        query_p = "MATCH (n:Persona) RETURN id(n) as id, n.risk_seed as risk, n.lat as lat, n.lon as lon"
        self._process_node_type(query_p, 'Persona', ['risk', 'lat', 'lon'])

        # 2. UBICACIONES (Features: Peligrosidad + Coordenadas Normalizadas)
        # Vector: [danger, lat_norm, lon_norm] -> Dim 3
        query_u = "MATCH (n:Ubicacion) RETURN id(n) as id, n.peligrosidad as danger, n.lat as lat, n.lon as lon"
        self._process_node_type(query_u, 'Ubicacion', ['danger', 'lat', 'lon'])

        # 3. WARNINGS (Features: Gravedad + Coordenadas Normalizadas)
        # Vector: [gravity, lat_norm, lon_norm] -> Dim 3
        query_w = "MATCH (n:Warning) RETURN id(n) as id, n.gravedad as gravity, n.lat as lat, n.lon as lon"
        self._process_node_type(query_w, 'Warning', ['gravity', 'lat', 'lon'])

    def _process_node_type(self, query, label, feature_cols):
        features = []
        indices = []
        
        with self.driver.session() as session:
            result = session.run(query)
            idx = 0
            for record in result:
                neo4j_id = record['id']
                
                # Extracción de valores raw
                vals = {}
                for col in feature_cols:
                    vals[col] = record[col] if record[col] is not None else 0.0

                # Normalización Específica
                lat_norm = self._normalize(vals.get('lat', LAT_MIN), LAT_MIN, LAT_MAX)
                lon_norm = self._normalize(vals.get('lon', LON_MIN), LON_MIN, LON_MAX)
                
                # Construcción del vector de características según tipo
                if label == 'Persona':
                    # [risk, lat, lon]
                    feat_vec = [float(vals['risk']), lat_norm, lon_norm]
                elif label == 'Ubicacion':
                    # [danger, lat, lon]
                    feat_vec = [float(vals['danger']), lat_norm, lon_norm]
                elif label == 'Warning':
                    # [gravity, lat, lon]
                    feat_vec = [float(vals['gravity']), lat_norm, lon_norm]
                
                features.append(feat_vec)
                self.node_map[label][neo4j_id] = idx
                idx += 1
        
        # Asignar a PyTorch Geometric
        if features:
            self.data[label].x = torch.tensor(features, dtype=torch.float)
            self.data[label].num_nodes = len(features)
            
            # Guardamos ids originales para referencia futura (en predicción)
            self.data[label].original_ids = torch.tensor(list(self.node_map[label].keys()), dtype=torch.long)
            
            print(f"      - {label}: {len(features)} nodos cargados. Dimensión: {len(features[0])}")

    def load_edges(self):
        print("   -> Extrayendo relaciones...")
        # Definir relaciones originales y sus inversas para que el GNN propague info en ambas direcciones
        edge_types = [
            # (Src, Rel, Dst, Rev_Rel)
            ('Persona', 'VIVE_EN', 'Ubicacion', 'RESIDE_EN'),
            ('Persona', 'COMETIO', 'Warning', 'COMETIDO_POR'),
            ('Warning', 'OCURRIO_EN', 'Ubicacion', 'ESCENARIO_DE')
        ]

        with self.driver.session() as session:
            for src_label, rel_type, dst_label, rev_rel_type in edge_types:
                query = f"""
                MATCH (s:{src_label})-[r:{rel_type}]->(t:{dst_label})
                RETURN id(s) as src, id(t) as dst
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
                    # Relación Original (Direccional)
                    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
                    self.data[src_label, rel_type, dst_label].edge_index = edge_index
                    
                    # Relación Inversa (Para que el mensaje vuelva)
                    rev_edge_index = torch.tensor([dst_indices, src_indices], dtype=torch.long)
                    self.data[dst_label, rev_rel_type, src_label].edge_index = rev_edge_index
                    
                    print(f"      - {rel_type} / {rev_rel_type}: {len(src_indices)} conexiones bidireccionales.")

    def get_data(self):
        return self.data
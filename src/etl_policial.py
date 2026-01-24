import torch
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData

class PoliceETL:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.data = HeteroData()
        
        # Mapeos de ID Neo4j -> Índice PyTorch
        self.node_map = {'Persona': {}, 'Ubicacion': {}, 'Warning': {}}

    def load_nodes(self):
        print("   -> Extrayendo nodos y coordenadas 3D...")
        
        # 1. PERSONAS (Features: Risk Seed + Coordenadas Normalizadas)
        # Normalizamos coords dividiendo por 100 para que la red neuronal no explote
        query_p = "MATCH (n:Persona) RETURN id(n) as id, n.risk_seed as risk, n.x as x, n.y as y, n.z as z"
        self._process_node_type(query_p, 'Persona', feature_cols=['risk', 'x', 'y', 'z'])

        # 2. UBICACIONES (Features: Peligrosidad + Coordenadas)
        query_u = "MATCH (n:Ubicacion) RETURN id(n) as id, n.peligrosidad as danger, n.x as x, n.y as y, n.z as z"
        self._process_node_type(query_u, 'Ubicacion', feature_cols=['danger', 'x', 'y', 'z'])

        # 3. WARNINGS (Features: Gravedad + Coordenadas)
        query_w = "MATCH (n:Warning) RETURN id(n) as id, n.gravedad as gravity, n.x as x, n.y as y, n.z as z"
        self._process_node_type(query_w, 'Warning', feature_cols=['gravity', 'x', 'y', 'z'])

    def _process_node_type(self, query, label, feature_cols):
        features = []
        indices = []
        
        with self.driver.session() as session:
            result = session.run(query)
            idx = 0
            for record in result:
                neo4j_id = record['id']
                
                # Construir vector de características
                # Ejemplo: [0.3, 0.45, -0.12, 0.0]
                feat_vec = [float(record[col]) if record[col] is not None else 0.0 for col in feature_cols]
                
                # Normalización básica de coordenadas (dividir por 100 si son muy grandes)
                # feat_vec[1] /= 100.0  # x
                # feat_vec[2] /= 100.0  # y
                
                features.append(feat_vec)
                self.node_map[label][neo4j_id] = idx
                idx += 1
        
        # Asignar a PyTorch Geometric
        if features:
            self.data[label].x = torch.tensor(features, dtype=torch.float)
            self.data[label].num_nodes = len(features)
            print(f"      - {label}: {len(features)} nodos cargados.")

    def load_edges(self):
        print("   -> Extrayendo relaciones...")
        # Definir qué relaciones nos importan para el modelo
        # (Origen, Relación, Destino)
        edge_types = [
            ('Persona', 'VIVE_EN', 'Ubicacion'),
            ('Persona', 'COMETIO', 'Warning'),
            ('Warning', 'OCURRIO_EN', 'Ubicacion')
        ]

        with self.driver.session() as session:
            for src_label, rel_type, dst_label in edge_types:
                query = f"""
                MATCH (s:{src_label})-[r:{rel_type}]->(t:{dst_label})
                RETURN id(s) as src, id(t) as dst
                """
                src_indices = []
                dst_indices = []
                
                result = session.run(query)
                for record in result:
                    s_id, t_id = record['src'], record['dst']
                    
                    # Verificar que ambos nodos existen en nuestro mapeo
                    if s_id in self.node_map[src_label] and t_id in self.node_map[dst_label]:
                        src_indices.append(self.node_map[src_label][s_id])
                        dst_indices.append(self.node_map[dst_label][t_id])

                if src_indices:
                    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
                    self.data[src_label, rel_type, dst_label].edge_index = edge_index
                    print(f"      - {rel_type}: {len(src_indices)} conexiones.")

    def get_data(self):
        return self.data
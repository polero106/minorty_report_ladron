import torch
from torch_geometric.data import HeteroData
from neo4j import GraphDatabase
import pandas as pd

class Neo4jHeteroConnector:
    """
    Conector para transformar datos de Neo4j en un objeto HeteroData de PyG
    e inyectar predicciones criminales.
    """
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def fetch_hetero_data(self):
        """
        Consulta Neo4j para obtener nodos y relaciones, y los convierte a HeteroData.
        Asume etiquetas de nodos como 'Persona' y relaciones como 'CONOCE'.
        """
        data = HeteroData()
        
        with self.driver.session() as session:
            # 1. Obtener Nodos (ejemplo con etiqueta Persona)
            # Podríamos extender esto a múltiples etiquetas
            result_nodes = session.run("MATCH (n:Persona) RETURN id(n) AS id, n.name AS name")
            nodes_df = pd.DataFrame([dict(record) for record in result_nodes])
            
            if nodes_df.empty:
                return data

            # Mapeo de IDs de Neo4j a índices 0..N-1
            node_mapping = {neo_id: i for i, neo_id in enumerate(nodes_df['id'])}
            num_nodes = len(nodes_df)
            
            # Asignamos características (por ahora dummies si no hay n.embeddings)
            data['Persona'].x = torch.eye(num_nodes) # One-hot como ejemplo
            data['Persona'].node_ids = nodes_df['id'].tolist()

            # 2. Obtener Relaciones
            result_rels = session.run("MATCH (a:Persona)-[r:CONOCE]->(b:Persona) RETURN id(a) AS source, id(b) AS target")
            rels_df = pd.DataFrame([dict(record) for record in result_rels])
            
            if not rels_df.empty:
                edge_index = torch.tensor([
                    [node_mapping[s] for s in rels_df['source']],
                    [node_mapping[t] for t in rels_df['target']]
                ], dtype=torch.long)
                
                data['Persona', 'CONOCE', 'Persona'].edge_index = edge_index

        return data

    def inject_predicted_crime(self, source_neo_ids, target_neo_ids, scores):
        """
        Inyecta relaciones PREDICTED_CRIME en Neo4j basadas en los resultados del modelo.
        """
        query = """
        UNWIND $batches AS batch
        MATCH (a) WHERE id(a) = batch.source
        MATCH (b) WHERE id(b) = batch.target
        MERGE (a)-[r:PREDICTED_CRIME]->(b)
        SET r.score = batch.score
        """
        batches = [
            {"source": s, "target": t, "score": float(sc)} 
            for s, t, sc in zip(source_neo_ids, target_neo_ids, scores)
        ]
        
        with self.driver.session() as session:
            session.run(query, batches=batches)
        print(f"Inyectadas {len(batches)} relaciones PREDICTED_CRIME en Neo4j.")

# Ejemplo de uso (comentado):
# connector = Neo4jHeteroConnector("bolt://localhost:7687", "neo4j", "password")
# hetero_data = connector.fetch_hetero_data()
# print(hetero_data)
# connector.inject_predicted_crime([1, 2], [3, 4], [0.95, 0.88])

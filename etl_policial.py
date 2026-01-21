import torch
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData


class PoliceETL:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.data = HeteroData()

        # Diccionarios para traducir IDs: Neo4j -> PyTorch
        self.person_map = {}
        self.location_map = {}
        self.crime_map = {}

    def load_nodes(self):
        query = """
        MATCH (n) 
        RETURN id(n) as neo4j_id, labels(n) as labels, properties(n) as props
        """

        # Listas temporales para construir tensores
        person_features = []
        location_features = []
        crime_features = []

        with self.driver.session() as session:
            result = session.run(query)

            # Contadores para asignar índices 0, 1, 2...
            p_idx, l_idx, c_idx = 0, 0, 0

            for record in result:
                nid = record['neo4j_id']
                labels = record['labels']

                # Aquí simulamos features (embeddings).
                # En producción, esto vendría de un modelo NLP o propiedades reales.
                dummy_embedding = [0.5, 0.5]

                if 'Persona' in labels:
                    self.person_map[nid] = p_idx
                    person_features.append(dummy_embedding)  # Feature X
                    p_idx += 1

                elif 'Ubicacion' in labels:
                    self.location_map[nid] = l_idx
                    location_features.append(dummy_embedding)
                    l_idx += 1

                elif 'Crime' in labels:
                    self.crime_map[nid] = c_idx
                    crime_features.append(dummy_embedding)
                    c_idx += 1

        # Convertir listas a Tensores de PyTorch (Nodos x Features)
        self.data['Persona'].x = torch.tensor(
            person_features, dtype=torch.float)
        self.data['Ubicacion'].x = torch.tensor(
            location_features, dtype=torch.float)
        self.data['Crime'].x = torch.tensor(crime_features, dtype=torch.float)

        # Guardamos el mapeo inverso para cuando hagamos el reporte (Write-back)
        self.data['Persona'].n_id = torch.tensor(
            list(self.person_map.keys()), dtype=torch.long)
        self.data['Ubicacion'].n_id = torch.tensor(
            list(self.location_map.keys()), dtype=torch.long)

        print(
            f"Nodos procesados: {p_idx} Personas, {l_idx} Ubicaciones, {c_idx} Crímenes")

    def load_edges(self):
        # Extraemos relaciones específicas.
        # Ejemplo: Persona -> WILL_COMMIT -> Crime
        query = """
        MATCH (s)-[r]->(t)
        RETURN id(s) as src, id(t) as dst, type(r) as type
        """

        sources_commit = []
        targets_commit = []

        sources_loc = []
        targets_loc = []

        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                src, dst, r_type = record['src'], record['dst'], record['type']

                # Relación: Persona -> WILL_COMMIT -> Crime
                if r_type == 'WILL_COMMIT' and src in self.person_map and dst in self.crime_map:
                    sources_commit.append(self.person_map[src])
                    targets_commit.append(self.crime_map[dst])

                # Relación: Crime -> OCCURS_IN -> Ubicacion
                elif r_type == 'OCCURS_IN' and src in self.crime_map and dst in self.location_map:
                    sources_loc.append(self.crime_map[src])
                    targets_loc.append(self.location_map[dst])

        # Construir edge_index (formato [2, num_edges])
        self.data['Persona', 'WILL_COMMIT', 'Crime'].edge_index = torch.tensor(
            [sources_commit, targets_commit], dtype=torch.long
        )

        self.data['Crime', 'OCCURS_IN', 'Ubicacion'].edge_index = torch.tensor(
            [sources_loc, targets_loc], dtype=torch.long
        )

        print("Relaciones construidas y mapeadas.")

    def get_data(self):
        return self.data


# --- ZONA DE EJECUCIÓN ---
if __name__ == "__main__":
    # Ajusta con tus credenciales (usa ssc si estás en local/windows con problemas SSL)
    URI = "neo4j+ssc://5d9c9334.databases.neo4j.io"
    AUTH = ("neo4j", "oTzaPYT99TgH-GM2APk0gcFlf9k16wrTcVOhtfmAyyA")

    etl = PoliceETL(URI, AUTH)
    etl.load_nodes()
    etl.load_edges()

    graph_data = etl.get_data()
    print("\n--- OBJETO HETERODATA FINAL ---")
    print(graph_data)

    # Verificación rápida
    print(f"\nEjemplo de tensores:")
    print(f"Features de Personas: {graph_data['Persona'].x.shape}")
    print(
        f"Aristas Crime-Location: {graph_data['Crime', 'OCCURS_IN', 'Ubicacion'].edge_index.shape}")

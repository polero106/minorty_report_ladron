# Prompt para Generar Notebook Minority Report Graph-GAN

**Rol:** Eres un experto en Inteligencia Artificial, especializado en Graph Neural Networks (GNNs) y Bases de Datos de Grafos.

**Tarea:** Genera un Jupyter Notebook completo y ejecutable para el proyecto "Minority Report Graph-GAN".

**Contexto:** Estamos simulando un escenario adversarial donde un "Criminal" intenta crear conexiones indetectables en una red, y una "Policía" intenta detectarlos.

**Requisitos del Notebook:**

1.  **Celdas de Importación:**
    *   Incluye `torch`, `torch_geometric` (específicamente `GATConv`, `SAGEConv`), `neo4j` (GraphDatabase), `pandas`, `networkx`, `matplotlib.pyplot`.

2.  **Clase `Neo4jConnector`:**
    *   Implementa el patrón Singleton o una clase instanciable con `uri`, `user`, `password`.
    *   Método `get_graph_data()`: Debe ejecutar una query Cypher (ej. `MATCH (n)-[r]->(m) RETURN n, r, m`) y convertir los resultados a un objeto `torch_geometric.data.Data`. Asegúrate de mapear los IDs de Neo4j a índices continuos (0, 1, 2...) para los tensores de PyTorch.
    *   Método `update_predictions(node_ids, scores)`: Recibe IDs de nodos (originales de Neo4j) y sus puntajes de criminalidad, y actualiza una propiedad `criminal_prob` en Neo4j.

3.  **Modelos (Arquitectura GAN):**
    *   **Discriminador (`PoliceNet`):** Clase que hereda de `torch.nn.Module`. Usa 2 o 3 capas `SAGEConv`. Salida: Probabilidad (Sigmoid) de ser criminal.
    *   **Generador (`CriminalNet`):** Clase que hereda de `torch.nn.Module`. Usa capas `GATConv` con `heads=4` para atención. Entrada: Vector latente (ruido) + características del grafo. Salida: Matriz de adyacencia o embeddings que sugieran nuevas aristas.

4.  **Bucle de Entrenamiento (Training Loop):**
    *   Inicializa optimizadores `Adam`.
    *   Ciclo por épocas (epochs):
        *   **Entrenar Discriminador:** Real (datos Neo4j) vs Fake (generados por CriminalNet). Loss: `BCELoss`.
        *   **Entrenar Generador:** Generar grafo/aristas -> Pasar por Discriminador -> Loss: Maximizar error del discriminador (o minimizar 1 - prob).
    *   Imprimir pérdidas de ambos en cada N épocas.

5.  **Visualización y Ejecución:**
    *   Función `visualize_graph(data)` usando `networkx` para mostrar el estado actual del grafo en el notebook.
    *   Celda final con un string de comando Cypher recomendado para visualizar los resultados en Neo4j Browser (ej. colorear nodos según `criminal_prob`).

**Notas Adicionales:**
*   Comenta el código extensamente en español.
*   Usa placeholders para las credenciales de Neo4j (`bolt://localhost:7687`, `neo4j`, `password`).
*   Asegura que las dimensiones de los tensores sean consistentes.

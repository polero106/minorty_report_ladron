criminal-graph-gan/
│
├── config/
│   ├── config.yaml          # Hiperparámetros (learning rate, epochs, hidden dim)
│   └── database.ini         # Credenciales de Neo4j (URI, usuario, password)
│
├── data/
│   ├── raw/                 # Datasets iniciales (ej. Elliptic, Cora o sintéticos)
│   └── processed/           # Grafos procesados listos para PyTorch Geometric
│
├── src/
│   ├── __init__.py
│   ├── database/
│   │   ├── connector.py     # Clase para conectar con Neo4j (Driver)
│   │   └── graph_loader.py  # Extrae subgrafos de Neo4j a objetos PyG Data
│   │
│   ├── models/
│   │   ├── layers.py        # Definición de capas personalizadas
│   │   ├── generator.py     # El "Agente Criminal" (GraphSAGE + MLP)
│   │   └── discriminator.py # El "Agente Policial" (GraphSAGE + Clasificador)
│   │
│   ├── training/
│   │   ├── gan_loss.py      # Funciones de pérdida (Adversarial + Camuflaje)
│   │   └── trainer.py       # Bucle de entrenamiento Minimax
│   │
│   └── utils/
│       ├── visualization.py # Scripts para colorear nodos en Neo4j según riesgo
│       └── metrics.py       # Precisión, Recall, Indistinguibilidad
│
├── notebooks/               # Jupyter notebooks para prototipado rápido
├── main.py                  # Punto de entrada para ejecutar el entrenamiento
├── requirements.txt         # Dependencias (torch, torch-geometric, neo4j)
└── README.md
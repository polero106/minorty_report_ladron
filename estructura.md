pre-crime-graph-analysis/
├── .env                        # Variables de entorno (Credenciales Neo4j, API Keys)
├── .gitignore
├── README.md
├── requirements.txt            # Dependencias (fastapi, neo4j, torch, torch-geometric)
├── main.py                     # Punto de entrada de la aplicación FastAPI
│
├── app/
│   ├── __init__.py
│   ├── config.py               # Configuración global (Settings)
│   │
│   ├── api/                    # Capa de API (FastAPI)
│   │   ├── __init__.py
│   │   ├── routes.py           # Endpoints (e.g., /predict/risk, /analyze/suspect)
│   │   └── schemas.py          # Modelos Pydantic (Request/Response bodies)
│   │
│   ├── database/               # Capa de Datos (Neo4j)
│   │   ├── __init__.py
│   │   ├── connection.py       # Gestión del Driver de Neo4j
│   │   └── repository.py       # Queries Cypher (MATCH (p:Person)...)
│   │
│   ├── ml/                     # Capa de Machine Learning (Deep Learning Geométrico)
│   │   ├── __init__.py
│   │   ├── dataset_loader.py   # Convierte datos de Neo4j a objetos 'Data' de PyG
│   │   │
│   │   ├── models/             # Definiciones de Arquitecturas Neuronales
│   │   │   ├── __init__.py
│   │   │   ├── gat.py          # Clase PreCrimeGAT (Atención)
│   │   │   ├── graphsage.py    # Clase PreCrimeSAGE (Inductivo/Embeddings)
│   │   │   └── gan.py          # Clases Generator & Discriminator (Adversarial)
│   │   │
│   │   └── pipelines/          # Lógica de entrenamiento e inferencia
│   │       ├── __init__.py
│   │       ├── training.py     # Scripts de entrenamiento de modelos
│   │       └── inference.py    # Carga modelos guardados para predecir en tiempo real
│   │
│   └── services/               # Lógica de Negocio (Orquestador)
│       ├── __init__.py
│       └── precrime_service.py # Une la DB y el ML para dar respuesta a la API
│
├── data/                       # Almacenamiento local de datos
│   ├── raw/                    # CSVs o JSONs originales (Minority Report data)
│   └── processed/              # Grafos procesados o modelos entrenados (.pth)
│
└── notebooks/                  # Jupyter Notebooks para experimentación y EDA
    ├── 01_data_loading.ipynb
    └── 02_model_prototyping.ipynb
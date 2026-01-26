minorty_report_policia/
├── .env                    # Credenciales Neo4j
├── .github/                # Copilot instructions
├── README.md               # Documentación
├── requirements.txt        # Dependencias
├── models/
│   └── agente_precrime.pth # Modelo entrenado (CRÍTICO)
├── src/                    # Core del sistema
│   ├── city_generator.py   # Generación de datos sintéticos
│   ├── entrenamiento_gan.py# Entrenamiento GAN
│   ├── etl_policial.py     # Carga datos desde Neo4j
│   ├── prediccion.py       # Inferencia standalone
│   └── prediction_service.py# Servicio de predicción
└── viz/
    └── dashboard_madrid_3d.py # Dashboard principal (INTERFAZ)
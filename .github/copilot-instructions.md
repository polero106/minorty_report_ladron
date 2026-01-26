# Minority Report Graph-GAN - AI Agent Instructions

## 🎯 Project Overview
Sistema de predicción de crimen basado en Graph Neural Networks (GNN) que implementa una arquitectura GAN adversarial:
- **CriminalGenerator (GAT)**: Genera conexiones ilícitas que intentan evadir detección
- **PoliceDiscriminator (GraphSAGE)**: Detecta anomalías y clasifica nodos como criminales o seguros
- **Backend**: Neo4j para persistencia del grafo, PyTorch Geometric para deep learning

## 🏗️ Architecture Patterns

### Data Flow Critical Path
1. **Neo4j → ETL → PyTorch Geometric**: Los datos se extraen con `etl_policial.PoliceETL` (módulo en `src/__pycache__` no versionado - importar con `from etl_policial import PoliceETL`)
2. **HeteroData Structure**: Usar `torch_geometric.data.HeteroData` con tipos de nodo: `Persona`, `Warning`, `Ubicacion`
3. **Edge Types**: `('Persona', 'COMETIO', 'Warning')` y `('Warning', 'OCURRIO_EN', 'Ubicacion')`

### Model Architecture Consistency
⚠️ **CRITICAL**: Los modelos `GraphEncoder` y `PoliceDiscriminator` deben tener arquitectura idéntica en:
- [src/entrenamiento_gan.py](src/entrenamiento_gan.py) (entrenamiento)
- [src/prediccion.py](src/prediccion.py) (inferencia)  
- [src/prediction_service.py](src/prediction_service.py) (servicio)

**Template Arquitectura**:
```python
GraphEncoder:
  HeteroConv[SAGEConv(-1, 32)] → ReLU → HeteroConv[SAGEConv(-1, 32)] → ReLU

PoliceDiscriminator:
  Linear(64, 64) → LeakyReLU → Dropout(0.3) → Linear(64, 32) → Linear(32, 1) → Sigmoid
```

### Coordinate System (Madrid)
- **Normalization**: `LAT_MIN=40.30, LAT_MAX=40.55, LON_MIN=-3.85, LON_MAX=-3.50`
- **Storage**: Siempre normalizado [0,1] en Neo4j y features
- **Display**: Denormalizar en [viz/dashboard_madrid_3d.py](viz/dashboard_madrid_3d.py) usando `denormalize_lat/lon()`
- **Real Streets**: [src/city_generator.py](src/city_generator.py) usa OSMnx para calles reales de Madrid

## 🔧 Development Workflows

### Entrenamiento del Modelo
```bash
# Ejecutar desde root (no desde src/)
python src/entrenamiento_gan.py
```
**Guardar checkpoint**: `models/agente_precrime.pth` con keys: `'encoder'`, `'discriminator'`, `'generator'`

### Predicción/Inferencia
```bash
python src/prediccion.py  # Ejecuta inferencia y muestra Top N amenazas
```

### Visualización 3D
```bash
python viz/dashboard_madrid_3d.py  # Panel + PyDeck dashboard interactivo
```

### Generar Datos Sintéticos
```python
from src.city_generator import CityGenerator
gen = CityGenerator(URI, AUTH)
gen.generate_data(num_personas=1000, num_ubicaciones=50)
```

## 🔑 Environment Setup

### Neo4j Connection
- **Local**: `docker-compose.yml` con APOC habilitado
- **Cloud**: Credenciales en `.env`: `NEO4J_URI` y `NEO4J_PASSWORD`
- **Default**: `neo4j+ssc://c6226feb.databases.neo4j.io` (AuraDB)

### Python Environment
```bash
pip install -r requirements.txt
# Critical: torch-geometric requiere versión compatible con torch
```

## ⚠️ Common Pitfalls

1. **Import Path**: `etl_policial` NO está en `src/etl_policial.py` - está compilado en `__pycache__`. Importar directamente: `from etl_policial import PoliceETL`

2. **Device Mismatch**: Siempre mover `data.to(device)` antes de inferencia

3. **Feature Dimensions**: Las features de nodos son `[risk/danger, lat_norm, lon_norm]` (3D)

4. **Edge Index Type**: Debe ser `torch.long`, no `float`

5. **Batch Normalization en Generador**: Requiere `batch_size > 1` - usar `person_emb[0:min(32, len(...))]` si es necesario

## 📊 Data Conventions

### Neo4j Schema
```cypher
(:Persona {id, nombre, edad, risk_seed, x, y, z})
(:Warning {id, tipo, gravedad})
(:Ubicacion {id, nombre, peligrosidad, x, y, z})
```

### Feature Engineering
- **Persona.x**: `[risk_seed, lat_norm, lon_norm]`
- **Ubicacion.x**: `[peligrosidad, lat_norm, lon_norm]`
- **Warning.x**: One-hot encoding de tipo de crimen (si se implementa)

## 🎨 Visualization Tools

- **3D Map**: PyDeck layers para nodos + arcos de amenazas
- **Barrios**: Mapeo ficticio en `BARRIOS` dict (0-20) para visualización
- **Color Coding**: Rojo (alta amenaza) → Amarillo (ubicaciones) → Azul (seguro)

## 📝 Code Style Notes

- **Language**: Comentarios en español, código en inglés
- **Naming**: `snake_case` para funciones, `PascalCase` para clases
- **Imports**: Agrupar `torch`, `torch_geometric`, luego `neo4j`, luego locales

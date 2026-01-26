# Minority Report Graph-GAN

Modelo adversarial de prevención de crimen sobre grafos heterogéneos en Neo4j usando PyTorch Geometric.

## 📐 Arquitectura
- **Datos:** Grafo heterogéneo con nodos `Persona`, `Warning`, `Ubicacion` y aristas `(:Persona)-[:COMETIO]->(:Warning)-[:OCURRIO_EN]->(:Ubicacion)` generado con OSMnx (Madrid) vía [src/city_generator.py](src/city_generator.py).
- **Encoder:** `HeteroConv` con `SAGEConv` (32 → 32) para extraer embeddings de cada tipo ([src/entrenamiento_gan.py](src/entrenamiento_gan.py), [src/prediccion.py](src/prediccion.py), [src/prediction_service.py](src/prediction_service.py)).
- **Discriminador (Police):** MLP `Linear(64→64) → LeakyReLU → Dropout → Linear(64→32) → Linear(32→1) → Sigmoid` sobre pares Persona–Ubicacion.
- **Generador (Criminal):** MLP que combina embedding de persona + ruido y produce embeddings sintéticos de ubicación.
- **Dashboard:** Panel + PyDeck/Bokeh en [viz/dashboard_madrid_3d.py](viz/dashboard_madrid_3d.py) que consume `PredictionService` para mostrar amenazas y red de sospechosos.

## 🔀 Flujo funcional
1) **Generación de ciudad**: `CityGenerator` descarga el callejero real de Madrid (OSMnx), crea personas con `risk_seed`, ubicaciones con `peligrosidad` y crímenes `Warning`; se guarda todo en Neo4j con relaciones `COMETIO` y `OCURRIO_EN`.
2) **Entrenamiento adversarial**: `entrenar_policia()` carga el grafo de Neo4j, arma pares reales Persona–Ubicacion, entrena encoder + discriminador vs. generador, y guarda `models/agente_precrime.pth` (solo encoder+discriminador para inferencia).
3) **Predicción por lote**: `prediccion.py` recarga datos, filtra personas de alto riesgo (>0.6) y ubicaciones peligrosas (>0.5), evalúa todas las combinaciones y muestra el TOP 10 >80% de probabilidad.
4) **Servicio para dashboard**: `PredictionService` carga modelo y grafo, ejecuta inferencia (umbral 0.5), des-normaliza coordenadas y entrega DataFrames de amenazas, métricas y red de sospechosos para el panel 3D.

## 🧠 Entrenamiento Graph-GAN
- **Datos reales**: pares `(Persona, Ubicacion)` derivados de `(:Persona)-[:COMETIO]->(:Warning)-[:OCURRIO_EN]->(:Ubicacion)`; usan todas las relaciones disponibles del grafo Neo4j cargado por `PoliceETL`.
- **Forward encoder**: `HeteroConv[SAGEConv(-1,32)] → ReLU → HeteroConv[SAGEConv(-1,32)] → ReLU` para obtener embeddings de 32 dims por tipo.
- **Discriminador (policía)**: clasifica pares Persona–Ubicacion con BCE + label smoothing en reales (0.9); optimizador Adam lr 5e-4 (encoder+discriminador).
- **Generador (criminal)**: MLP toma embedding de persona (32) + ruido `z∼N(0,I)` de 16 dims y genera embedding sintético de ubicación; optimizador Adam lr 1e-3.
- **Bucle**: 150 épocas, batch completo de pares reales; fase D (reales vs. fakes) + fase G (engañar al D). Se registra pérdida cada 10 épocas.
- **Checkpoint**: guarda en `models/agente_precrime.pth` solo `encoder` + `discriminator` para inferencia ([src/entrenamiento_gan.py](src/entrenamiento_gan.py)).

## 🛠️ Preparación
- Python 3.10+, `pip install -r requirements.txt`.
- Variables en `.env`: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (por defecto apunta a AuraDB de ejemplo).
- Neo4j activo y accesible (puerto 7687).

## 🚀 Orden de ejecución
1. Generar datos sintéticos:
   ```bash
   python src/city_generator.py
   ```
2. Entrenar modelo (solo la primera vez o si reentrenas):
   ```bash
   python src/entrenamiento_gan.py
   ```
3. Lanzar dashboard 3D y disparar inferencia desde el botón rojo:
   ```bash
   BOKEH_ALLOW_WS_ORIGIN=* python viz/dashboard_madrid_3d.py
   ```

## ▶️ Ejecución directa
BOKEH_ALLOW_WS_ORIGIN=* python viz/dashboard_madrid_3d.py
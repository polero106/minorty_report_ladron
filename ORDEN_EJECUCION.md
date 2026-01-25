# 🔄 Orden de Ejecución - Sistema Minority Report Madrid

## 📋 Resumen de Cambios

Con el **nuevo `city_generator.py`**, el sistema ahora trabaja con coordenadas reales de Madrid obtenidas de OpenStreetMap. Ya NO se usa `etl_policial.py` que normalizaba las coordenadas.

### ⚠️ Cambios Importantes

1. **`etl_policial.py` → OBSOLETO** - Reemplazado por `data_loader.py`
2. **Coordenadas Reales** - Todo trabaja con lat/lon reales (40.30-40.55, -3.85 a -3.50)
3. **Sin Normalización** - Los datos se usan tal como vienen de OSMnx
4. **Nuevo Modelo Necesario** - El `agente_precrime.pth` antiguo fue entrenado con datos normalizados

---

## 🚀 Orden de Ejecución Correcto

### 1️⃣ **Generar la Ciudad Sintética** 
```bash
cd /ruta/al/proyecto
python src/city_generator.py
```

**¿Qué hace?**
- Descarga el grafo de calles reales del distrito Centro de Madrid usando OSMnx
- Genera personas, ubicaciones y warnings con coordenadas reales
- Guarda todo en Neo4j con propiedades `lat` y `lon` reales

**Resultado:** Base de datos Neo4j poblada con ~2000 personas, ~150 ubicaciones y warnings

---

### 2️⃣ **Entrenar el Modelo GAN**
```bash
python src/entrenamiento_gan.py
```

**¿Qué hace?**
- Carga datos desde Neo4j usando `MadridDataLoader` (coordenadas reales)
- Entrena el sistema adversario:
  - **Generador (Criminal):** Intenta predecir crímenes
  - **Discriminador (Policía):** Aprende a detectar patrones criminales
- Guarda el modelo entrenado en `models/agente_precrime.pth`

**Duración:** ~5-10 minutos (300 epochs)

**Resultado:** Modelo `agente_precrime.pth` con coordenadas reales

---

### 3️⃣ **Ejecutar Predicción (Opcional - CLI)**
```bash
python src/prediccion.py
```

**¿Qué hace?**
- Carga el modelo entrenado
- Analiza la red actual en Neo4j
- Muestra en consola las TOP 10 amenazas detectadas

**Resultado:** Reporte en terminal con probabilidades de crimen

---

### 4️⃣ **Visualizar en Dashboard 3D**
```bash
panel serve viz/dashboard_madrid_3d.py --show --port 5006
```

**¿Qué hace?**
- Inicia un servidor web con Panel
- Carga el modelo y datos usando `PredictionService`
- Visualiza:
  - Mapa 3D de Madrid con PyDeck
  - Personas (puntos azules)
  - Ubicaciones (puntos amarillos)
  - Al hacer clic en "EJECUTAR ANÁLISIS IA":
    - Arcos rojos (conexiones criminales predichas)
    - Puntos rojos (ubicaciones de crimen futuro)

**Acceso:** http://localhost:5006

---

## 🗂️ Estructura de Archivos

### Archivos Principales (en orden de uso)

```
minorty_report_policia/
│
├── src/
│   ├── city_generator.py          # 1️⃣ GENERA datos sintéticos (OSMnx → Neo4j)
│   ├── data_loader.py              # ✅ NUEVO - Carga datos reales (reemplaza etl_policial.py)
│   ├── entrenamiento_gan.py        # 2️⃣ ENTRENA modelo GAN
│   ├── prediccion.py               # 3️⃣ PREDICE en CLI
│   ├── prediction_service.py       # Servicio usado por dashboard
│   └── etl_policial.py             # ❌ OBSOLETO - NO USAR
│
├── viz/
│   └── dashboard_madrid_3d.py      # 4️⃣ VISUALIZA en web
│
├── models/
│   └── agente_precrime.pth         # Modelo entrenado (se genera en paso 2️⃣)
│
├── .env                             # Credenciales Neo4j
└── ORDEN_EJECUCION.md              # Este archivo
```

---

## 🔧 Configuración Previa

### Variables de Entorno (.env)

```env
NEO4J_URI=neo4j+ssc://xxxxx.databases.neo4j.io
NEO4J_PASSWORD=tu_password_aqui
```

### Dependencias

```bash
pip install -r requirements.txt
```

Principales:
- `neo4j` - Conexión con base de datos
- `torch` + `torch-geometric` - Deep Learning en grafos
- `osmnx` - Descarga de calles reales
- `panel` + `pydeck` - Dashboard interactivo

---

## 🐛 Troubleshooting

### ❌ Error: "Model not found"
**Problema:** No existe `models/agente_precrime.pth`  
**Solución:** Ejecuta primero el paso 2️⃣ `entrenamiento_gan.py`

### ❌ Error: "No module named 'etl_policial'"
**Problema:** Código antiguo que aún importa `etl_policial`  
**Solución:** Verifica que todos los archivos usen `data_loader.py`

### ❌ Dashboard muestra puntos fuera de Madrid
**Problema:** Coordenadas mal normalizadas  
**Solución:** Con el nuevo sistema esto NO debería pasar. Si ocurre:
1. Verifica que `city_generator.py` guarde lat/lon correctos en Neo4j
2. Comprueba que `data_loader.py` NO normalice las coordenadas

### ❌ Error: "Failed to connect to Neo4j"
**Problema:** Credenciales incorrectas o sin conexión  
**Solución:** 
1. Verifica el archivo `.env`
2. Comprueba que Neo4j Aura esté activo
3. Revisa el firewall/red

---

## 📊 Diferencias con el Sistema Anterior

| Aspecto | Sistema Antiguo | Sistema Nuevo |
|---------|----------------|---------------|
| **Coordenadas** | Normalizadas [0,1] | Reales [lat, lon] |
| **Carga de Datos** | `etl_policial.py` | `data_loader.py` |
| **Origen Datos** | Sintéticos aleatorios | OSMnx (calles reales) |
| **Visualización** | Requiere denormalización | Directo al mapa |
| **Modelo** | Entrenado con [0,1] | Entrenado con reales |

---

## 📝 Notas Adicionales

1. **Cada vez que ejecutes `city_generator.py`**, se generarán datos NUEVOS. El modelo debe ser re-entrenado.

2. **El entrenamiento es estocástico** - Cada ejecución dará resultados ligeramente diferentes.

3. **Para producción**, considera:
   - Cachear el grafo OSMnx (no descargarlo cada vez)
   - Usar GPU para entrenamiento más rápido
   - Implementar validación cruzada

4. **Coordenadas de Madrid**:
   - Latitud: ~40.30 a 40.55
   - Longitud: ~-3.85 a -3.50
   - Centro: 40.416775, -3.703790

---

## 🎯 Flujo Completo (Resumen Visual)

```
┌─────────────────────┐
│  1. city_generator  │
│  (OSMnx → Neo4j)    │
└──────────┬──────────┘
           │ Guarda coordenadas REALES
           ▼
┌─────────────────────┐
│  Neo4j Database     │
│  (Personas, Ubicac, │
│   Warnings + coords)│
└──────────┬──────────┘
           │
           │ Lee data_loader.py (SIN normalizar)
           ▼
┌─────────────────────┐
│ 2. entrenamiento_gan│
│  (Entrena modelo)   │
└──────────┬──────────┘
           │ Guarda
           ▼
┌─────────────────────┐
│ agente_precrime.pth │
└──────────┬──────────┘
           │
           ├──────────────────┬──────────────────┐
           ▼                  ▼                  ▼
    ┌────────────┐   ┌──────────────┐   ┌─────────────┐
    │3.prediccion│   │4. dashboard  │   │API (futuro) │
    │   (CLI)    │   │   (Web UI)   │   │             │
    └────────────┘   └──────────────┘   └─────────────┘
```

---

## 🚨 Importante: Re-entrenar el Modelo

⚠️ **Si ya tienes un `agente_precrime.pth` del sistema antiguo, DEBES re-entrenarlo** porque:

1. El modelo viejo espera coordenadas normalizadas [0, 1]
2. El nuevo sistema usa coordenadas reales [40.x, -3.x]
3. La escala es completamente diferente

**Solución:**
```bash
# Borra el modelo antiguo
rm models/agente_precrime.pth

# Re-genera datos
python src/city_generator.py

# Re-entrena
python src/entrenamiento_gan.py
```

---

¿Dudas? Revisa el código fuente o los comentarios en cada archivo.

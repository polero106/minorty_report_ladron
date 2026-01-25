# 🎯 GUÍA RÁPIDA - Sistema Minority Report

## 📌 ORDEN DE EJECUCIÓN (Nuevo Sistema)

### ✅ Sistema Actualizado - Coordenadas Reales

```
┌─────────────────────────────────────────────────────┐
│  PASO 1: Generar Ciudad Sintética                  │
│  Comando: python src/city_generator.py              │
│                                                     │
│  ✓ Descarga calles reales de Madrid (OSMnx)       │
│  ✓ Genera personas, ubicaciones y crímenes        │
│  ✓ Guarda en Neo4j con coordenadas REALES         │
│     (lat: ~40.4, lon: ~-3.7)                       │
└─────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│  PASO 2: Entrenar Modelo GAN                       │
│  Comando: python src/entrenamiento_gan.py           │
│                                                     │
│  ✓ Lee datos de Neo4j (data_loader.py)            │
│  ✓ Entrena red adversaria (300 epochs)            │
│  ✓ Guarda models/agente_precrime.pth              │
│     (modelo entrenado con coordenadas reales)      │
└─────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│  PASO 3: Visualizar Dashboard                      │
│  Comando: panel serve viz/dashboard_madrid_3d.py   │
│           --show --port 5006                        │
│                                                     │
│  ✓ Carga modelo y datos                           │
│  ✓ Muestra mapa 3D de Madrid                      │
│  ✓ Click en "EJECUTAR ANÁLISIS IA"                │
│  ✓ Ver predicciones de crímenes                   │
│                                                     │
│  URL: http://localhost:5006                        │
└─────────────────────────────────────────────────────┘
```

---

## ⚠️ IMPORTANTE: Cambio de Sistema

### ❌ ANTIGUO (NO USAR)
```python
from etl_policial import PoliceETL  # ❌ OBSOLETO
# Coordenadas normalizadas [0, 1]
```

### ✅ NUEVO (USAR)
```python
from data_loader import MadridDataLoader  # ✅ CORRECTO
# Coordenadas reales [40.x, -3.x]
```

---

## 🔧 Si tienes el modelo antiguo

```bash
# 1. Borra el modelo viejo
rm models/agente_precrime.pth

# 2. Regenera datos
python src/city_generator.py

# 3. Re-entrena modelo
python src/entrenamiento_gan.py

# 4. Lanza dashboard
panel serve viz/dashboard_madrid_3d.py --show
```

---

## 📂 Archivos Clave

| Archivo | Estado | Función |
|---------|--------|---------|
| `src/city_generator.py` | ✅ Usar | Genera datos con OSMnx |
| `src/data_loader.py` | ✅ Usar | Carga datos (coords reales) |
| `src/entrenamiento_gan.py` | ✅ Usar | Entrena modelo |
| `src/prediction_service.py` | ✅ Usar | Servicio de predicción |
| `viz/dashboard_madrid_3d.py` | ✅ Usar | Dashboard web |
| `src/etl_policial.py` | ❌ OBSOLETO | NO usar (normaliza coords) |

---

## 🐛 Problemas Comunes

### Dashboard muestra puntos fuera de Madrid
**Causa:** Modelo entrenado con sistema antiguo  
**Solución:** Re-entrena el modelo (ver arriba)

### Error "No module named 'etl_policial'"
**Causa:** Código no actualizado  
**Solución:** Cambiar import a `data_loader`

### Modelo no encuentra en models/
**Causa:** No has ejecutado entrenamiento  
**Solución:** Ejecuta paso 2

---

## 📚 Documentación Completa

Ver **ORDEN_EJECUCION.md** para detalles técnicos completos.

---

## 🎨 Visualización Esperada

En el dashboard verás:
- 🔵 Puntos azules = Personas
- 🟡 Puntos amarillos = Ubicaciones
- 🔴 Arcos rojos = Predicciones de crimen
- 🗺️ Mapa base = Madrid real (OSM)

---

**¿Dudas?** Revisa ORDEN_EJECUCION.md o los comentarios en el código.

# 🚀 Instrucciones de Ejecución - Minority Report Graph-GAN

## 📋 Orden de Ejecución

Sigue estos pasos en **orden secuencial** para ejecutar el sistema completo:

---

## **Paso 1: Configuración Inicial**

### 1.1 Instalar dependencias
```bash
pip install -r requirements.txt
```

### 1.2 Configurar variables de entorno
Crea un archivo `.env` en la raíz del proyecto:
```
NEO4J_URI=neo4j+ssc://tu_uri.neo4j.io
NEO4J_PASSWORD=tu_password
NEO4J_USER=neo4j
```

### 1.3 Verificar conexión con Neo4j
```bash
python -c "from neo4j import GraphDatabase; print('Neo4j conectado')"
```

---

## **Paso 2: Generar Datos Sintéticos**

Ejecuta el generador de datos para crear la base de datos de ejemplo en Madrid:

```bash
python src/city_generator.py
```

**Salida esperada:**
- Crea ~1000 personas, ~50 ubicaciones y ~200 warnings
- Genera conexiones realistas en el grafo de Madrid
- Llena la base de datos Neo4j

---

## **Paso 3: Entrenar el Modelo GAN (Primera ejecución)**

⚠️ **Solo si es la primera vez** o necesitas reentrenar. Si ya existe `models/agente_precrime.pth`, salta a Paso 4.

```bash
python src/entrenamiento_gan.py
```

**Salida esperada:**
- Entrena durante varias épocas
- Imprime pérdidas del generador y discriminador
- Guarda el modelo entrenado en `models/agente_precrime.pth`
- Muestra gráficos de convergencia

---

## **Paso 4: Visualizar en Dashboard 3D e Invocar IA**

Abre el dashboard interactivo de Madrid:

```bash
BOKEH_ALLOW_WS_ORIGIN=* python viz/dashboard_madrid_3d.py
```

**Detalles importantes:**
- La variable `BOKEH_ALLOW_WS_ORIGIN=*` permite WebSocket en cualquier origen
- El dashboard se ejecuta en `http://localhost:5006`
- **Abre tu navegador en:** `http://localhost:5006`

### 🚨 EJECUTAR ANÁLISIS IA DESDE EL DASHBOARD

Una vez abierto el dashboard:
1. Verás un **botón rojo** con el texto: **"🚨 EJECUTAR ANÁLISIS IA"**
2. **Haz clic en el botón** para invocar automáticamente `prediction_service`
3. El dashboard cargará:
   - Mapa 3D de Madrid con nodos de personas y ubicaciones
   - Colores por nivel de riesgo (Rojo=Alto, Amarillo=Ubicaciones, Azul=Seguro)
   - Red de conexiones entre sospechosos de alto riesgo
   - Heatmap de densidad criminal
   - Métricas en tiempo real:
     - Total de amenazas detectadas
     - Nivel de riesgo general
     - Personas de alto riesgo en monitoreo
     - Índice de amenaza (0-100)
     - Patrones temporales

**El botón automáticamente:**
- Carga el modelo desde `models/agente_precrime.pth`
- Ejecuta inferencia sobre todos los nodos del grafo
- Clasifica amenazas en tiempo real
- Actualiza todas las visualizaciones

---

## 📊 Flujo Completo Simplificado

```bash
# Paso 1: Generar datos
python src/city_generator.py

# Paso 2: Entrenar modelo (solo primera vez)
python src/entrenamiento_gan.py

# Paso 3: Abrir dashboard
BOKEH_ALLOW_WS_ORIGIN=* python viz/dashboard_madrid_3d.py

# Paso 4: En el navegador → Clic en botón "🚨 EJECUTAR ANÁLISIS IA"
```

---

## 🔄 Ejecución Sin Reentrenamiento

Si ya tienes el modelo entrenado (`models/agente_precrime.pth`):

```bash
# Abre directamente el dashboard
BOKEH_ALLOW_WS_ORIGEN=* python viz/dashboard_madrid_3d.py

# Luego haz clic en "🚨 EJECUTAR ANÁLISIS IA" en el navegador
```

---

## ⚠️ Notas Importantes

1. **Archivo crítico:** `models/agente_precrime.pth` debe existir antes de abrir el dashboard
   - Si no existe, ejecuta primero: `python src/entrenamiento_gan.py`
2. **Neo4j debe estar corriendo:** Verifica que la BD esté accesible
3. **Puertos requeridos:**
   - Neo4j: `7687` (SSL)
   - Dashboard: `5006` (Bokeh)
4. **Tiempo de ejecución:**
   - Generación datos: ~1-2 minutos
   - Entrenamiento: ~5-10 minutos
   - Dashboard inicio: ~5 segundos
   - Análisis IA (botón): ~2-3 minutos (primera ejecución), ~30s-1min (ejecutadas posteriores)

---

## 🛠️ Troubleshooting

### Error: "No module named 'etl_policial'"
```bash
pip install -r requirements.txt
# O reinicia el kernel de Python
```

### Error: "Connection refused" (Neo4j)
```bash
# Verifica que Neo4j esté corriendo:
docker ps | grep neo4j
# O inicia Neo4j en Docker:
docker-compose up -d
```

### Error: "BOKEH_ALLOW_WS_ORIGIN" no reconocido
```bash
# En Windows, usa:
set BOKEH_ALLOW_WS_ORIGIN=* && python viz/dashboard_madrid_3d.py

# O instala wscat:
pip install bokeh>=2.4.0
```

---

## 📝 Archivo de Modelo Entrenado

El archivo `models/agente_precrime.pth` contiene:
- `'encoder'`: Codificador GraphSAGE entrenado
- `'discriminator'`: Discriminador de policía
- `'generator'`: Generador de criminales

**Tamaño:** ~50-100 MB

---

## ✅ Checklist de Ejecución

- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] `.env` configurado con credenciales Neo4j
- [ ] Neo4j accesible y corriendo
- [ ] Paso 1: Datos sintéticos generados (`python src/city_generator.py`)
- [ ] Paso 2: Modelo GAN entrenado (ver `models/agente_precrime.pth` existe)
- [ ] Paso 3: Dashboard abierto en `http://localhost:5006`
- [ ] Paso 4: ✨ Haz clic en botón **"🚨 EJECUTAR ANÁLISIS IA"** en el dashboard

---

## 🎯 Próximos Pasos

Después de ejecutar el dashboard:
1. Explora las amenazas detectadas en el mapa
2. Examina personas/ubicaciones de alto riesgo
3. Ejecuta nuevamente `prediccion.py` para actualizar scores
4. Recarga el dashboard (`Ctrl+R` en el navegador)

---

**Última actualización:** Enero 2026

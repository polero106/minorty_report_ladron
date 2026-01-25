# Minority Report Graph-GAN

## 🕵️ Contexto del Proyecto
Inspirado en "Minority Report", este sistema implementa un Graph-GAN (Generative Adversarial Network sobre Grafos) utilizando **PyTorch Geometric (PyG)** y **Neo4j**.

El objetivo es modelar una "carrera armamentista" entre criminales y policías en una red social o de transacciones, donde:
- **CriminalNet (Generador):** Intenta crear nuevas conexiones ilícitas (aristas) que pasen desapercibidas.
- **PoliceNet (Discriminador):** Intenta clasificar nodos y conexiones como "Seguros" o "Criminales".

## 🚀 Inicio Rápido

### 📖 Ver [ORDEN_EJECUCION.md](ORDEN_EJECUCION.md) para instrucciones completas

**Resumen del flujo:**

1. **Generar ciudad sintética con datos reales de Madrid:**
   ```bash
   python src/city_generator.py
   ```

2. **Entrenar el modelo GAN:**
   ```bash
   python src/entrenamiento_gan.py
   ```

3. **Visualizar predicciones en dashboard 3D:**
   ```bash
   panel serve viz/dashboard_madrid_3d.py --show --port 5006
   ```

## ⚠️ Importante: Nuevo Sistema de Coordenadas

**A partir de la última actualización, el sistema trabaja con coordenadas REALES de Madrid** obtenidas de OpenStreetMap:

- ✅ **USAR:** `src/data_loader.py` - Carga datos con coordenadas reales
- ❌ **NO USAR:** `src/etl_policial.py` - OBSOLETO (normalizaba coordenadas)

Si tienes un modelo `agente_precrime.pth` antiguo, debes **re-entrenarlo** con el nuevo sistema.

## 🎯 Roles
### El Criminal (Generador)
- **Objetivo:** Generar ataques adversariales (nuevas aristas/nodos) que engañen a la policía.
- **Tecnología:** Graph Attention Network (GAT). Usa atención para identificar vulnerabilidades estructurales en el grafo y proponer conexiones fraudulentas.

### La Policía (Discriminador)
- **Objetivo:** Detectar anomalías y clasificar correctamente a los actores de la red.
- **Tecnología:** GraphSAGE. Observa vecindarios de nodos para determinar si un nodo es malicioso (1) o benigno (0).

## 🏗️ Arquitectura Técnica

### Stack Tecnológico
- **Base de Datos de Grafos:** Neo4j (Persistencia de datos y relaciones).
- **Deep Learning:** PyTorch & PyTorch Geometric.
- **Modelos:**
  - **Generador:** `GATConv` (Graph Attention Network).
  - **Discriminador:** `SAGEConv` (GraphSAGE).
- **Visualización:** NetworkX (local) y Neo4j Bloom/Browser (remoto).

### Flujo de Datos
1. **Neo4jConnector:** Extrae el subgrafo relevante mediante consultas Cypher.
2. **Preprocesamiento:** Conversión a objetos `torch_geometric.data.Data`.
3. **Entrenamiento Adversarial:**
   - **Paso 1:** PoliceNet entrena con datos reales (etiquetados).
   - **Paso 2:** CriminalNet genera conexiones falsas.
   - **Paso 3:** PoliceNet entrena para distinguir reales de falsas.
   - **Paso 4:** CriminalNet entrena para maximizar el error de PoliceNet.
4. **Persistencia:** Los scores de predicción y nuevas conexiones se escriben de vuelta en Neo4j.

## 🚀 Instrucciones de Uso (Notebook)

El proyecto está diseñado para ejecutarse en un Jupyter Notebook.

### Prerrequisitos
- Neo4j Desktop o AuraDB activo.
- Entorno Python con: `torch`, `torch_geometric`, `neo4j`, `pandas`, `networkx`.

### Estructura del Notebook
1. **Configuración:** Conexión a la BD y carga de librerías.
2. **Neo4jConnector:** Clase para lectura/escritura en grafos.
3. **Definición de Modelos:** `PoliceNet` (SAGE) y `CriminalNet` (GAT).
4. **Entrenamiento:** Bucle GAN alternado.
5. **Visualización:** Graficado de resultados y querys para exploración en Neo4j.

## 📊 Visualización
Para ver la "red de calor criminal" en Neo4j, usa queries que resalten nodos con alto `criminal_score` predicho por PoliceNet.
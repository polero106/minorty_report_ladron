
import torch
import sys
import os
import pandas as pd
from dotenv import load_dotenv

# Configuración de Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
models_dir = os.path.join(os.path.dirname(current_dir), 'models')

from etl_policial import PoliceETL
from entrenamiento_gan import PreCrimeModel

load_dotenv()

def predecir_amenazas():
    print("INICIANDO SISTEMA DE PREDICCIÓN PRE-CRIME...")
    
    # 1. Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    URI = os.getenv("NEO4J_URI", "neo4j+ssc://5d9c9334.databases.neo4j.io")
    AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "oTzaPYT99TgH-GM2APk0gcFlf9k16wrTcVOhtfmAyyA"))
    
    # 2. Cargar Datos Frescos
    print("   -> Cargando estado actual de la ciudad...")
    etl = PoliceETL(URI, AUTH)
    etl.load_nodes()
    etl.load_edges()
    data = etl.get_data().to(device)
    
    # 3. Cargar Modelo Entrenado
    model_path = os.path.join(models_dir, 'agente_precrime.pth')
    if not os.path.exists(model_path):
        print(f"Error: No se encuentra el modelo en {model_path}")
        return

    print("   -> Cargando cerebro del Agente...")
    model = PreCrimeModel(data.metadata()).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. Generar Embeddings Actuales
    with torch.no_grad():
        z_dict = model(data.x_dict, data.edge_index_dict)
    
    print("   -> Analizando patrones de riesgo...")
    
    # 5. Lógica de Inferencia: Filtrar Candidatos
    # Buscamos Personas con Risk Seed > 0.8 y Ubicaciones con Peligrosidad > 0.7
    
    # Obtenemos tensores de features originales (en CPU)
    x_persona = data['Persona'].x.cpu()
    x_ubicacion = data['Ubicacion'].x.cpu()
    
    # Indices de alto riesgo (Feature 0 es risk/danger)
    high_risk_persons_idx = torch.where(x_persona[:, 0] > 0.8)[0].to(device)
    high_danger_locs_idx = torch.where(x_ubicacion[:, 0] > 0.7)[0].to(device)
    
    print(f"      - Personas bajo vigilancia extrema: {len(high_risk_persons_idx)}")
    print(f"      - Zonas rojas activas: {len(high_danger_locs_idx)}")
    
    if len(high_risk_persons_idx) == 0 or len(high_danger_locs_idx) == 0:
        print("No hay suficientes amenazas activas para generar predicciones críticas.")
        return

    # 6. Calcular Probabilidades (Producto Cruzado Filtrado)
    predictions = []
    
    # Para no explotar la memoria, iteramos sobre las personas sospechosas
    # y calculamos score contra todas las zonas peligrosas
    z_person_subset = z_dict['Persona'][high_risk_persons_idx]
    z_loc_subset = z_dict['Ubicacion'][high_danger_locs_idx]
    
    # Matriz de [Num_Personas_Suspect, Num_Zonas_Danger]
    scores = torch.matmul(z_person_subset, z_loc_subset.t())
    probs = torch.sigmoid(scores)
    
    # 7. Extraer Top Amenazas
    # Aplanamos y ordenamos
    flat_probs = probs.flatten()
    top_k = min(5, len(flat_probs))
    top_values, top_indices_flat = torch.topk(flat_probs, k=top_k)
    
    # Recuperar índices originales
    # Indice aplanado -> (fila, col)
    row_indices = top_indices_flat // probs.size(1)
    col_indices = top_indices_flat % probs.size(1)
    
    print("\nTOP 5 AMENAZAS DETECTADAS POR LA IA")
    print("===========================================")
    
    for i in range(top_k):
        p_real_idx = high_risk_persons_idx[row_indices[i]].item()
        u_real_idx = high_danger_locs_idx[col_indices[i]].item()
        prob = top_values[i].item()
        
        # Recuperar IDs originales de Neo4j (guardados en ETL)
        neo4j_p_id = data['Persona'].original_ids[p_real_idx].item()
        neo4j_u_id = data['Ubicacion'].original_ids[u_real_idx].item()
        
        # Recuperamos datos raw para mostrar (Lat/Lon)
        # Nota: aquí están normalizados, en un caso real haríamos query a Neo4j por ID para detalles
        # pero usaremos los features normalizados como referencia rápida explicativa
        risk_score = x_persona[p_real_idx, 0].item()
        danger_score = x_ubicacion[u_real_idx, 0].item()
        
        print(f"#{i+1}: Probabilidad de Crimen: {prob*100:.2f}%")
        print(f"    - Sujeto ID: {neo4j_p_id} (Nivel Riesgo: {risk_score:.2f})")
        print(f"    - Objetivo ID: {neo4j_u_id} (Peligrosidad Zona: {danger_score:.2f})")
        print("-------------------------------------------")

if __name__ == "__main__":
    predecir_amenazas()

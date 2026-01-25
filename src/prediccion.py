import torch
import torch.nn as nn
import sys
import os
import pandas as pd
from dotenv import load_dotenv

# Configuración de rutas para importar módulos locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from torch_geometric.nn import SAGEConv, HeteroConv
from data_loader import MadridDataLoader

load_dotenv()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# 1. REPLICAMOS LA ARQUITECTURA (Debe ser idéntica a entrenamiento_gan.py)
# ==============================================================================

class GraphEncoder(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), 32) 
            for edge_type in metadata[1]
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), 32) 
            for edge_type in metadata[1]
        }, aggr='sum')
        
    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict

class PoliceDiscriminator(nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, person_emb, location_emb):
        x = torch.cat([person_emb, location_emb], dim=1)
        return self.net(x)

# ==============================================================================
# 2. LÓGICA DE PREDICCIÓN (PRE-CRIME)
# ==============================================================================

def ejecutar_prediccion():
    print("\n🕵️  INICIANDO SISTEMA DE PREDICCIÓN MINORITY REPORT...")
    print("=====================================================")

    # 1. Cargar Datos de la Ciudad (Neo4j)
    URI = os.getenv("NEO4J_URI", "neo4j+ssc://5d9c9334.databases.neo4j.io")
    AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "oTzaPYT99TgH-GM2APk0gcFlf9k16wrTcVOhtfmAyyA"))
    
    print("   -> Conectando con la base de datos...")
    loader = MadridDataLoader(URI, AUTH)
    loader.load_nodes()
    loader.load_edges()
    data = loader.get_data().to(device)
    loader.close()
    
    # 2. Cargar el Modelo Entrenado (Discriminador + Encoder)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(models_dir, 'agente_precrime.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ Error: No encuentro el modelo en {model_path}")
        return

    print(f"   -> Cargando cerebro del agente desde: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Instanciar y cargar pesos
    encoder = GraphEncoder(data.metadata()).to(device)
    discriminator = PoliceDiscriminator().to(device)
    
    encoder.load_state_dict(checkpoint['encoder'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    
    encoder.eval()
    discriminator.eval()

    # 3. Generar Perfiles (Embeddings)
    print("   -> Analizando perfiles psicológicos y geoespaciales...")
    with torch.no_grad():
        z_dict = encoder(data.x_dict, data.edge_index_dict)
        z_personas = z_dict['Persona']
        z_ubicaciones = z_dict['Ubicacion']

    # 4. Filtrado Inteligente (Para no calcular millones de combinaciones)
    # Buscamos: Personas con Risk Seed > 0.6  Y  Ubicaciones con Peligrosidad > 0.5
    
    # Accedemos a los features originales (CPU) para filtrar
    # Asumimos que feature 0 es risk_seed y peligrosidad respectivamente
    x_personas_cpu = data['Persona'].x.cpu()
    x_ubicaciones_cpu = data['Ubicacion'].x.cpu()
    
    high_risk_indices = torch.where(x_personas_cpu[:, 0] > 0.6)[0].to(device)
    high_danger_indices = torch.where(x_ubicaciones_cpu[:, 0] > 0.5)[0].to(device)
    
    print(f"   -> Objetivos bajo vigilancia: {len(high_risk_indices)} Personas | {len(high_danger_indices)} Zonas")
    
    if len(high_risk_indices) == 0 or len(high_danger_indices) == 0:
        print("   ⚠️ No hay suficientes amenazas de alto nivel para generar un reporte.")
        return

    # 5. Predicción Masiva (Batch)
    reporte = []
    
    print("   -> Calculando probabilidades de crimen futuro...")
    with torch.no_grad():
        # Iteramos para predecir
        # (Hacemos un bucle simple para no saturar memoria si hay muchos)
        for p_idx in high_risk_indices:
            # Vector de la persona (repetido N veces, una por cada ubicación peligrosa)
            p_emb = z_personas[p_idx].repeat(len(high_danger_indices), 1)
            # Vectores de todas las ubicaciones peligrosas
            u_emb = z_ubicaciones[high_danger_indices]
            
            # El Discriminador opina: ¿Es esto un crimen Real (1) o Falso (0)?
            # Una probabilidad alta significa que el patrón encaja con un crimen real
            probs = discriminator(p_emb, u_emb).flatten()
            
            # Guardamos los que superen el 80% de probabilidad
            mask = probs > 0.80
            
            if mask.any():
                indices_validos = torch.where(mask)[0]
                for idx in indices_validos:
                    u_real_idx = high_danger_indices[idx]
                    prob = probs[idx].item()
                    
                    # Recuperar datos para el reporte
                    risk_val = x_personas_cpu[p_idx, 0].item()
                    danger_val = x_ubicaciones_cpu[u_real_idx, 0].item()
                    
                    reporte.append({
                        "PROBABILIDAD": prob,
                        "ID_SUJETO": int(p_idx.item()), # ID interno del tensor
                        "RIESGO_SUJETO": risk_val,
                        "ID_UBICACION": int(u_real_idx.item()),
                        "PELIGROSIDAD_ZONA": danger_val
                    })

    # 6. Mostrar Resultados
    if not reporte:
        print("\n✅ No se detectaron amenazas inminentes con >80% de probabilidad.")
    else:
        # Ordenar por probabilidad descendente
        df = pd.DataFrame(reporte)
        df = df.sort_values(by="PROBABILIDAD", ascending=False).head(10)
        
        print("\n🚨  ALERTA TEMPRANA: TOP 10 AMENAZAS DETECTADAS  🚨")
        print("=================================================================================")
        print(f"{'PROB %':<10} | {'ID SUJETO':<12} | {'RIESGO':<10} | {'ID ZONA':<10} | {'PELIGROSIDAD':<12}")
        print("-" * 75)
        for _, row in df.iterrows():
            print(f"{row['PROBABILIDAD']*100:.2f}%     | P_{int(row['ID_SUJETO']):04d}       | {row['RIESGO_SUJETO']:.2f}       | U_{int(row['ID_UBICACION']):04d}     | {row['PELIGROSIDAD_ZONA']:.2f}")
        print("=================================================================================")

if __name__ == "__main__":
    ejecutar_prediccion()
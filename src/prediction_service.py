import torch
import torch.nn as nn
import sys
import os
import pandas as pd
from dotenv import load_dotenv

# Ajuste de path para importar módulos hermanos si es necesario
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importamos las capas de grafo necesarias
from torch_geometric.nn import SAGEConv, HeteroConv
from etl_policial import PoliceETL
from city_generator import CityGenerator

load_dotenv()

# ==============================================================================
# DEFINICIÓN DE ARQUITECTURA (Necesaria para cargar el modelo .pth)
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
# SERVICIO DE PREDICCIÓN
# ==============================================================================

class PredictionService:
    def __init__(self):
        print("Initializing Prediction Service...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Configuración DB
        self.uri = os.getenv("NEO4J_URI", "neo4j+ssc://c6226feb.databases.neo4j.io")
        self.auth = ("neo4j", os.getenv("NEO4J_PASSWORD", "8G7YN9W2V7Y_RQDCqWTHrryWd-G8GnNIF3ep9vslp6k"))
        
        # 2. Regenerar datos con city_generator
        print("   -> Regenerando datos sintéticos...")
        try:
            gen = CityGenerator(self.uri, self.auth)
            personas, ubicaciones, warnings = gen.generate_data(num_personas=500, num_ubicaciones=15)
            gen.save_to_neo4j(personas, ubicaciones, warnings)
            gen.close()
            print("   ✅ Datos regenerados exitosamente")
        except Exception as e:
            print(f"   ⚠️  Aviso al regenerar datos: {e}")
            print("   Continuando con datos existentes en Neo4j...")
        
        # 3. Cargar Datos del Grafo (Snapshot actual)
        print("   -> Loading graph data from Neo4j...")
        self.etl = PoliceETL(self.uri, self.auth)
        self.etl.load_nodes()
        self.etl.load_edges()
        self.data = self.etl.get_data().to(self.device)
        
        # 3. Cargar Modelos
        print("   -> Loading pre-trained models...")
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        model_path = os.path.join(models_dir, 'agente_precrime.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.encoder = GraphEncoder(self.data.metadata()).to(self.device)
        self.discriminator = PoliceDiscriminator().to(self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        
        self.encoder.eval()
        self.discriminator.eval()
        
    def predict_threats(self, risk_threshold=0.6, danger_threshold=0.5):
        """
        Ejecuta la inferencia y devuelve un DataFrame con las amenazas detectadas.
        """
        print("PredictionService: Running inference...")
        
        # 1. Generar Embeddings
        with torch.no_grad():
            z_dict = self.encoder(self.data.x_dict, self.data.edge_index_dict)
            z_personas = z_dict['Persona']
            z_ubicaciones = z_dict['Ubicacion']

        # 2. Obtener datos crudos para filtrado y coordenadas (CPU)
        x_personas_cpu = self.data['Persona'].x.cpu()
        x_ubicaciones_cpu = self.data['Ubicacion'].x.cpu()
        
        # 3. Filtrar candidatos (Sujetos de riesgo y Zonas peligrosas)
        # Asumimos Feature 0 = Risk/Danger
        high_risk_indices = torch.where(x_personas_cpu[:, 0] > risk_threshold)[0].to(self.device)
        high_danger_indices = torch.where(x_ubicaciones_cpu[:, 0] > danger_threshold)[0].to(self.device)
        
        print(f"   -> Candidatos filtrados: {len(high_risk_indices)} personas de riesgo, {len(high_danger_indices)} zonas peligrosas")
        
        if len(high_risk_indices) == 0 or len(high_danger_indices) == 0:
            print(f"   ⚠️ No hay suficientes candidatos (risk_threshold={risk_threshold}, danger_threshold={danger_threshold})")
            return pd.DataFrame() # Retorno vacío
            
        predictions = []

        # 4. Inferencia Cruzada
        with torch.no_grad():
            for p_idx in high_risk_indices:
                # Repetir vector persona para compararlo con todas las ubicaciones
                p_emb = z_personas[p_idx].repeat(len(high_danger_indices), 1)
                u_emb = z_ubicaciones[high_danger_indices]
                
                # Discriminador opina
                probs = self.discriminator(p_emb, u_emb).flatten()
                
                # Filtrar solo prob > 0.5 (Umbral más bajo para obtener más predicciones)
                mask = probs > 0.50
                
                if mask.any():
                    indices_validos = torch.where(mask)[0]
                    for idx in indices_validos:
                        u_real_idx = high_danger_indices[idx]
                        prob = probs[idx].item()
                        
                        # Recuperar coordenadas normalizadas y des-normalizarlas
                        # Features: [risk/danger, lat_norm, lon_norm]
                        LAT_MIN, LAT_MAX = 40.30, 40.55
                        LON_MIN, LON_MAX = -3.85, -3.50
                        
                        lat_norm_p = x_personas_cpu[p_idx, 1].item()
                        lon_norm_p = x_personas_cpu[p_idx, 2].item()
                        lat_norm_u = x_ubicaciones_cpu[u_real_idx, 1].item()
                        lon_norm_u = x_ubicaciones_cpu[u_real_idx, 2].item()
                        
                        # Des-normalizar
                        lat_sujeto = lat_norm_p * (LAT_MAX - LAT_MIN) + LAT_MIN
                        lon_sujeto = lon_norm_p * (LON_MAX - LON_MIN) + LON_MIN
                        lat_ubicacion = lat_norm_u * (LAT_MAX - LAT_MIN) + LAT_MIN
                        lon_ubicacion = lon_norm_u * (LON_MAX - LON_MIN) + LON_MIN
                        
                        predictions.append({
                            "id_sujeto": f"P_{p_idx.item()}",
                            "riesgo_sujeto": x_personas_cpu[p_idx, 0].item(),
                            "lat_sujeto": lat_sujeto,
                            "lon_sujeto": lon_sujeto,
                            
                            "id_ubicacion": f"U_{u_real_idx.item()}",
                            "peligrosidad_zona": x_ubicaciones_cpu[u_real_idx, 0].item(),
                            "lat_ubicacion": lat_ubicacion,
                            "lon_ubicacion": lon_ubicacion,
                            
                            "probabilidad": prob
                        })

        if not predictions:
            print("   ⚠️ No se generaron predicciones (umbral de probabilidad no alcanzado)")
            return pd.DataFrame()
        
        print(f"   ✅ {len(predictions)} amenazas detectadas")
        return pd.DataFrame(predictions)
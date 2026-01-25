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
# Usamos PoliceETL SOLO para cargar los datos de Neo4j
from etl_policial import PoliceETL

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
        self.uri = os.getenv("NEO4J_URI", "neo4j+ssc://5d9c9334.databases.neo4j.io")
        self.auth = ("neo4j", os.getenv("NEO4J_PASSWORD", "oTzaPYT99TgH-GM2APk0gcFlf9k16wrTcVOhtfmAyyA"))
        
        # 2. Cargar Datos del Grafo (Snapshot actual)
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
        
        if len(high_risk_indices) == 0 or len(high_danger_indices) == 0:
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
                
                # Filtrar solo prob > 0.8 (Alta certeza)
                mask = probs > 0.80
                
                if mask.any():
                    indices_validos = torch.where(mask)[0]
                    for idx in indices_validos:
                        u_real_idx = high_danger_indices[idx]
                        prob = probs[idx].item()
                        
                        # Recuperar IDs originales y Coordenadas
                        # Nota: El ETL guarda mapas de ID originales, pero aquí usaremos indices
                        # Recuperamos Lat/Lon de los features normalizados (indices 1 y 2)
                        # OJO: Para visualización real necesitaríamos des-normalizar si las guardamos normalizadas.
                        # Asumiremos que el ETL las tiene en raw en alguna parte o que data.x tiene lat/lon usables.
                        # Para este ejemplo, usaremos las lat/lon que el ETL cargó en los tensores.
                        
                        predictions.append({
                            "id_sujeto": f"P_{p_idx.item()}",
                            "riesgo_sujeto": x_personas_cpu[p_idx, 0].item(),
                            "lat_sujeto": x_personas_cpu[p_idx, 1].item(), # Asumiendo idx 1 es Lat
                            "lon_sujeto": x_personas_cpu[p_idx, 2].item(), # Asumiendo idx 2 es Lon
                            
                            "id_ubicacion": f"U_{u_real_idx.item()}",
                            "peligrosidad_zona": x_ubicaciones_cpu[u_real_idx, 0].item(),
                            "lat_ubicacion": x_ubicaciones_cpu[u_real_idx, 1].item(),
                            "lon_ubicacion": x_ubicaciones_cpu[u_real_idx, 2].item(),
                            
                            "probabilidad": prob
                        })

        if not predictions:
            return pd.DataFrame()
            
        return pd.DataFrame(predictions)
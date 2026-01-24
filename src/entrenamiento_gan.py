
import torch
import torch.nn.functional as F
import sys
import os

# Permitir importaciones locales (si se ejecuta como script)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torch_geometric.nn import SAGEConv, HeteroConv
from etl_policial import PoliceETL
from dotenv import load_dotenv

load_dotenv()

# Detectar GPU o CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PreCrimeModel(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        # Input=3. Usamos HeteroConv para manejar relaciones explícitamente
        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), 16) 
            for edge_type in metadata[1]
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), 16) 
            for edge_type in metadata[1]
        }, aggr='sum')
        
    def forward(self, x_dict, edge_index_dict):
        # Capa 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        # Capa 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        return x_dict

    def predict_link(self, z_dict, edge_label_index):
        # Extraemos vectores de las Personas y Ubicaciones que queremos comparar
        row, col = edge_label_index
        z_person = z_dict['Persona'][row]
        z_location = z_dict['Ubicacion'][col]
        
        # Calculamos afinidad (Producto Punto)
        score = (z_person * z_location).sum(dim=-1)
        return torch.sigmoid(score)

def entrenar_policia():
    print("Iniciando entrenamiento del Sistema Pre-Crimen Geoespacial...")
    
    # 1. CARGAR DATOS (ETL)
    URI = os.getenv("NEO4J_URI", "neo4j+ssc://5d9c9334.databases.neo4j.io")
    AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "oTzaPYT99TgH-GM2APk0gcFlf9k16wrTcVOhtfmAyyA"))
    
    etl = PoliceETL(URI, AUTH)
    etl.load_nodes()
    etl.load_edges()
    data = etl.get_data().to(device)

    # 2. DEFINIR OBJETIVO DE ENTRENAMIENTO
    print("   -> Construyendo ground truth (Persona -> Warning -> Ubicacion)...")
    
    # Buscamos caminos: (P)-[COMETIO]->(W)-[OCURRIO_EN]->(U)
    edge_index_cometio = data['Persona', 'COMETIO', 'Warning'].edge_index
    edge_index_ocurrio = data['Warning', 'OCURRIO_EN', 'Ubicacion'].edge_index
    
    # Mapear Warning -> Ubicacion
    w_to_u = {w.item(): u.item() for w, u in zip(edge_index_ocurrio[0], edge_index_ocurrio[1])}
    
    sources = []
    targets = []
    for i in range(edge_index_cometio.size(1)):
        p_idx = edge_index_cometio[0, i].item()
        w_idx = edge_index_cometio[1, i].item()
        if w_idx in w_to_u:
            sources.append(p_idx)
            targets.append(w_to_u[w_idx])
            
    # Casos Reales (Positivos)
    pos_edge_index = torch.tensor([sources, targets], dtype=torch.long, device=device)
    print(f"   -> Casos positivos identificados para entrenamiento: {pos_edge_index.size(1)}")

    if pos_edge_index.size(1) == 0:
        print("ERROR CRÍTICO: No se encontraron relaciones indirectas para entrenar.")
        return

    # 3. INICIALIZAR CONTENEDOR DE MODELOS
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'agente_precrime.pth')

    model = PreCrimeModel(data.metadata()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 4. BUCLE DE ENTRENAMIENTO
    model.train()
    print("   -> Entrenando Red Neuronal de Grafos (SAGEConv)...")
    for epoch in range(1, 101):
        optimizer.zero_grad()
        
        # A. Generar Embeddings
        z_dict = model(data.x_dict, data.edge_index_dict)
        
        # B. Predicción Positivos
        pos_pred = model.predict_link(z_dict, pos_edge_index)
        loss_pos = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        
        # C. Predicción Negativos (Aleatorios)
        neg_edge_index = torch.randint(0, data['Ubicacion'].num_nodes, (2, pos_edge_index.size(1)), device=device)
        neg_edge_index[0] = torch.randint(0, data['Persona'].num_nodes, (pos_edge_index.size(1),), device=device)
        
        neg_pred = model.predict_link(z_dict, neg_edge_index)
        loss_neg = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        
        loss = loss_pos + loss_neg
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"      Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    # 5. GUARDAR MODELO
    torch.save(model.state_dict(), model_path)
    print(f"\nModelo guardado exitosamente en: {model_path}")

if __name__ == "__main__":
    entrenar_policia()
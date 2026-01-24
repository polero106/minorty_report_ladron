import torch
import torch.nn.functional as F
import sys
import os

# Permitir importaciones locales (si se ejecuta como script)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torch_geometric.nn import SAGEConv, to_hetero
from etl_policial import PoliceETL

# Detectar GPU o CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PreCrimeModel(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        # Input=4 porque ahora tenemos [risk/peligro, x, y, z]
        self.conv1 = SAGEConv(4, 16) 
        self.conv2 = SAGEConv(16, 16)
        
        # Convertimos a GNN Heterogénea
        self.gnn = to_hetero(torch.nn.Sequential(
            self.conv1, torch.nn.ReLU(),
            self.conv2, torch.nn.ReLU()
        ), metadata, aggr='sum')
        
        # Clasificador final: Combina embeddings de Persona y Ubicación
        self.classifier = torch.nn.Linear(16, 1)

    def forward(self, x_dict, edge_index_dict):
        # Generar "perfiles criminales" (embeddings)
        return self.gnn(x_dict, edge_index_dict)

    def predict_link(self, z_dict, edge_label_index):
        # Extraemos vectores de las Personas y Ubicaciones que queremos comparar
        row, col = edge_label_index
        z_person = z_dict['Persona'][row]
        z_location = z_dict['Ubicacion'][col]
        
        # Calculamos afinidad (Producto Punto)
        score = (z_person * z_location).sum(dim=-1)
        return torch.sigmoid(score)

def entrenar_policia():
    print("🚨 Iniciando entrenamiento del Sistema Pre-Crimen 3D...")
    
    # 1. CARGAR DATOS (ETL)
    # ¡Asegúrate de poner tu URI y Contraseña correctas aquí!
    URI = "neo4j+ssc://xxxxxxxx.databases.neo4j.io" 
    AUTH = ("neo4j", "tu_password")
    
    etl = PoliceETL(URI, AUTH)
    etl.load_nodes()
    etl.load_edges()
    data = etl.get_data().to(device)

    # 2. DEFINIR OBJETIVO DE ENTRENAMIENTO
    # Como no tenemos una relación directa "CRIMEN_FUTURO", la simulamos para entrenar.
    # Lógica: Si una Persona cometió un Warning que ocurrió en una Ubicación,
    # existe una "conexión criminal" entre esa Persona y esa Ubicación.
    print("   -> Construyendo historial delictivo para entrenamiento...")
    
    # Buscamos caminos: (P)-[COMETIO]->(W)-[OCURRIO_EN]->(U)
    # (Esto es álgebra de índices con PyTorch)
    edge_index_cometio = data['Persona', 'COMETIO', 'Warning'].edge_index
    edge_index_ocurrio = data['Warning', 'OCURRIO_EN', 'Ubicacion'].edge_index
    
    # Cruzamos los índices para conectar Persona -> Ubicacion directamente
    # Simplicación: Usamos un diccionario para mapear Warning -> Ubicacion
    w_to_u = {w.item(): u.item() for w, u in zip(edge_index_ocurrio[0], edge_index_ocurrio[1])}
    
    sources = []
    targets = []
    for i in range(edge_index_cometio.size(1)):
        p_idx = edge_index_cometio[0, i].item()
        w_idx = edge_index_cometio[1, i].item()
        if w_idx in w_to_u:
            sources.append(p_idx)
            targets.append(w_to_u[w_idx])
            
    # Estos son nuestros "Casos Reales" para entrenar
    pos_edge_index = torch.tensor([sources, targets], dtype=torch.long, device=device)
    print(f"   -> Casos reales identificados: {pos_edge_index.size(1)}")

    # 3. INICIALIZAR MODELO
    model = PreCrimeModel(data.metadata()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 4. BUCLE DE ENTRENAMIENTO
    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        
        # A. Generar Embeddings
        z_dict = model(data.x_dict, data.edge_index_dict)
        
        # B. Predicción sobre casos REALES (Debe dar 1)
        pos_pred = model.predict_link(z_dict, pos_edge_index)
        loss_pos = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        
        # C. Predicción sobre casos FALSOS/ALEATORIOS (Debe dar 0)
        # Generamos pares al azar Persona-Ubicacion que no sean criminales
        neg_edge_index = torch.randint(0, data['Ubicacion'].num_nodes, (2, pos_edge_index.size(1)), device=device)
        # Ajustamos rango del source a num_personas
        neg_edge_index[0] = torch.randint(0, data['Persona'].num_nodes, (pos_edge_index.size(1),), device=device)
        
        neg_pred = model.predict_link(z_dict, neg_edge_index)
        loss_neg = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        
        loss = loss_pos + loss_neg
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} (Accuracy en training subiendo...)")

    # 5. GUARDAR AGENTE
    torch.save(model.state_dict(), 'agente_policia_3d.pth')
    print("\n✅ Agente Pre-Crimen 3D entrenado y guardado.")

if __name__ == "__main__":
    entrenar_policia()
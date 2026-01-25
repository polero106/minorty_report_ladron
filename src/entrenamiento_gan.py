
import torch
import torch.nn as nn
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

# ----------------------------------------------------------------------
# 1. Feature Extractor (GNN) - Para obtener embeddings iniciales del Grafo
# ----------------------------------------------------------------------
class GraphEncoder(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        # Usamos HeteroConv para manejar relaciones explícitamente y generar embeddings ricos
        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), 32) 
            for edge_type in metadata[1]
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), 32) 
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

# ----------------------------------------------------------------------
# 2. GENERADOR (El Criminal)
# ----------------------------------------------------------------------
class CriminalGenerator(nn.Module):
    def __init__(self, embedding_dim=32, noise_dim=16):
        super().__init__()
        # Input: Embedding de Persona (32) + Ruido (16)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + noise_dim, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, embedding_dim) # Output: Fake Location Embedding
        )
        
    def forward(self, person_emb, z_noise):
        # Concatenar Persona + Ruido
        x = torch.cat([person_emb, z_noise], dim=1)
        return self.net(x)

# ----------------------------------------------------------------------
# 3. DISCRIMINADOR (El Policía)
# ----------------------------------------------------------------------
class PoliceDiscriminator(nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()
        # Input: Embedding de Persona (32) + Embedding de Ubicación (32)
        # Queremos saber si este par (Persona, Ubicación) es una conexión REAL o FAKE
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid() # Probabilidad de ser Real
        )
        
    def forward(self, person_emb, location_emb):
        x = torch.cat([person_emb, location_emb], dim=1)
        return self.net(x)

# ----------------------------------------------------------------------
# 4. BUCLE DE ENTRENAMIENTO ADVERSARIO
# ----------------------------------------------------------------------
def entrenar_policia():
    print("Iniciando entrenamiento ADVERSARIO (GAN) - Pre-Crimen...")
    
    # --- CARGAR DATOS ---
    URI = os.getenv("NEO4J_URI", "neo4j+ssc://5d9c9334.databases.neo4j.io")
    AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "oTzaPYT99TgH-GM2APk0gcFlf9k16wrTcVOhtfmAyyA"))
    
    etl = PoliceETL(URI, AUTH)
    etl.load_nodes()
    etl.load_edges()
    data = etl.get_data().to(device)

    # --- PREPARAR GROUND TRUTH (CRÍMENES REALES) ---
    print("   -> Construyendo dataset de crímenes reales...")
    edge_index_cometio = data['Persona', 'COMETIO', 'Warning'].edge_index
    edge_index_ocurrio = data['Warning', 'OCURRIO_EN', 'Ubicacion'].edge_index
    
    # Mapeo rápido Warning -> Ubicacion
    w_to_u = {w.item(): u.item() for w, u in zip(edge_index_ocurrio[0], edge_index_ocurrio[1])}
    
    real_pairs = []
    for i in range(edge_index_cometio.size(1)):
        p_idx = edge_index_cometio[0, i].item()
        w_idx = edge_index_cometio[1, i].item()
        if w_idx in w_to_u:
            real_pairs.append([p_idx, w_to_u[w_idx]])
            
    real_pairs = torch.tensor(real_pairs, dtype=torch.long, device=device)
    print(f"   -> {len(real_pairs)} crímenes reales para entrenar.")
    
    if len(real_pairs) == 0:
        print("ERROR: No hay datos suficientes.")
        return

    # --- INICIALIZAR MODELOS ---
    encoder = GraphEncoder(data.metadata()).to(device)
    generator = CriminalGenerator().to(device)
    discriminator = PoliceDiscriminator().to(device)
    
    # Optimizadores separados
    # El encoder entrena junto con ambos o podría solo actualizarse con el Discriminador. 
    # Para simplificar y dar estabilidad, actualizaremos el encoder junto con el discriminador para aprender mejores representaciones del grafo real.
    opt_d = torch.optim.Adam(list(discriminator.parameters()) + list(encoder.parameters()), lr=0.0005)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.001) # El criminal aprende más rápido
    
    criterion = nn.BCELoss()
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, 'agente_precrime.pth')

    print("   -> Comenzando la Batalla Adversaria (Policía vs Criminal)...")
    
    # Embedding Dimension Params
    EMBED_DIM = 32
    NOISE_DIM = 16
    BATCH_SIZE = len(real_pairs) # Full batch por simplicidad
    
    for epoch in range(1, 301):
        # ---------------------
        # FASE 0: Obtener Embeddings Actuales del Grafo
        # ---------------------
        # Pass forward para obtener representaciones latentes de Personas y Ubicaciones
        z_dict = encoder(data.x_dict, data.edge_index_dict)
        z_personas = z_dict['Persona']   # [Num_Personas, 32]
        z_ubicaciones = z_dict['Ubicacion'] # [Num_Ubicaciones, 32]
        
        # Seleccionar embeddings de los pares reales
        real_p_emb = z_personas[real_pairs[:, 0]]
        real_u_emb = z_ubicaciones[real_pairs[:, 1]]
        
        # ---------------------
        # FASE A: Entrenar DISCRIMINADOR (Policía)
        # ---------------------
        # Maximizar log(D(x)) + log(1 - D(G(z)))
        opt_d.zero_grad()
        
        # 1. Real Data (Label ~0.9 para Smoothing)
        pred_real = discriminator(real_p_emb, real_u_emb)
        label_real = torch.full_like(pred_real, 0.9) 
        loss_real = criterion(pred_real, label_real)
        
        # 2. Fake Data (Criminal genera Ubicación falsa para una Persona real)
        # Tomamos las mismas personas reales para ver dónde cometerían otro crimen (falso)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
        fake_u_emb = generator(real_p_emb.detach(), noise) # Detach para no actualizar G aquí
        
        pred_fake = discriminator(real_p_emb.detach(), fake_u_emb)
        label_fake = torch.zeros_like(pred_fake)
        loss_fake = criterion(pred_fake, label_fake)
        
        loss_d = (loss_real + loss_fake) / 2
        loss_d.backward()
        opt_d.step()
        
        # ---------------------
        # FASE B: Entrenar GENERADOR (Criminal)
        # ---------------------
        # Maximizar log(D(G(z))) -> Minimizar log(1 - D(G(z)))
        # Queremos que el discriminador diga "1" (Real) para nuestros Fakes
        opt_g.zero_grad()
        
        # Recalculamos fakes (ahora queremos gradiente en G)
        # Nota: Usamos las mismas personas. 
        # Importante: Volvemos a generar ruido para independencia estocástica o usamos el mismo. Usaremos nuevo.
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
        # Necesitamos embeddings frescos si el encoder cambió? 
        # En este paso, el encoder ya se actualizó en Fase A. 
        # Podríamos re-ejecutar encoder() si queremos ser muy precisos, pero es costoso. 
        # Asumiremos que el cambio es pequeño o usamos .detach() en encoder para G.
        # Generalmente G toma input fijo o del encoder.
        # Para que el gradiente fluya SOLO a G y no a Encoder (que es del policía), detachamos input de G.
        
        fake_u_emb_g = generator(real_p_emb.detach(), noise)
        pred_fake_g = discriminator(real_p_emb.detach(), fake_u_emb_g)
        
        # El Criminal quiere engañar al policía -> Label Target = 1
        label_g = torch.ones_like(pred_fake_g)
        loss_g = criterion(pred_fake_g, label_g)
        
        loss_g.backward()
        opt_g.step()
        
        # ---------------------
        # LOGS
        # ---------------------
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss D (Policía): {loss_d.item():.4f} | Loss G (Criminal): {loss_g.item():.4f}")

    # --- GUARDAR SOLO EL POLICÍA (Discriminator + Encoder) ---
    # Para predicción necesitamos:
    # 1. Encoder (para saber quién es quién en el grafo)
    # 2. Discriminator (para calcular la probabilidad de crimen)
    
    # Guardaremos un diccionario con ambos estados
    final_state = {
        'encoder': encoder.state_dict(),
        'discriminator': discriminator.state_dict()
    }
    torch.save(final_state, save_path)
    print(f"\nEntrenamiento Finalizado.")
    print(f"Agente Pre-Crimen (Policía) guardado en: {save_path}")

if __name__ == "__main__":
    entrenar_policia()
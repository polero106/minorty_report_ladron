import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

# ---------------------------------------------------------
# 1. EL MODELO BASE (GNN)
# ---------------------------------------------------------


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # SAGEConv es ideal para aprender de vecinos
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        # Primera capa de convolución + Activación ReLU
        x = self.conv1(x, edge_index).relu()
        # Segunda capa
        x = self.conv2(x, edge_index)
        return x

# ---------------------------------------------------------
# 2. LA "POLICÍA" (Discriminador)
# ---------------------------------------------------------


class PoliceDiscriminator(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        # Creamos una GNN base
        self.gnn = GNN(hidden_channels, hidden_channels)

        # Convertimos la GNN estándar en una GNN Heterogénea
        # para que entienda que hay "Personas" y "Ubicaciones"
        self.gnn = to_hetero(self.gnn, metadata, aggr='sum')

        # Clasificador final: ¿Es crimen real (1) o falso (0)?
        self.classifier = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        # 1. Obtener embeddings de los nodos
        x_dict = self.gnn(data.x_dict, data.edge_index_dict)

        # 2. Para clasificar enlaces, combinamos nodo origen y destino.
        #    Ejemplo: ¿Hay crimen entre Persona 'i' y Ubicacion 'j'?

        # (Aquí simplificamos asumiendo predicción sobre nodos para el ejemplo general,
        #  pero en el loop de entrenamiento usaremos el producto punto de los embeddings)
        return x_dict

# ---------------------------------------------------------
# 3. EL "CRIMINAL" (Generador)
# ---------------------------------------------------------


class CrimeGenerator(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.gnn = GNN(hidden_channels, hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata, aggr='sum')

    def forward(self, data):
        # El criminal mira el estado actual y genera embeddings
        # que se usarán para proponer nuevos enlaces
        return self.gnn(data.x_dict, data.edge_index_dict)

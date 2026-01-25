
import torch
import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/src')
from src.data_loader import MadridDataLoader

load_dotenv()

def debug_data():
    URI = os.getenv("NEO4J_URI", "neo4j+ssc://5d9c9334.databases.neo4j.io")
    AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "oTzaPYT99TgH-GM2APk0gcFlf9k16wrTcVOhtfmAyyA"))
    
    print("Loading data with MadridDataLoader (real coordinates)...")
    loader = MadridDataLoader(URI, AUTH)
    loader.load_nodes()
    loader.load_edges()
    data = loader.get_data()
    loader.close()
    
    print("\nMetaData:", data.metadata())
    
    print("\nNode Features:")
    for node_type in data.node_types:
        if 'x' in data[node_type]:
            print(f"  {node_type}: {data[node_type].x.shape}")
            # Muestra un ejemplo de coordenadas
            if data[node_type].x.shape[0] > 0:
                example = data[node_type].x[0]
                print(f"    Example: value={example[0]:.3f}, lat={example[1]:.3f}, lon={example[2]:.3f}")
        else:
            print(f"  {node_type}: NO FEATURES")
            
    print("\nEdge Indices:")
    for edge_type in data.edge_types:
        if 'edge_index' in data[edge_type]:
            print(f"  {edge_type}: {data[edge_type].edge_index.shape}, dtype={data[edge_type].edge_index.dtype}")
        else:
            print(f"  {edge_type}: MISSING")

if __name__ == "__main__":
    debug_data()

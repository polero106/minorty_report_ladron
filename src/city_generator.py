import time
import random
import numpy as np
import osmnx as ox
import networkx as nx
from faker import Faker
from neo4j import GraphDatabase

# Configuración (Cámbialo con tus datos de Aura o Local)
URI = "neo4j+ssc://5d9c9334.databases.neo4j.io"
AUTH = ("neo4j", "oTzaPYT99TgH-GM2APk0gcFlf9k16wrTcVOhtfmAyyA")

class CityGenerator:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.fake = Faker('es_ES') # Datos en español
        print("Descargando grafo de calles de Madrid (Distrito Centro)...")
        # Usamos 'network_type=drive' o 'walk' para obtener nodos de calles
        self.G = ox.graph_from_place('Centro, Madrid, Spain', network_type='all')
        self.nodes = list(self.G.nodes(data=True)) # Lista de (id, data)
        print(f"Conectado a la Matrix. Grafo cargado con {len(self.nodes)} nodos.")

    def close(self):
        self.driver.close()

    def get_random_point_on_street(self):
        """Selecciona un nodo aleatorio del grafo de calles de Madrid."""
        node_id, data = random.choice(self.nodes)
        return {
            'x': data['x'], # Longitude
            'y': data['y'], # Latitude
            'z': float(np.random.uniform(0, 50)) # Altura simulada para 3D
        }

    def generate_data(self, num_personas=1000, num_ubicaciones=50):
        print(f"Fabricando ciudad con {num_personas} ciudadanos y {num_ubicaciones} zonas...")
        
        # 1. GENERAR UBICACIONES (BOLAS AMARILLAS)
        ubicaciones = []
        for i in range(num_ubicaciones):
            coords = self.get_random_point_on_street()
            ubicaciones.append({
                'id': f"LOC_{i:03d}",
                'nombre': self.fake.street_name(),
                'tipo': 'Zona Comercial' if random.random() > 0.5 else 'Residencial',
                'peligrosidad': float(np.random.beta(3, 3)), # Distribución más equilibrada (Campana centrada en 0.5)
                'x': coords['x'],
                'y': coords['y'],
                'z': coords['z'],
                'color': '#FFD700' # Amarillo para la UI 3D
            })

        # 2. GENERAR PERSONAS (CIUDADANOS)
        personas = []
        for i in range(num_personas):
            # La "Risk Seed" es crucial: define la propensión latente al crimen.
            # Usamos una distribución normal más amplia y con media ligeramente superior
            risk_seed = np.random.normal(0.4, 0.25) 
            risk_seed = np.clip(risk_seed, 0.0, 1.0) # Forzar entre 0 y 1

            coords = self.get_random_point_on_street()
            
            personas.append({
                'id': f"P_{i:05d}",
                'nombre': self.fake.name(),
                'edad': random.randint(18, 80),
                'profesion': self.fake.job(),
                'risk_seed': float(risk_seed), # IMPORTANTE para el modelo
                'x': coords['x'], 
                'y': coords['y'], 
                'z': coords['z']
            })

        # 3. GENERAR WARNINGS/CRÍMENES PASADOS (BOLAS ROJAS)
        # Solo el 30% de la población tiene antecedentes (antes 10%)
        warnings = []
        criminales = [p for p in personas if p['risk_seed'] > 0.45] # Bajamos el umbral para tener más criminales
        
        for crim in criminales:
            if random.random() > 0.2: # 80% de probabilidad de tener antecedentes si eres criminal
                coords = self.get_random_point_on_street() # Crimen en lugar real
                warnings.append({
                    'id': f"WARN_{random.randint(0, 99999)}",
                    'delito': random.choice(['Robo', 'Agresión', 'Fraude', 'Homicidio']),
                    'gravedad': float(crim['risk_seed'] + np.random.uniform(0, 0.2)),
                    'fecha': self.fake.date_between(start_date='-2y', end_date='today').isoformat(),
                    'autor_id': crim['id'],
                    'x': coords['x'], # El crimen ocurre en un punto espacial
                    'y': coords['y'],
                    'z': coords['z'],
                    'color': '#FF0000' # Rojo para la UI 3D
                })

        return personas, ubicaciones, warnings

    def save_to_neo4j(self, personas, ubicaciones, warnings):
        print("Guardando en Neo4j usando UNWIND (Bulk Import)...")
        start_time = time.time()

        with self.driver.session() as session:
            # A. CREAR UBICACIONES (Nodos Amarillos)
            session.run("""
            UNWIND $batch AS row
            MERGE (u:Ubicacion {id: row.id})
            SET u.nombre = row.nombre, 
                u.peligrosidad = row.peligrosidad,
                u.x = row.x, u.y = row.y, u.z = row.z,
                u.lat = row.y, u.lon = row.x,
                u.color = row.color
            """, batch=ubicaciones)
            print(f"   -> {len(ubicaciones)} Ubicaciones creadas.")

            # B. CREAR PERSONAS (Nodos Neutros)
            session.run("""
            UNWIND $batch AS row
            MERGE (p:Persona {id: row.id})
            SET p.nombre = row.nombre, 
                p.edad = row.edad, 
                p.risk_seed = row.risk_seed,
                p.x = row.x, p.y = row.y, p.z = row.z,
                p.lat = row.y, p.lon = row.x
            """, batch=personas)
            print(f"   -> {len(personas)} Personas creadas.")

            # C. CREAR RELACIONES DE VIVIENDA (Aleatorio)
            # Conectamos personas a ubicaciones cercanas (simulado)
            session.run("""
            MATCH (p:Persona), (u:Ubicacion)
            WHERE rand() < 0.05  // Conexión aleatoria del 5%
            MERGE (p)-[:VIVE_EN]->(u)
            """)
            print("   -> Relaciones VIVE_EN generadas.")

            # D. CREAR WARNINGS (Nodos Rojos) y Conexiones
            session.run("""
            UNWIND $batch AS row
            MATCH (p:Persona {id: row.autor_id})
            MERGE (w:Warning {id: row.id})
            SET w.delito = row.delito,
                w.gravedad = row.gravedad,
                w.fecha = row.fecha,
                w.x = row.x, w.y = row.y, w.z = row.z,
                w.lat = row.y, w.lon = row.x,
                w.color = row.color
            MERGE (p)-[:COMETIO]->(w)
            """, batch=warnings)
            print(f"   -> {len(warnings)} Warnings (Crímenes) creados.")
            
            # E. RELACIONAR WARNINGS CON UBICACIONES
            # (El crimen ocurrió cerca de una ubicación)
            session.run("""
            MATCH (w:Warning), (u:Ubicacion)
            WHERE abs(w.x - u.x) < 0.005 AND abs(w.y - u.y) < 0.005 // Distancia Lat/Lon pequeña
            MERGE (w)-[:OCURRIO_EN]->(u)
            """)

        end_time = time.time()
        print(f"Todo listo en {end_time - start_time:.2f} segundos.")

if __name__ == "__main__":
    # 1. Instanciar Generador
    gen = CityGenerator(URI, AUTH)
    
    # 2. Generar Datos en Memoria (Listas de Diccionarios)
    personas, ubicaciones, warnings = gen.generate_data(num_personas=3000, num_ubicaciones=300)
    
    # 3. Volcar a la API de Neo4j
    gen.save_to_neo4j(personas, ubicaciones, warnings)
    
    gen.close()
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
        
        # 1. COORDENADAS REALES (OSMnx)
        # Usamos 'network_type=drive' (coches) o 'walk' (peatones). Usaremos 'all' para calles y plazas.
        # Esto descarga nodos reales con atributos 'y' (Latitud) y 'x' (Longitud)
        self.G = ox.graph_from_place('Centro, Madrid, Spain', network_type='all')
        self.nodes = list(self.G.nodes(data=True)) # Lista de tuplas: (node_id, atributos)
        
        print(f"Matrix Real Cargada: {len(self.nodes)} intersecciones de Madrid listas.")

    def clear_database(self):
        """Limpia todos los nodos y relaciones de Neo4j"""
        print("   -> Limpiando base de datos...")
        with self.driver.session() as session:
            try:
                # Eliminar todas las relaciones y nodos
                session.run("MATCH (n) DETACH DELETE n")
                print("   ✅ Base de datos limpia")
            except Exception as e:
                print(f"   ⚠️  Error al limpiar: {e}")

    def close(self):
        self.driver.close()

    def get_real_madrid_point(self):
        """
        Selecciona un nodo REAL del callejero de Madrid.
        Returns: {x: lon, y: lat, z: alt}
        """
        node_id, data = random.choice(self.nodes)
        
        # Extraer Lat/Lon REALES del nodo de OpenStreetMap
        # NO NORMALIZAR: Se devuelven tal cual (ej: 40.41, -3.70)
        return {
            'x': data['x'], # Longitud real
            'y': data['y'], # Latitud real
            'z': float(np.random.uniform(600, 650)) # Altura media de Madrid (~600-650m)
        }

    def generate_data(self, num_personas=1000, num_ubicaciones=50):
        print(f"Fabricando ciudad sintética con {num_personas} ciudadanos y {num_ubicaciones} zonas...")
        
        # ---------------------------------------------------------
        # 1. GENERAR UBICACIONES (Zonas de Interés)
        # ---------------------------------------------------------
        ubicaciones = []
        for i in range(num_ubicaciones):
            coords = self.get_real_madrid_point() # Punto real
            
            ubicaciones.append({
                'id': f"LOC_{i:03d}",
                'nombre': f"{self.fake.street_name()} {random.randint(1, 100)}", # Calle Real Inventada
                'tipo': 'Zona Comercial' if random.random() > 0.5 else 'Residencial',
                'peligrosidad': float(np.random.beta(2, 5)), # Beta skewed to 0. (Mayoría seguras)
                'x': coords['x'], # Lon
                'y': coords['y'], # Lat
                'z': coords['z'],
                'color': '#FFD700' # Amarillo
            })

        # ---------------------------------------------------------
        # 2. GENERAR PERSONAS (Ciudadanos)
        # ---------------------------------------------------------
        personas = []
        for i in range(num_personas):
            # La "Risk Seed" define la propensión al crimen
            risk_seed = np.random.normal(0.3, 0.2) 
            risk_seed = np.clip(risk_seed, 0.0, 1.0) 

            coords = self.get_real_madrid_point() # Viven en puntos reales
            
            personas.append({
                'id': f"P_{i:05d}",
                'nombre': self.fake.name(),
                'edad': random.randint(18, 80),
                'profesion': self.fake.job(),
                'risk_seed': float(risk_seed),
                'x': coords['x'], 
                'y': coords['y'], 
                'z': coords['z']
            })

        # ---------------------------------------------------------
        # 3. GENERAR WARNINGS (Crímenes) - CON COHERENCIA ESPACIAL
        # ---------------------------------------------------------
        # Lógica: Los crímenes ocurren EN o MUY CERCA de una Ubicación existente.
        warnings = []
        
        # Filtramos criminales potenciales (Risk Seed alta)
        criminales = [p for p in personas if p['risk_seed'] > 0.6] 
        print(f"   -> {len(criminales)} sujetos peligrosos potenciales.")
        
        for crim in criminales:
            # Probabilidad de cometer crimen basada en su risk_seed (muy baja)
            if random.random() < (crim['risk_seed'] * 0.2): 
                
                # SELECCIONAR ESCENA DEL CRIMEN
                # El crimen ocurre en una de las ubicaciones generadas (o muy cerca)
                scene = random.choice(ubicaciones)
                
                # Añadir Jitter (Variación aleatoria de ~10 metros)
                # 0.0001 grados ~ 11 metros
                jitter_lat = random.uniform(-0.0001, 0.0001)
                jitter_lon = random.uniform(-0.0001, 0.0001)
                
                warnings.append({
                    'id': f"WARN_{random.randint(10000, 99999)}",
                    'delito': random.choice(['Robo', 'Agresión', 'Vandalismo', 'Hurto']),
                    'gravedad': float(crim['risk_seed'] + np.random.uniform(0, 0.1)), # Gravedad ligada al riesgo
                    'fecha': self.fake.date_between(start_date='-1y', end_date='today').isoformat(),
                    'autor_id': crim['id'],
                    'scene_id': scene['id'], # Guardamos referencia para el grafo
                    
                    # 2. COORDENADAS CRÍMENES (Warnings)
                    # Heredan la pos de la ubicación + jitter
                    'x': scene['x'] + jitter_lon, 
                    'y': scene['y'] + jitter_lat,
                    'z': scene['z'],
                    'color': '#FF0000', # Rojo
                    'type': 'Crimen'
                })

        return personas, ubicaciones, warnings

    def save_to_neo4j(self, personas, ubicaciones, warnings):
        print("Guardando en Neo4j usando UNWIND (Bulk Import)...")
        start_time = time.time()

        with self.driver.session() as session:
            # A. CREAR UBICACIONES
            session.run("""
            UNWIND $batch AS row
            MERGE (u:Ubicacion {id: row.id})
            SET u.nombre = row.nombre, 
                u.peligrosidad = row.peligrosidad,
                u.x = row.x, u.y = row.y, u.z = row.z,
                u.lat = row.y, u.lon = row.x,  // Propiedades Lat/Lon explícitas
                u.color = row.color
            """, batch=ubicaciones)
            print(f"   -> {len(ubicaciones)} Ubicaciones creadas.")

            # B. CREAR PERSONAS
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

            # C. RELACIONES VIVIENDA (probabilidad baja para no superar límites)
            session.run("""
            MATCH (p:Persona), (u:Ubicacion)
            WHERE rand() < 0.005
            MERGE (p)-[:VIVE_EN]->(u)
            """)

            # D. CREAR WARNINGS (Nodos Rojos)
            # Ahora tienen coordenadas reales asociadas a la escena
            session.run("""
            UNWIND $batch AS row
            MATCH (p:Persona {id: row.autor_id}) // El autor debe existir
            MERGE (w:Warning {id: row.id})
            SET w.delito = row.delito,
                w.gravedad = row.gravedad,
                w.fecha = row.fecha,
                w.x = row.x, w.y = row.y, 
                w.lat = row.y, w.lon = row.x, // Coordenadas geograficas
                w.color = row.color,
                w.type = row.type
            MERGE (p)-[:COMETIO]->(w)
            """, batch=warnings)
            
            # E. RELACIONAR CRÍMENES CON SU ESCENA (Exacta)
            # Usamos el scene_id que guardamos anteriormente para evitar busquedas espaciales costosas
            session.run("""
            UNWIND $batch AS row
            MATCH (w:Warning {id: row.id})
            MATCH (u:Ubicacion {id: row.scene_id})
            MERGE (w)-[:OCURRIO_EN]->(u)
            """, batch=warnings)
            
            print(f"   -> {len(warnings)} Warnings (Crímenes) creados y vinculados.")

        end_time = time.time()
        print(f"Ciudad Generada en {end_time - start_time:.2f} segundos.")

if __name__ == "__main__":
    gen = CityGenerator(URI, AUTH)
    
    # Generamos dataset
    personas, ubicaciones, warnings = gen.generate_data(num_personas=500, num_ubicaciones=15)
    
    # Guardamos
    gen.save_to_neo4j(personas, ubicaciones, warnings)
    
    gen.close()
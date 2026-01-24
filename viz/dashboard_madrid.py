import panel as pn
import hvplot.pandas
import pandas as pd
from neo4j import GraphDatabase
import numpy as np
from dotenv import load_dotenv

# Cargar variables de entorno (.env)
load_dotenv()

# Inicializar extensión de Panel

pn.extension('mapbox')

# --- CONFIGURACIÓN ---
# Coordenadas base: Madrid, Puerta del Sol
CENTER_LAT = 40.4167
CENTER_LON = -3.7032

# Factor de escala para convertir unidades abstractas (x,y) a grados lat/lon
# Asumimos que el mapa 3D tiene un rango de +/- 100 unidades.
# Queremos que eso cubra aprox 5-10km.
# 1 grado ~ 111km. 0.05 grados ~ 5.5km.
SCALE_FACTOR = 0.0005 

# Credenciales Neo4j (Idealmente usar vbles de entorno, aquí hardcoded por simplicidad del prototipo)
URI = "neo4j+ssc://5d9c9334.databases.neo4j.io"
AUTH = ("neo4j", "oTzaPYT99TgH-GM2APk0gcFlf9k16wrTcVOhtfmAyyA")

class MadridCrimeDashboard:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        
    def close(self):
        self.driver.close()
        
    def load_data(self):
        query = """
        MATCH (n)
        WHERE n:Persona OR n:Ubicacion OR n:Warning
        RETURN labels(n)[0] as tipo, 
               id(n) as id, 
               n.x as x, n.y as y, 
               n.risk_seed as risk, 
               n.peligrosidad as danger,
               n.gravedad as gravity,
               n.delito as delito
        """
        data = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                # Transformar coords abstractas a Geo (Madrid)
                abs_x = record['x'] if record['x'] is not None else 0
                abs_y = record['y'] if record['y'] is not None else 0
                
                lat = CENTER_LAT + (abs_y * SCALE_FACTOR)
                # Ajuste de longitud por latitud (aprox cos(40) ~ 0.76)
                lon = CENTER_LON + (abs_x * SCALE_FACTOR / 0.76)
                
                row = {
                    'tipo': record['tipo'],
                    'id': record['id'],
                    'lat': lat,
                    'lon': lon,
                    'detalle': ''
                }
                
                if record['tipo'] == 'Persona':
                    risk = record['risk'] if record['risk'] is not None else 0
                    row['score'] = risk
                    row['detalle'] = f"Risk: {risk:.2f}"
                    row['size'] = 5
                    
                elif record['tipo'] == 'Ubicacion':
                    danger = record['danger'] if record['danger'] is not None else 0
                    row['score'] = danger
                    row['detalle'] = f"Peligro: {danger:.2f}"
                    row['size'] = 10
                    
                elif record['tipo'] == 'Warning':
                    grav = record['gravity'] if record['gravity'] is not None else 0
                    delito = record['delito'] if record['delito'] is not None else 'Crimen'
                    row['score'] = grav
                    row['detalle'] = f"{delito} (Gravedad: {grav:.2f})"
                    row['size'] = 8
                
                data.append(row)
                
        return pd.DataFrame(data)

def create_dashboard():
    dashboard = MadridCrimeDashboard(URI, AUTH)
    try:
        df = dashboard.load_data()
    finally:
        dashboard.close()
        
    if df.empty:
        return pn.pane.Markdown("## No se encontraron datos en Neo4j.")

    # Separar capas
    df_personas = df[df['tipo'] == 'Persona']
    # Filtrar solo personas de alto riesgo para no saturar el mapa
    df_suspects = df_personas[df_personas['score'] > 0.7].copy()
    
    df_locs = df[df['tipo'] == 'Ubicacion'].copy()
    df_warnings = df[df['tipo'] == 'Warning'].copy()
    
    # Visualización
    # 1. Ubicaciones (Amarillo)
    map_locs = df_locs.hvplot.points(
        'lon', 'lat', 
        geo=True, 
        color='gold', 
        size='size',
        hover_cols=['detalle', 'tipo'],
        label='Ubicaciones',
        tiles='CartoLight',
        alpha=0.8
    )
    
    # 2. Warnings/Crímenes (Rojo)
    map_warn = df_warnings.hvplot.points(
        'lon', 'lat', 
        geo=True, 
        color='red', 
        size='size',
        hover_cols=['detalle', 'tipo'],
        label='Crímenes (Warnings)',
        alpha=0.9
    )
    
    # 3. Personas de Alto Riesgo (Azul Oscuro/Negro)
    map_suspects = df_suspects.hvplot.points(
        'lon', 'lat', 
        geo=True, 
        color='navy', 
        size='size',
        hover_cols=['detalle', 'tipo'],
        label='Sospechosos (High Risk)',
        alpha=0.6
    )
    
    # Combinar
    final_map = (map_locs * map_warn * map_suspects).opts(
        width=1000, 
        height=700,
        title="Minority Report Madrid: Pre-Crimen Dashboard",
        active_tools=['wheel_zoom', 'pan']
    )
    
    # Layout Final
    layout = pn.Column(
        pn.pane.Markdown("# 🕵️ Minority Report: Madrid Surveillance"),
        pn.pane.Markdown("Visualización geoespacial de predicción de crímenes."),
        final_map
    )
    
    return layout

# Para servir con 'panel serve viz/dashboard_madrid.py'
if __name__.startswith('bokeh'):
    layout = create_dashboard()
    layout.servable()
elif __name__ == '__main__':
    # Modo prueba local simple
    create_dashboard().show()

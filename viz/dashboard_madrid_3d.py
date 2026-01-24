import panel as pn
import pydeck as pdk
import pandas as pd
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Obtener API Key
MAPBOX_KEY = os.getenv('MAPBOX_API_KEY')
if not MAPBOX_KEY:
    print("WARNING: MAPBOX_API_KEY not found in .env")

# Inicializar Panel con token
pn.extension('deckgl')

# --- CONFIGURACIÓN ---
URI = "neo4j+ssc://5d9c9334.databases.neo4j.io"
AUTH = ("neo4j", "oTzaPYT99TgH-GM2APk0gcFlf9k16wrTcVOhtfmAyyA")

# Centro de Madrid
MADRID_LAT = 40.4167
MADRID_LON = -3.7032

class MadridDashboard3D:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        
    def close(self):
        self.driver.close()
        
    def get_data(self):
        data = {"Persona": [], "Ubicacion": [], "Warning": [], "Relaciones": []}
        
        with self.driver.session() as session:
            # 1. NODOS
            query_nodes = """
            MATCH (n) 
            WHERE (n:Persona OR n:Ubicacion OR n:Warning) 
            RETURN labels(n)[0] as tipo, id(n) as id, n.lat as lat, n.lon as lon, properties(n) as props
            """
            result_nodes = session.run(query_nodes)
            for record in result_nodes:
                lat = record['lat']
                lon = record['lon']
                props = record['props']
                
                # Filtrado Estricto de Coordenadas (Madrid aprox: Lat 40, Lon -3)
                if lat is None or lon is None or not (39 < lat < 41) or not (-5 < lon < -2):
                    continue

                # --- Generación de Tooltip HTML ---
                html_info = ""
                tipo = record['tipo']
                
                if tipo == 'Persona':
                    nombre = props.get('nombre', 'Desconocido')
                    risk = props.get('risk_seed', 0.0)
                    try:
                        risk_val = float(risk)
                    except:
                        risk_val = 0.0
                    html_info = f"<b>CIUDADANO</b><br>Nombre: {nombre}<br>Risk Seed: {risk_val:.2f}"
                    
                elif tipo == 'Ubicacion':
                    html_info = "<b>UBICACIÓN</b>"
                    for k, v in props.items():
                        if k not in ['lat', 'lon', 'id']:
                            html_info += f"<br>{k}: {v}"
                            
                elif tipo == 'Warning':
                    html_info = "<b>ALERTA POLICIAL</b>"
                    # Ordenar un poco si es posible, o simplemente listar
                    # Priorizar Delito, Gravedad, Fecha
                    delito = props.get('delito', 'N/A')
                    gravedad = props.get('gravedad', 'N/A')
                    fecha = props.get('fecha', 'N/A')
                    
                    html_info += f"<br>Delito: {delito}"
                    html_info += f"<br>Gravedad: {gravedad}"
                    html_info += f"<br>Fecha: {fecha}"
                    
                    # Añadir el resto
                    for k, v in props.items():
                        if k not in ['lat', 'lon', 'id', 'author_id', 'delito', 'gravedad', 'fecha']:
                            html_info += f"<br>{k}: {v}"

                row = {
                    "id": record['id'],
                    "lat": lat,
                    "lon": lon,
                    "type": tipo,
                    "html_info": html_info
                }
                
                # Configuración de Colores y Radios para cada tipo
                if tipo == 'Persona':
                    row['color'] = [0, 100, 255, 160] # Azul Transparente
                    row['radius'] = 8 
                elif tipo == 'Ubicacion':
                    row['color'] = [255, 200, 0, 200] # Amarillo Oro
                    row['radius'] = 25
                elif tipo == 'Warning':
                    row['color'] = [255, 0, 0, 255] # Rojo Sólido
                    row['radius'] = 35
                
                data[tipo].append(row)
                
            # 2. RELACIONES (Arcos)
            # Solo traemos relaciones donde ambos nodos tengan coordenadas válidas
            # Y la distancia no sea absurda (evitar líneas cruzando el mapa entero)
            query_rels = """
            MATCH (s)-[r]->(t)
            WHERE ((s:Persona AND t:Warning) OR (s:Persona AND t:Ubicacion))
            RETURN s.lat as src_lat, s.lon as src_lon, t.lat as dst_lat, t.lon as dst_lon, type(r) as type
            """
            result_rels = session.run(query_rels)
            for record in result_rels:
                s_lat, s_lon = record['src_lat'], record['src_lon']
                t_lat, t_lon = record['dst_lat'], record['dst_lon']
                
                # Validación estricta de coordenadas (evita (0,0) y nulls)
                if (s_lat and s_lon and t_lat and t_lon and
                    39 < s_lat < 41 and -5 < s_lon < -2 and
                    39 < t_lat < 41 and -5 < t_lon < -2):
                    
                    # Distancia de Manhattan simple para filtrar arcos muy largos (ruido visual)
                    dist = abs(s_lat - t_lat) + abs(s_lon - t_lon)
                    if dist < 0.05: # Aprox 5km max
                        data["Relaciones"].append({
                            "source": [s_lon, s_lat],
                            "target": [t_lon, t_lat],
                            "color": [200, 200, 200, 100], # Gris sutil transparente
                            "type": record['type']
                        })
                    
        return data

    def view(self):
        raw_data = self.get_data()
        
        # Convertir a DataFrames
        df_personas = pd.DataFrame(raw_data["Persona"])
        df_ubicaciones = pd.DataFrame(raw_data["Ubicacion"])
        df_warnings = pd.DataFrame(raw_data["Warning"])
        df_relaciones = pd.DataFrame(raw_data["Relaciones"])
        
        layers = []


        
        # --- Layers de Puntos ---
        if not df_ubicaciones.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=df_ubicaciones,
                get_position='[lon, lat]',
                get_fill_color='color',
                get_radius='radius',
                pickable=True,
                opacity=0.8,
                radius_min_pixels=3,
                radius_max_pixels=30
            ))
            
        if not df_warnings.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=df_warnings,
                get_position='[lon, lat]',
                get_fill_color='color',
                get_radius='radius',
                pickable=True,
                radius_min_pixels=5,
                radius_max_pixels=40,
                stroked=True,
                get_line_color=[255, 255, 255],
                line_width_min_pixels=1
            ))
            
        if not df_personas.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=df_personas,
                get_position='[lon, lat]',
                get_fill_color='color',
                get_radius='radius',
                pickable=True,
                radius_min_pixels=2,
                radius_max_pixels=10
            ))
            


        # --- Base Map & View ---
        view_state = pdk.ViewState(
            latitude=MADRID_LAT,
            longitude=MADRID_LON,
            zoom=12,
            pitch=45, # Mantenemos algo de inclinación para profundidad
            bearing=0
        )
        
        # IMPORTANTE: mapbox_key obligatorio para estilos dark/satellite
        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=view_state,
            layers=layers,
            api_keys={"mapbox": MAPBOX_KEY}, 
            tooltip={"html": "{html_info}", "style": {"color": "white"}}
        )
        

        # --- LEYENDA FLOTANTE (HTML/CSS) ---
        legend_html = """
        <div style="
            position: fixed; 
            bottom: 30px; 
            right: 30px; 
            background-color: rgba(20, 20, 20, 0.85); 
            padding: 15px; 
            border-radius: 8px; 
            border: 1px solid #444; 
            z-index: 9999; 
            color: #eee; 
            font-family: sans-serif;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            min-width: 150px;
        ">
            <h4 style="margin: 0 0 10px 0; border-bottom: 1px solid #666; padding-bottom: 5px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #fff;">Leyenda</h4>
            
             <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="width: 12px; height: 12px; background-color: rgba(0, 100, 255, 0.8); border-radius: 50%; display: inline-block; margin-right: 10px; border: 1px solid #fff;"></span>
                <span style="font-size: 12px;">Personas</span>
            </div>
            
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="width: 12px; height: 12px; background-color: rgba(255, 200, 0, 0.8); border-radius: 50%; display: inline-block; margin-right: 10px; border: 1px solid #fff;"></span>
                <span style="font-size: 12px;">Ubicaciones</span>
            </div>
            
            <div style="display: flex; align-items: center;">
                <span style="width: 12px; height: 12px; background-color: rgba(255, 0, 0, 1); border-radius: 50%; display: inline-block; margin-right: 10px; border: 1px solid #fff;"></span>
                <span style="font-size: 12px;">Alertas</span>
            </div>
        </div>
        """

        return pn.Column(
            pn.pane.Markdown("# 🏙️ Minority Report: Madrid Surveillance"),
            pn.pane.Markdown(f"""
            **Estado del Sistema**:
            - 👥 Personas: {len(df_personas)}
            - 🏢 Zonas: {len(df_ubicaciones)}
            - 🚨 Alertas: {len(df_warnings)}
            """),
            pn.Column(
                pn.pane.DeckGL(deck, height=700, sizing_mode='stretch_width'),
                pn.pane.HTML(legend_html, height=0, width=0, sizing_mode="fixed", margin=0) # Trick to inject fixed HTML
            ) 
        )

# Entry point para 'panel serve'
if __name__.startswith('bokeh'):
    dashboard = MadridDashboard3D(URI, AUTH)
    layout = dashboard.view()
    layout.servable()
elif __name__ == '__main__':
    # Para pruebas locales rápidas
    dashboard = MadridDashboard3D(URI, AUTH)
    layout = dashboard.view()
    layout.show()

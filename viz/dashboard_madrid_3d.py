
import panel as pn
import pydeck as pdk
import pandas as pd
import numpy as np
import sys
import os
import hvplot.pandas
import networkx as nx
import plotly.graph_objects as go
from bokeh.plotting import figure, from_networkx
from bokeh.models import HoverTool, ColumnDataSource, Label
from bokeh.palettes import Category20
import torch
from datetime import datetime, timedelta
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction_service import PredictionService
# ==============================================================================
# 0. CONSTANTES (Deben coincidir con ETL)
# ==============================================================================
LAT_MIN, LAT_MAX = 40.30, 40.55
LON_MIN, LON_MAX = -3.85, -3.50

def denormalize_lat(y): 
    return y * (LAT_MAX - LAT_MIN) + LAT_MIN

def denormalize_lon(x): 
    return x * (LON_MAX - LON_MIN) + LON_MIN

def peligrosidad_to_color(peligrosidad):
    """Mapea peligrosidad [0,1] a colores: Azul (seguro) -> Rojo (peligroso)"""
    # Escala de colores fríos a cálidos: Azul -> Verde -> Amarillo -> Rojo
    if peligrosidad < 0.25:
        # Azul puro (muy seguro)
        r, g, b = 0, 100, 255
    elif peligrosidad < 0.50:
        # Azul a Verde
        norm = (peligrosidad - 0.25) / 0.25
        r = int(0)
        g = int(100 + (155 * norm))
        b = int(255 - (200 * norm))
    elif peligrosidad < 0.75:
        # Verde a Amarillo
        norm = (peligrosidad - 0.50) / 0.25
        r = int(0 + (255 * norm))
        g = int(255)
        b = int(55 - (55 * norm))
    else:
        # Amarillo a Rojo (muy peligroso)
        norm = (peligrosidad - 0.75) / 0.25
        r = int(255)
        g = int(255 - (100 * norm))
        b = int(0)
    
    return [r, g, b, 200]  # Alpha = 200 para translucidez

def create_network_graph(nodos_df, edges_df):
    """Crea un grafo de red de sospechosos con NetworkX y Bokeh"""
    if nodos_df.empty:
        # Si no hay datos, retornar figura vacía
        p = figure(
            width=600, height=500,
            title="🕸️ Red de Sospechosos",
            toolbar_location=None,
            background_fill_color="#000000"
        )
        p.text([0.5], [0.5], text=["Sin datos de red"], 
               text_color="#888888", text_font_size="14pt", text_align="center")
        return p
    
    # Crear grafo NetworkX
    G = nx.Graph()
    
    # Añadir nodos con atributos
    for idx, row in nodos_df.iterrows():
        node_id = row.get('id', f'N_{idx}')
        risk_val = row.get('risk', 0.5)
        G.add_node(node_id, 
                   risk=risk_val,
                   label=str(node_id),
                   risk_percent=int(risk_val * 100))
    
    # Añadir aristas si existen
    num_connections = 0
    if not edges_df.empty and 'source' in edges_df.columns and 'target' in edges_df.columns:
        for idx, row in edges_df.iterrows():
            G.add_edge(row['source'], row['target'])
            num_connections += 1
    
    # Layout del grafo (spring layout para distribución estética)
    pos = nx.spring_layout(G, k=2.0, iterations=50)
    
    # Crear figura Bokeh con título dinámico
    title_text = f"🕸️ Red Criminal: {len(G.nodes())} Sospechosos | {num_connections} Conexiones"
    p = figure(
        width=600, height=500,
        title=title_text,
        toolbar_location=None,
        x_range=(-1.3, 1.3),
        y_range=(-1.3, 1.3),
        background_fill_color="#000000",
        border_fill_color="#000000"
    )
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.grid.visible = False
    
    # Dibujar aristas (líneas finas grises con mejor visualización)
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        p.line([x0, x1], [y0, y1], 
               line_width=1.5, 
               line_alpha=0.4, 
               line_color="#666666")
    
    # Preparar datos de nodos
    node_xs = [pos[node][0] for node in G.nodes()]
    node_ys = [pos[node][1] for node in G.nodes()]
    node_colors = []
    node_sizes = []
    node_labels = []
    node_risks = []
    node_risk_categories = []
    
    for node in G.nodes():
        risk = G.nodes[node].get('risk', 0.5)
        node_labels.append(G.nodes[node].get('label', str(node)))
        node_risks.append(f"{int(risk * 100)}%")
        
        # Color según riesgo: Cian (bajo) -> Naranja (medio) -> Rojo (alto)
        if risk < 0.5:
            node_colors.append('#00FFFF')
            node_sizes.append(15)
            node_risk_categories.append('Bajo')
        elif risk < 0.7:
            node_colors.append('#FFA500')
            node_sizes.append(20)
            node_risk_categories.append('Medio')
        else:
            node_colors.append('#FF0033')
            node_sizes.append(25)
            node_risk_categories.append('ALTO')
    
    # Crear ColumnDataSource con más información
    source = ColumnDataSource(data=dict(
        x=node_xs,
        y=node_ys,
        colors=node_colors,
        sizes=node_sizes,
        labels=node_labels,
        risks=node_risks,
        categories=node_risk_categories
    ))
    
    # Dibujar nodos (círculos brillantes)
    nodes_renderer = p.circle('x', 'y', 
                              size='sizes',
                              fill_color='colors', 
                              fill_alpha=0.85,
                              line_color='#FFFFFF', 
                              line_width=2,
                              source=source)
    
    # Añadir etiquetas de texto visibles en los nodos
    from bokeh.models import LabelSet
    labels = LabelSet(x='x', y='y', text='labels',
                     x_offset=-8, y_offset=-8,
                     text_font_size="8pt", text_color="#FFFFFF",
                     text_alpha=0.8,
                     source=source, text_align='center')
    p.add_layout(labels)
    
    # Añadir hover tool mejorado con más información
    hover = HoverTool(
        renderers=[nodes_renderer],
        tooltips=[
            ("🆔 ID", "@labels"),
            ("⚠️ Riesgo", "@risks"),
            ("📊 Categoría", "@categories")
        ]
    )
    p.add_tools(hover)
    
    return p

def create_3d_network_map(nodos_df):
    """Crea mapa 3D con Pydeck mostrando ubicaciones y conexiones de sospechosos"""
    if nodos_df.empty:
        # Retornar mapa vacío
        return pdk.Deck(
            initial_view_state=INITIAL_VIEW_STATE,
            layers=[],
            map_style=MAP_STYLE
        )
    
    # Obtener coordenadas de nodos desde service.data
    x_pers = service.data['Persona'].x.cpu().numpy()
    
    # Preparar datos de sospechosos con coordenadas
    suspects_coords = []
    for _, row in nodos_df.iterrows():
        id_str = str(row['id'])
        persona_id = int(id_str.split('_')[-1]) if '_' in id_str else int(id_str)
        
        if persona_id < len(x_pers):
            suspects_coords.append({
                'lat': denormalize_lat(x_pers[persona_id, 1]),
                'lon': denormalize_lon(x_pers[persona_id, 2]),
                'risk': row['risk'],
                'id': id_str,
                'elevation': int(row['risk'] * 500)  # Elevación basada en riesgo
            })
    
    df_suspects = pd.DataFrame(suspects_coords)
    
    if df_suspects.empty:
        return pdk.Deck(
            initial_view_state=INITIAL_VIEW_STATE,
            layers=[],
            map_style=MAP_STYLE
        )
    
    # Crear datos de conexiones (arcos entre todos los sospechosos)
    connections = []
    for i in range(len(df_suspects)):
        for j in range(i + 1, len(df_suspects)):
            connections.append({
                'source_lat': df_suspects.iloc[i]['lat'],
                'source_lon': df_suspects.iloc[i]['lon'],
                'target_lat': df_suspects.iloc[j]['lat'],
                'target_lon': df_suspects.iloc[j]['lon']
            })
    
    df_connections = pd.DataFrame(connections)
    
    # CAPA 1: Ubicaciones de Sospechosos (ScatterplotLayer con elevación)
    layer_suspects = pdk.Layer(
        "ColumnLayer",
        data=df_suspects,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=2,
        get_fill_color=[255, 50, 50, 255],  # Rojo neón
        radius=50,
        pickable=True,
        auto_highlight=True
    )
    
    # CAPA 2: Conexiones (ArcLayer 3D)
    layer_arcs = pdk.Layer(
        "ArcLayer",
        data=df_connections,
        get_source_position=["source_lon", "source_lat"],
        get_target_position=["target_lon", "target_lat"],
        get_source_color=[0, 255, 255, 180],  # Cian translúcido
        get_target_color=[255, 140, 0, 180],  # Naranja translúcido
        get_width=3,
        get_height=0.3,  # Altura del arco
        pickable=True
    )
    
    # Crear vista inicial centrada en Madrid con más altura
    view_state_3d = pdk.ViewState(
        latitude=40.416775,
        longitude=-3.703790,
        zoom=12,
        pitch=50,  # Ángulo para ver los arcos 3D
        bearing=0
    )
    
    return pdk.Deck(
        initial_view_state=view_state_3d,
        layers=[layer_arcs, layer_suspects],  # Arcos primero, luego puntos encima
        map_style=MAP_STYLE,
        tooltip={
            "html": "<b>Sospechoso de Alto Riesgo</b><br>ID: {id}<br>Riesgo: {risk:.0%}",
            "style": {"backgroundColor": "#1a1a1a", "color": "#FF0033"}
        }
    )

def create_temporal_radar_chart(df_pred):
    """Genera Spider Chart de riesgo por hora del día (0-23h)"""
    from bokeh.models import Range1d
    
    # DATOS SINTÉTICOS (temporal): Simulación de distribución horaria
    # TODO: En producción, extraer de df_pred con columna 'timestamp' o 'hora'
    # Ejemplo: df_pred['hora'] = pd.to_datetime(df_pred['timestamp']).dt.hour
    #          risk_by_hour = df_pred.groupby('hora')['probabilidad'].mean().reindex(range(24), fill_value=0).tolist()
    
    hours = list(range(24))
    
    # Verificar si hay datos temporales reales
    if 'timestamp' in df_pred.columns or 'hora' in df_pred.columns:
        # Extraer hora si existe timestamp
        if 'timestamp' in df_pred.columns:
            df_pred['hora_temp'] = pd.to_datetime(df_pred['timestamp'], errors='coerce').dt.hour
        else:
            df_pred['hora_temp'] = df_pred['hora']
        
        # Calcular riesgo promedio por hora
        hourly_risk = df_pred.groupby('hora_temp')['probabilidad'].mean()
        risk_by_hour = [hourly_risk.get(h, 0.2) for h in hours]
        
        # Normalizar a rango 0-1
        max_risk = max(risk_by_hour) if max(risk_by_hour) > 0 else 1
        risk_by_hour = [r / max_risk for r in risk_by_hour]
    else:
        # DATOS SINTÉTICOS: Patrón típico de criminalidad urbana
        # Basado en estudios criminológicos: picos nocturnos (22-04h) y tarde (18-20h)
        risk_by_hour = [
            0.3, 0.2, 0.15, 0.25, 0.4, 0.3, 0.2, 0.15, 0.2, 0.3,  # 00-09h (madrugada activa, mañana baja)
            0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.35, 0.4, 0.5, 0.6,    # 10-19h (mediodía-tarde)
            0.7, 0.8, 0.9, 0.7  # 20-23h (pico nocturno)
        ]
        
        # Escalar según cantidad de predicciones
        if len(df_pred) > 10:
            scale_factor = min(len(df_pred) / 100.0, 1.0)
            risk_by_hour = [r * scale_factor for r in risk_by_hour]
    
    # Configurar gráfico radial
    p = figure(
        width=400, height=400,
        title="🕒 Patrón Temporal de Crimen (24h)",
        toolbar_location=None,
        x_range=Range1d(-1.2, 1.2),
        y_range=Range1d(-1.2, 1.2),
        background_fill_color="#000000",
        border_fill_color="#000000"
    )
    
    # Ocultar ejes
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.grid.visible = False
    
    # Dibujar rejilla circular (3 anillos)
    for radius in [0.33, 0.66, 1.0]:
        angles = np.linspace(0, 2*np.pi, 100)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        p.line(xs, ys, color="#333333", alpha=0.3, line_width=1)
    
    # Línea de referencia al 80% (Umbral de Alerta) - línea punteada roja
    alert_radius = 0.80
    angles_alert = np.linspace(0, 2*np.pi, 100)
    xs_alert = alert_radius * np.cos(angles_alert)
    ys_alert = alert_radius * np.sin(angles_alert)
    p.line(xs_alert, ys_alert, 
           color="#FF0033", 
           alpha=0.6, 
           line_width=2, 
           line_dash='dashed')
    
    # Dibujar líneas radiales (cada 3 horas)
    for hour in [0, 3, 6, 9, 12, 15, 18, 21]:
        angle = (hour / 24) * 2 * np.pi - np.pi/2  # Empezar en 12 (top)
        p.line([0, np.cos(angle)], [0, np.sin(angle)], color="#333333", alpha=0.3, line_width=1)
    
    # Crear polígono de datos
    angles_data = [(h / 24) * 2 * np.pi - np.pi/2 for h in hours]
    xs = [risk_by_hour[i] * np.cos(angles_data[i]) for i in range(24)]
    ys = [risk_by_hour[i] * np.sin(angles_data[i]) for i in range(24)]
    
    # Cerrar el polígono
    xs.append(xs[0])
    ys.append(ys[0])
    
    # Dibujar área rellena (cian translúcido)
    p.patch(xs, ys, color="#00FFFF", alpha=0.4, line_width=2, line_color="#00FFFF")
    
    # Añadir etiquetas de hora (0, 6, 12, 18)
    label_hours = [0, 6, 12, 18]
    label_texts = ["00h", "06h", "12h", "18h"]
    for hour, text in zip(label_hours, label_texts):
        angle = (hour / 24) * 2 * np.pi - np.pi/2
        label_x = 1.15 * np.cos(angle)
        label_y = 1.15 * np.sin(angle)
        label = Label(x=label_x, y=label_y, text=text, 
                     text_font_size="10pt", text_color="#00FFFF",
                     text_align="center", text_baseline="middle")
        p.add_layout(label)
    
    return p

# ==============================================================================
# 1. CONFIGURACIÓN E INICIALIZACIÓN
# ==============================================================================
pn.extension('deckgl', notifications=True, sizing_mode="stretch_width")

# Inicializar Servicio IA
try:
    service = PredictionService()
    print("✅ Servicio IA cargado correctamente.")
except Exception as e:
    print(f"❌ Error cargando servicio IA: {e}")
    service = None

# ==============================================================================
# 2. PREPARACIÓN DEL MAPA BASE
# ==============================================================================
MAP_STYLE = "mapbox://styles/mapbox/dark-v10"
INITIAL_VIEW_STATE = pdk.ViewState(
    latitude=40.416775, 
    longitude=-3.703790, 
    zoom=13.5,  # Zoom aumentado para ver mejor los nodos (antes 12.5)
    pitch=45, 
    bearing=0
)

# Definir tooltips
TOOLTIP_CONFIG = {
    "html": "<b>{tipo}</b><br>ID: {id_visual}",
    "style": {"backgroundColor": "#1a1a1a", "color": "#ffffff", "zIndex": "999"}
}

# Preparar capas iniciales
if service:
    # Extraemos datos a CPU y DENORMALIZAMOS
    x_pers = service.data['Persona'].x.cpu().numpy()
    x_locs = service.data['Ubicacion'].x.cpu().numpy()
    
    # Índices: 0=Risk/Danger, 1=Lat, 2=Lon
    df_personas_base = pd.DataFrame({
        'lat': denormalize_lat(x_pers[:, 1]),
        'lon': denormalize_lon(x_pers[:, 2]),
        'tipo': 'Persona',
        'id_visual': [f'P_{i}' for i in range(len(x_pers))]
    })
    
    df_ubicaciones_base = pd.DataFrame({
        'lat': denormalize_lat(x_locs[:, 1]),
        'lon': denormalize_lon(x_locs[:, 2]),
        'tipo': 'Ubicación',
        'id_visual': [f'U_{i}' for i in range(len(x_locs))]
    })
    
    # Obtener nodos Warning (Crímenes) - necesitamos mapearlos a ubicaciones
    edge_index_ocurrio = service.data['Warning', 'OCURRIO_EN', 'Ubicacion'].edge_index.cpu().numpy()
    
    # Mapear cada Warning a su Ubicación para obtener coordenadas
    warnings_locations = []
    for i in range(edge_index_ocurrio.shape[1]):
        w_idx = edge_index_ocurrio[0, i]
        u_idx = edge_index_ocurrio[1, i]
        warnings_locations.append({
            'lat': denormalize_lat(x_locs[u_idx, 1]),
            'lon': denormalize_lon(x_locs[u_idx, 2]),
            'tipo': 'Crimen',
            'id_visual': f'W_{w_idx}'
        })
    
    df_warnings_base = pd.DataFrame(warnings_locations) if warnings_locations else pd.DataFrame()
    
    layer_personas = pdk.Layer(
        "ScatterplotLayer",
        data=df_personas_base,
        get_position=["lon", "lat"],
        get_color=[0, 100, 255, 120], # Azul con opacidad media (alpha ~0.47)
        get_radius=8,  # Tamaño intermedio entre 5 y 10
        pickable=True,
        radius_min_pixels=1
    )
    
    layer_ubicaciones = pdk.Layer(
        "ScatterplotLayer",
        data=df_ubicaciones_base,
        get_position=["lon", "lat"],
        get_color=[255, 200, 0, 255], # Ámbar opaco (alpha=1.0)
        get_radius=50,  # Ligeramente más grande para destacar
        pickable=True,
        radius_min_pixels=3
    )
    
    layer_warnings = pdk.Layer(
        "ScatterplotLayer",
        data=df_warnings_base,
        get_position=["lon", "lat"],
        get_color=[255, 50, 50, 255], # Rojo intenso opaco (alpha=1.0)
        get_radius=25,  # Más grande para destacar como amenaza
        pickable=True,
        radius_min_pixels=2,
        stroked=True,
        get_line_color=[255, 255, 255, 255],  # Borde blanco para contraste
        line_width_min_pixels=2
    ) if not df_warnings_base.empty else None
    
    initial_layers = [layer_personas, layer_ubicaciones]
    if layer_warnings:
        initial_layers.append(layer_warnings)
else:
    initial_layers = []

deck_pane = pn.pane.DeckGL(
    pdk.Deck(
        initial_view_state=INITIAL_VIEW_STATE,
        layers=initial_layers,
        map_style=MAP_STYLE,
        tooltip=TOOLTIP_CONFIG
    ),
    min_height=600,
    sizing_mode='stretch_both'
)

# ==============================================================================
# 3. COMPONENTES DEL DASHBOARD
# ==============================================================================

btn_predict = pn.widgets.Button(
    name='🚨 EJECUTAR ANÁLISIS IA', 
    button_type='danger', 
    icon='brain',
    height=60,
    sizing_mode='stretch_width'
)

# KPIS HUMANIZADOS CON NEÓN
kpi_total = pn.indicators.Number(
    name='Amenazas Detectadas', 
    value=0, 
    format='{value}', 
    colors=[(100, 'white'), (1000, 'orange'), (50000, 'red')],
    font_size='28pt'
)

kpi_status = pn.pane.Markdown("## 🟢 Sistema Estable", styles={'color': '#00ff00', 'text-align': 'center'})

# KPI Riesgo con Markdown custom para control total
kpi_prob_label = pn.pane.Markdown(
    "<div style='text-align:center;'><p style='color:#888; font-size:12pt; margin:0;'>Nivel de Riesgo</p><h2 style='color:#00FF00; font-size:24pt; margin:5px 0;'>BAJO</h2></div>",
    sizing_mode='stretch_width'
)

# KPI Zona Crítica con Markdown custom
kpi_critical_zone = pn.pane.Markdown(
    "<div style='text-align:center;'><p style='color:#888; font-size:12pt; margin:0;'>Zona Crítica</p><h2 style='color:#00FFFF; font-size:24pt; margin:5px 0;'>-</h2></div>",
    sizing_mode='stretch_width'
)

# KPI: TASA DE MITIGACIÓN PRE-CRIMEN (Nuevo)
kpi_mitigation_rate = pn.pane.Markdown(
    "<div style='text-align:center; border: 2px solid #39FF14; border-radius:8px; padding:10px;'>" +
    "<p style='color:#888; font-size:12pt; margin:0;'>TASA DE MITIGACIÓN</p>" +
    "<h1 style='color:#39FF14; font-size:32pt; margin:5px 0; text-shadow: 0 0 15px #39FF14;'>96.5%</h1>" +
    "<p style='color:#39FF14; font-size:10pt; margin:0;'>ESTADO: BAJO CONTROL OPERATIVO</p>" +
    "</div>",
    sizing_mode='stretch_width'
)

# Contenedor de gráficos
row_plots = pn.Row(min_height=350, sizing_mode='stretch_width')

# Mapa de Zonas Peligrosas (pequeño)
zones_map_pane = pn.pane.DeckGL(
    pdk.Deck(
        initial_view_state=INITIAL_VIEW_STATE,
        layers=[],
        map_style=MAP_STYLE,
        tooltip=TOOLTIP_CONFIG
    ),
    min_height=400,
    height=400,
    sizing_mode='stretch_width'
)

# KPI: Índice de Amenaza Inminente (Gauge visual con colores dinámicos)
kpi_threat_index = pn.indicators.Gauge(
    name='🎯 Amenaza', 
    value=0, 
    bounds=(0, 100),
    colors=[(50, '#00FFFF'), (75, '#FFA500'), (100, '#FF0033')],  # Cian -> Naranja -> Rojo Alerta
    format='{value}%',
    width=350,
    height=300  # Más espacio para evitar superposición
)

# Grafo de Red de Sospechosos (contenedor dinámico)
network_plot = pn.Column()

# NUEVO: Mapa 3D Geoespacial de Red de Sospechosos (Pydeck)
network_3d_map_pane = pn.pane.DeckGL(
    pdk.Deck(
        initial_view_state=INITIAL_VIEW_STATE,
        layers=[],
        map_style=MAP_STYLE
    ),
    min_height=500,
    sizing_mode='stretch_width'
)

# Radar Chart Temporal (24h)
radar_chart_pane = pn.pane.Bokeh(sizing_mode='stretch_width', min_height=400)

# Heatmap de Densidad Mejorado
heatmap_pane = pn.pane.DeckGL(
    pdk.Deck(
        initial_view_state=INITIAL_VIEW_STATE,
        layers=[],
        map_style=MAP_STYLE
    ),
    min_height=400,
    sizing_mode='stretch_width'
)

# ==============================================================================
# 4. LÓGICA DE CALLBACK
# ==============================================================================

def run_prediction(event):
    if not service:
        pn.state.notifications.error("El servicio de IA no está disponible.")
        return

    pn.state.notifications.info("Iniciando escaneo biométrico...", duration=1500)
    btn_predict.loading = True
    btn_predict.name = "PROCESANDO..."
    
    try:
        # Llamada al servicio con umbrales más bajos para obtener más predicciones
        df_pred = service.predict_threats(risk_threshold=0.4, danger_threshold=0.3)
        
        if df_pred.empty:
            pn.state.notifications.success("Análisis completado: Ciudad Segura.")
            
            # Reset
            deck_pane.object = pdk.Deck(
                initial_view_state=INITIAL_VIEW_STATE,
                layers=initial_layers,
                map_style=MAP_STYLE,
                tooltip=TOOLTIP_CONFIG
            )
            
            kpi_total.value = 0
            kpi_status.object = "## 🟢 Sistema Estable"
            kpi_status.styles = {'color': '#00ff00', 'text-align': 'center'}
            kpi_prob_label.object = "<div style='text-align:center;'><p style='color:#888; font-size:12pt; margin:0;'>Nivel de Riesgo</p><h2 style='color:#00FF00; font-size:24pt; margin:5px 0;'>BAJO</h2></div>"
            kpi_threat_index.value = 0
            kpi_critical_zone.object = "<div style='text-align:center;'><p style='color:#888; font-size:12pt; margin:0;'>Zona Crítica</p><h2 style='color:#00FFFF; font-size:24pt; margin:5px 0;'>-</h2></div>"
            row_plots.objects = [pn.pane.Markdown("### 🛡️ Sin actividad criminal detectada.")]
            network_plot.clear()
            network_plot.append(pn.pane.Markdown("### 🛡️ Sin red de sospechosos."))
            
        else:
            count = len(df_pred)
            max_prob = df_pred['probabilidad'].max()
            avg_prob = df_pred['probabilidad'].mean()
            
            # Convertir probabilidad promedio a porcentaje para el gauge
            threat_index_percent = min(int(avg_prob * 100), 100)
            
            # 1. KPI AMENAZAS
            kpi_total.value = count
            if count > 50000:
                text_status = "## 🔴 ALERTA CRÍTICA"
                color_status = "#ff0000" # Rojo neón
            elif count > 10000:
                text_status = "## 🟠 ALERTA ALTA"
                color_status = "#ffaa00"
            else:
                text_status = "## 🟡 ALERTA MODERADA"
                color_status = "#ffff00"
            
            kpi_status.object = text_status
            kpi_status.styles = {'color': color_status, 'text-align': 'center'}
            
            # 2. KPI PROBABILIDAD (HUMANIZADO CON COLORES NEÓN)
            if max_prob > 0.90:
                risk_text = "RIESGO INMINENTE"
                risk_color = "#FF0000"  # Rojo neón
            elif max_prob > 0.70:
                risk_text = "RIESGO ALTO"
                risk_color = "#FF8C00"  # Naranja neón
            else:
                risk_text = "RIESGO MEDIO"
                risk_color = "#FFA500"  # Naranja claro
            
            kpi_prob_label.object = f"<div style='text-align:center; border: 2px solid {risk_color}; border-radius:8px; padding:10px;'><p style='color:#888; font-size:12pt; margin:0;'>Nivel de Riesgo</p><h2 style='color:{risk_color}; font-size:26pt; margin:5px 0; text-shadow: 0 0 10px {risk_color};'>{risk_text}</h2></div>"
            
            # ACTUALIZAR GAUGE DE AMENAZA INMINENTE (colores automáticos por configuración)
            kpi_threat_index.value = threat_index_percent
            
            # ACTUALIZAR KPI DE TASA DE MITIGACIÓN (dinámico)
            # Lógica dummy: Asumimos que neutralizamos la mayoría de amenazas
            amenazas_neutralizadas = max(1, int(count * 0.965))  # 96.5% mitigado
            tasa_mitigacion = (amenazas_neutralizadas / count) * 100 if count > 0 else 96.5
            
            # Color según eficacia
            if tasa_mitigacion >= 95:
                mitigation_color = "#39FF14"  # Verde neón
                mitigation_status = "BAJO CONTROL OPERATIVO"
            elif tasa_mitigacion >= 85:
                mitigation_color = "#00FFFF"  # Cian
                mitigation_status = "CONTROL PARCIAL"
            else:
                mitigation_color = "#FFA500"  # Naranja
                mitigation_status = "REQUIERE ATENCIÓN"
            
            kpi_mitigation_rate.object = (
                f"<div style='text-align:center; border: 2px solid {mitigation_color}; border-radius:8px; padding:10px;'>" +
                f"<p style='color:#888; font-size:12pt; margin:0;'>TASA DE MITIGACIÓN</p>" +
                f"<h1 style='color:{mitigation_color}; font-size:32pt; margin:5px 0; text-shadow: 0 0 15px {mitigation_color};'>{tasa_mitigacion:.1f}%</h1>" +
                f"<p style='color:{mitigation_color}; font-size:10pt; margin:0;'>ESTADO: {mitigation_status}</p>" +
                "</div>"
            )
                
            # 3. KPI ZONA CRÍTICA (ID Simple CON NEÓN)
            top_zone_id = df_pred['id_ubicacion'].mode()[0]
            zone_color = "#FF3333" if threat_index_percent > 75 else "#FF8C00" if threat_index_percent > 50 else "#00FFFF"
            kpi_critical_zone.object = f"<div style='text-align:center; border: 2px solid {zone_color}; border-radius:8px; padding:10px;'><p style='color:#888; font-size:12pt; margin:0;'>Zona Crítica</p><h2 style='color:{zone_color}; font-size:26pt; margin:5px 0; text-shadow: 0 0 10px {zone_color};'>{top_zone_id}</h2></div>"
            
            # 4. ACTUALIZAR MAPA (sin puntos de crimen predichos, van en el mapa de abajo)
            deck_pane.object = pdk.Deck(
                initial_view_state=INITIAL_VIEW_STATE,
                layers=initial_layers,
                map_style=MAP_STYLE,
                tooltip=TOOLTIP_CONFIG
            )
            
            # 5. MAPA DE ZONAS PELIGROSAS (DeckGL con color por peligrosidad)
            # Agrupar por ubicación y calcular peligrosidad
            df_zonas = df_pred.groupby('id_ubicacion').agg({
                'lat_ubicacion': 'first',
                'lon_ubicacion': 'first',
                'peligrosidad_zona': 'first',
                'id_ubicacion': 'count'
            }).rename(columns={'id_ubicacion': 'num_amenazas'}).reset_index(drop=True)
            
            # Agregar color según peligrosidad
            df_zonas['color'] = df_zonas['peligrosidad_zona'].apply(peligrosidad_to_color)
            
            # Crear capa DeckGL de zonas con saturación de color
            layer_zonas = pdk.Layer(
                "ScatterplotLayer",
                data=df_zonas,
                get_position=["lon_ubicacion", "lat_ubicacion"],
                get_fill_color="color",  # Usar la columna color calculada
                get_radius=80,  # Radio para visualizar claramente las zonas
                stroked=True,
                get_line_color=[255, 255, 255, 255],  # Borde blanco
                line_width_min_pixels=2,
                pickable=True,
                radius_min_pixels=5
            )
            
            # Actualizar el mapa de zonas
            zones_map_pane.object = pdk.Deck(
                initial_view_state=INITIAL_VIEW_STATE,
                layers=[layer_zonas],
                map_style=MAP_STYLE,
                tooltip={
                    "html": "<b>Zona Peligrosa</b><br>Peligrosidad: {peligrosidad_zona:.2f}<br>Amenazas: {num_amenazas}",
                    "style": {"backgroundColor": "#1a1a1a", "color": "#ffffff"}
                }
            )
            
            row_plots.objects = [zones_map_pane]
            
            # 6. GENERAR NETWORK GRAPH DE SOSPECHOSOS
            try:
                # Obtener datos de sospechosos de alto riesgo
                nodos_df, edges_df = service.get_suspect_network(limit_nodes=25)
                
                # Filtrar solo sospechosos de alto riesgo (riesgo > 0.5)
                if not nodos_df.empty:
                    sospechosos_altos = nodos_df[nodos_df['risk'] > 0.5].copy()
                    
                    # Si hay aristas, filtrar para incluir solo nodos de alto riesgo
                    if not edges_df.empty and 'source' in edges_df.columns:
                        # Obtener IDs de sospechosos de alto riesgo
                        high_risk_ids = set(sospechosos_altos['id'].values)
                        
                        # Filtrar aristas que conecten nodos de alto riesgo
                        edges_filtered = edges_df[
                            edges_df['source'].isin(high_risk_ids) & 
                            edges_df['target'].isin(high_risk_ids)
                        ].copy()
                    else:
                        edges_filtered = pd.DataFrame()
                    
                    # Crear network graph con Bokeh (2D)
                    network_graph = create_network_graph(sospechosos_altos, edges_filtered)
                    
                    # Crear mapa 3D geoespacial con Pydeck
                    network_3d_deck = create_3d_network_map(sospechosos_altos)
                    
                    network_plot.clear()
                    network_plot.append(pn.pane.Bokeh(network_graph, sizing_mode='stretch_width'))
                    
                    # Actualizar mapa 3D
                    network_3d_map_pane.object = network_3d_deck
                else:
                    network_plot.clear()
                    network_plot.append(pn.pane.Markdown("### ⚠️ No hay datos de sospechosos de alto riesgo."))
                    
                    # Reset mapa 3D
                    network_3d_map_pane.object = pdk.Deck(
                        initial_view_state=INITIAL_VIEW_STATE,
                        layers=[],
                        map_style=MAP_STYLE
                    )
            except Exception as e:
                print(f"Error generando network graph: {e}")
                import traceback
                traceback.print_exc()
                network_plot.clear()
                network_plot.append(pn.pane.Markdown(f"### ⚠️ Error generando grafo: {str(e)}"))
            
            # 7. GENERAR RADAR CHART TEMPORAL
            try:
                radar_plot = create_temporal_radar_chart(df_pred)
                radar_chart_pane.object = radar_plot
            except Exception as e:
                print(f"Error generando radar chart: {e}")
            
            # 8. GENERAR HEATMAP DE DENSIDAD MEJORADO
            try:
                # HeatmapLayer para efecto de difuminado suave (sin cuadrados)
                layer_heatmap = pdk.Layer(
                    "HeatmapLayer",
                    data=df_pred,
                    get_position=["lon_ubicacion", "lat_ubicacion"],
                    get_weight="probabilidad",
                    radiusPixels=80,      # Radio más grande para difuminado suave
                    intensity=1.5,        # Intensidad moderada
                    threshold=0.02,       # Threshold bajo para más cobertura
                    colorRange=[
                        [0, 0, 255, 0],       # Azul totalmente transparente (fondo)
                        [0, 255, 255, 80],    # Cian (bajo riesgo)
                        [0, 255, 0, 120],     # Verde (riesgo medio-bajo)
                        [255, 255, 0, 160],   # Amarillo (riesgo medio)
                        [255, 128, 0, 200],   # Naranja (riesgo alto)
                        [255, 0, 0, 255]      # Rojo intenso (riesgo crítico)
                    ]
                )
                
                # Usar solo HeatmapLayer para efecto suave y continuo
                heatmap_pane.object = pdk.Deck(
                    initial_view_state=INITIAL_VIEW_STATE,
                    layers=[layer_heatmap],
                    map_style=MAP_STYLE,
                    tooltip={
                        "html": "<b>🔥 Densidad de Crimen</b><br>Zona de alta actividad criminal",
                        "style": {"backgroundColor": "#000", "color": "#FF0000"}
                    }
                )
            except Exception as e:
                print(f"Error generando heatmap: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        error_msg = f"Error crítico: {str(e)}"
        print(error_msg)
        pn.state.notifications.error(error_msg, duration=0) # Persistente
        import traceback
        traceback.print_exc()
        
    finally:
        btn_predict.loading = False
        btn_predict.name = "🚨 EJECUTAR ANÁLISIS IA"

btn_predict.on_click(run_prediction)

# ==============================================================================
# 5. LAYOUT FINAL
# ==============================================================================

template = pn.template.MaterialTemplate(
    title='MINORITY REPORT MADRID | PRE-CRIME UNIT',
    theme='dark',
    header_background='#000000',
    sidebar=[
        pn.pane.Markdown("### 🎛️ Centro de Comando"),
        btn_predict,
        pn.pane.Markdown("---"),
        pn.pane.Markdown("### Estado de Red"),
        pn.Row(pn.indicators.BooleanStatus(value=True, width=20, height=20), pn.pane.Markdown("Neural Link: **ONLINE**")),
        pn.Row(pn.indicators.BooleanStatus(value=True, width=20, height=20), pn.pane.Markdown("Neo4j Core: **CONNECTED**")),
        pn.layout.Divider(),
        pn.pane.Markdown("*\"Lo que no se mide, no se puede prevenir.\"*")
    ],
    main=[
        pn.Row(
            pn.Card(kpi_status, title="Estado", sizing_mode='stretch_width', styles={'border': '2px solid #00ff00', 'background': '#0a0a0a'}),
            pn.Card(kpi_total, title="Amenazas Activas", sizing_mode='stretch_width', styles={'background': '#0a0a0a'}),
            pn.Card(kpi_prob_label, title="", sizing_mode='stretch_width', styles={'background': '#0a0a0a'}),
            pn.Card(kpi_critical_zone, title="", sizing_mode='stretch_width', styles={'background': '#0a0a0a'}),
            pn.Card(kpi_mitigation_rate, title="", sizing_mode='stretch_width', styles={'background': '#0a0a0a'}),
            sizing_mode='stretch_width'
        ),
        pn.Row(
            pn.Card(kpi_threat_index, title="", sizing_mode='stretch_width', styles={'background': '#000'}),
            pn.Card(radar_chart_pane, title="🕒 Patrón Temporal de Riesgo", sizing_mode='stretch_width', header_background='#111', styles={'background': '#000'}),
            sizing_mode='stretch_width'
        ),
        pn.Card(deck_pane, title="📍 Mapa Táctico 3D - LEYENDA: 🔵 Personas | 🟡 Ubicaciones | 🔴 Crímenes", sizing_mode='stretch_both', min_height=600, header_background='#111', styles={'background': '#000'}),
        pn.Row(
            pn.Card(heatmap_pane, title="🔥 Heatmap de Densidad Criminal", sizing_mode='stretch_width', header_background='#111', styles={'background': '#000'}),
            pn.Column(
                pn.Card(network_plot, title="🕸️ Red de Conexiones - Sospechosos Alto Riesgo", sizing_mode='stretch_width', header_background='#111', styles={'background': '#000'}),
                pn.Card(network_3d_map_pane, title="🌐 Mapa Geoespacial 3D - Red Criminal", sizing_mode='stretch_width', header_background='#111', styles={'background': '#000'}),
                sizing_mode='stretch_width'
            ),
            sizing_mode='stretch_width'
        )
    ]
)

template.servable()
if __name__ == '__main__':
    pn.serve(template, show=True, port=5006)
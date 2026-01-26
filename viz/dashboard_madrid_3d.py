
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
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category20
import torch

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
    zoom=11, 
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
        get_color=[0, 100, 255, 140], # Azul cian translúcido
        get_radius=10,  # Reducido de 20 a 10
        pickable=True,
        radius_min_pixels=1
    )
    
    layer_ubicaciones = pdk.Layer(
        "ScatterplotLayer",
        data=df_ubicaciones_base,
        get_position=["lon", "lat"],
        get_color=[255, 200, 0, 140], # Ámbar translúcido
        get_radius=40,  # Reducido de 80 a 40
        pickable=True,
        radius_min_pixels=2
    )
    
    layer_warnings = pdk.Layer(
        "ScatterplotLayer",
        data=df_warnings_base,
        get_position=["lon", "lat"],
        get_color=[255, 100, 100, 180], # Rojo translúcido
        get_radius=15,  # Tamaño medio entre personas y ubicaciones
        pickable=True,
        radius_min_pixels=1,
        stroked=True,
        get_line_color=[255, 0, 0, 255],
        line_width_min_pixels=1
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

# KPIS HUMANIZADOS
kpi_total = pn.indicators.Number(
    name='Amenazas Detectadas', 
    value=0, 
    format='{value}', 
    colors=[(100, 'white'), (1000, 'orange'), (50000, 'red')]
)

kpi_status = pn.pane.Markdown("## 🟢 Sistema Estable", styles={'color': '#00ff00', 'text-align': 'center'})

kpi_prob_label = pn.indicators.String(name='Nivel de Riesgo', value='BAJO')

kpi_critical_zone = pn.indicators.String(name='Zona Crítica', value='-')

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

# KPI: Índice de Amenaza Inminente (Gauge visual)
kpi_threat_index = pn.indicators.Gauge(
    name='Índice de Amenaza Inminente (%)', 
    value=0, 
    bounds=(0, 100),
    width=300,
    height=300
)

# Grafo de Red de Sospechosos (contenedor dinámico)
network_plot = pn.Column()

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
            kpi_prob_label.value = "BAJO"
            kpi_threat_index.value = 0
            kpi_threat_index.styles = {'color': '#00ffff'}
            kpi_critical_zone.value = "-"
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
            
            # 2. KPI PROBABILIDAD (HUMANIZADO)
            if max_prob > 0.90:
                kpi_prob_label.value = "RIESGO INMINENTE"
            elif max_prob > 0.70:
                kpi_prob_label.value = "RIESGO ALTO"
            else:
                kpi_prob_label.value = "RIESGO MEDIO"
            
            # ACTUALIZAR GAUGE DE AMENAZA INMINENTE
            if threat_index_percent < 50:
                kpi_threat_index.styles = {'color': '#00ffff'}  # Cian
            else:
                kpi_threat_index.styles = {'color': '#ff0000', 'animation': 'blink 1s infinite'}  # Rojo parpadeante
            kpi_threat_index.value = threat_index_percent
                
            # 3. KPI ZONA CRÍTICA (ID Simple)
            top_zone_id = df_pred['id_ubicacion'].mode()[0]
            kpi_critical_zone.value = top_zone_id
            
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
            
            # 6. MAPA DE SOSPECHOSOS Y ZONAS PELIGROSAS
            try:
                # Obtener datos de sospechosos de alto riesgo
                nodos_df, _ = service.get_suspect_network(limit_nodes=20)
                
                if not nodos_df.empty:
                    # Filtrar solo sospechosos de alto riesgo (riesgo > 0.5)
                    sospechosos_altos = nodos_df[nodos_df['risk'] > 0.5].copy()
                    
                    # Obtener coordenadas de personas de alto riesgo
                    x_pers = service.data['Persona'].x.cpu().numpy()
                    x_locs = service.data['Ubicacion'].x.cpu().numpy()
                    
                    # Crear DataFrame de sospechosos con coordenadas
                    if len(sospechosos_altos) > 0:
                        sospechosos_coords = []
                        for _, row in sospechosos_altos.iterrows():
                            # Extraer el número del ID (formato: 'P_255' -> 255)
                            id_str = str(row['id'])
                            persona_id = int(id_str.split('_')[-1]) if '_' in id_str else int(id_str)
                            
                            if persona_id < len(x_pers):
                                sospechosos_coords.append({
                                    'lat': denormalize_lat(x_pers[persona_id, 1]),
                                    'lon': denormalize_lon(x_pers[persona_id, 2]),
                                    'risk': row['risk'],
                                    'id_visual': f'S_{persona_id}'
                                })
                        
                        df_sospechosos = pd.DataFrame(sospechosos_coords)
                        
                        # Crear capa de sospechosos (rojo oscuro)
                        layer_sospechosos = pdk.Layer(
                            "ScatterplotLayer",
                            data=df_sospechosos,
                            get_position=["lon", "lat"],
                            get_fill_color=[200, 0, 0, 220],  # Rojo oscuro
                            get_radius=30,
                            stroked=True,
                            get_line_color=[255, 255, 255, 255],
                            line_width_min_pixels=2,
                            pickable=True,
                            radius_min_pixels=5
                        )
                    else:
                        layer_sospechosos = None
                    
                    # Crear capa de zonas peligrosas con colores cálido-frío
                    df_zonas_map = df_zonas.copy()  # Usar df_zonas calculado arriba
                    df_zonas_map['color'] = df_zonas_map['peligrosidad_zona'].apply(peligrosidad_to_color)
                    
                    layer_zonas_suspects = pdk.Layer(
                        "ScatterplotLayer",
                        data=df_zonas_map,
                        get_position=["lon_ubicacion", "lat_ubicacion"],
                        get_fill_color="color",
                        get_radius=60,
                        stroked=True,
                        get_line_color=[255, 255, 255, 200],
                        line_width_min_pixels=1,
                        pickable=True,
                        radius_min_pixels=3
                    )
                    
                    # Armar el mapa con ambas capas
                    layers_suspects = [layer_zonas_suspects]
                    if layer_sospechosos is not None:
                        layers_suspects.append(layer_sospechosos)
                    
                    suspects_map_pane = pn.pane.DeckGL(
                        pdk.Deck(
                            initial_view_state=INITIAL_VIEW_STATE,
                            layers=layers_suspects,
                            map_style=MAP_STYLE,
                            tooltip={
                                "html": "<b>Sospechoso/Zona</b><br>ID: {id_visual}",
                                "style": {"backgroundColor": "#1a1a1a", "color": "#ffffff"}
                            }
                        ),
                        min_height=600,
                        sizing_mode='stretch_both'
                    )
                    
                    network_plot.clear()
                    network_plot.append(suspects_map_pane)
                else:
                    network_plot.clear()
                    network_plot.append(pn.pane.Markdown("### ⚠️ No hay datos de sospechosos."))
            except Exception as e:
                print(f"Error generando mapa de sospechosos: {e}")
                import traceback
                traceback.print_exc()
                network_plot.clear()
                network_plot.append(pn.pane.Markdown(f"### ⚠️ Error generando mapa: {str(e)}"))

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
            pn.Card(kpi_status, title="Estado", sizing_mode='stretch_width'),
            pn.Card(kpi_total, title="Amenazas Activas", sizing_mode='stretch_width'),
            pn.Card(kpi_prob_label, title="Probabilidad Máxima", sizing_mode='stretch_width'),
            pn.Card(kpi_critical_zone, title="Zona Crítica", sizing_mode='stretch_width'),
            sizing_mode='stretch_width'
        ),
        pn.Row(
            pn.Card(kpi_threat_index, title="🎯 Índice de Amenaza Inminente", sizing_mode='stretch_width'),
            sizing_mode='stretch_width'
        ),
        pn.Card(deck_pane, title="📍 Visualización Geoespacial en Tiempo Real", sizing_mode='stretch_both', min_height=600, header_background='#111'),
        pn.Row(
            pn.Card(row_plots, title="🌡️ Mapa de Saturación de Zonas Peligrosas", sizing_mode='stretch_width', header_background='#111'),
            pn.Card(network_plot, title="� Mapa de Sospechosos y Zonas Peligrosas", sizing_mode='stretch_width', header_background='#111'),
            sizing_mode='stretch_width'
        )
    ]
)

template.servable()
if __name__ == '__main__':
    pn.serve(template, show=True, port=5006)
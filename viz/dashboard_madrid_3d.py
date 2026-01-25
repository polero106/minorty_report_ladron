
import panel as pn
import pydeck as pdk
import pandas as pd
import numpy as np
import sys
import os
import hvplot.pandas

# Importar el servicio
# Importar el servicio

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

# Mapeo Ficticio de Barrios
BARRIOS = {
    0: "Centro", 1: "Arganzuela", 2: "Retiro", 3: "Salamanca", 
    4: "Chamartín", 5: "Tetuán", 6: "Chamberí", 7: "Fuencarral-El Pardo",
    8: "Moncloa-Aravaca", 9: "Latina", 10: "Carabanchel", 11: "Usera",
    12: "Puente de Vallecas", 13: "Moratalaz", 14: "Ciudad Lineal",
    15: "Hortaleza", 16: "Villaverde", 17: "Villa de Vallecas",
    18: "Vicálvaro", 19: "San Blas-Canillejas", 20: "Barajas"
}

def get_barrio_name(u_id_str):
    """Convierte 'U_10' -> 'Carabanchel'"""
    try:
        # Extraer número del ID string (ej: U_10 -> 10)
        idx = int(str(u_id_str).split('_')[1])
        # Usar módulo para asignar un nombre siempre
        return BARRIOS.get(idx % 21, f"Distrito {idx}")
    except:
        return "Zona Desconocida"

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
        'id_visual': [f'{get_barrio_name(f"U_{i}")}' for i in range(len(x_locs))]
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
        # Llamada al servicio
        df_pred = service.predict_threats(risk_threshold=0.6, danger_threshold=0.5)
        
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
            kpi_critical_zone.value = "-"
            row_plots.objects = [pn.pane.Markdown("### 🛡️ Sin actividad criminal detectada.")]
            
        else:
            count = len(df_pred)
            max_prob = df_pred['probabilidad'].max()
            
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
                
            # 3. KPI ZONA CRÍTICA (Nombre Real)
            top_zone_id = df_pred['id_ubicacion'].mode()[0]
            kpi_critical_zone.value = get_barrio_name(top_zone_id)
            
            # 4. ACTUALIZAR MAPA
            # Capa 1: Arcos de amenaza
            layer_arcs = pdk.Layer(
                "ArcLayer",
                data=df_pred,
                get_source_position=["lon_sujeto", "lat_sujeto"],
                get_target_position=["lon_ubicacion", "lat_ubicacion"],
                get_source_color=[0, 255, 255, 80], # Cyan tenue
                get_target_color=[255, 0, 0, 200],  # Rojo intenso
                get_width=2,
                pickable=True
            )
            
            # Capa 2: Puntos de impacto (Crimen previsto)
            layer_crimes = pdk.Layer(
                "ScatterplotLayer",
                data=df_pred,
                get_position=["lon_ubicacion", "lat_ubicacion"],
                get_fill_color=[255, 0, 0, 200],
                get_line_color=[255, 255, 255],
                get_radius=25,  # Reducido de 50 a 25
                stroked=True,
                filled=True,
                radius_min_pixels=3,
                pickable=True
            )

            deck_pane.object = pdk.Deck(
                initial_view_state=INITIAL_VIEW_STATE,
                layers=initial_layers + [layer_arcs, layer_crimes],
                map_style=MAP_STYLE,
                tooltip=TOOLTIP_CONFIG
            )
            
            # 5. GRÁFICOS
            # Ranking Top 5 Barrios (Bar Chart Horizontal)
            # Primero mapeamos IDs a Nombres
            df_pred['Barrio'] = df_pred['id_ubicacion'].apply(get_barrio_name)
            
            # Agrupar y contar
            df_ranking = df_pred['Barrio'].value_counts().head(5).reset_index()
            df_ranking.columns = ['Barrio', 'Amenazas']
            df_ranking = df_ranking.sort_values(by='Amenazas', ascending=True) # Para que el Top 1 salga arriba en hbar
            
            bar_plot = df_ranking.hvplot.barh(
                x='Barrio', 
                y='Amenazas', 
                title='🔥 Top 5 Zonas de Riesgo',
                color='#ff0040',
                height=300,
                responsive=True,
                grid=True
            ).opts(fontsize={'title': 14, 'labels': 12, 'xticks': 10, 'yticks': 10})
            
            heatmap_plot = df_pred.hvplot.points(
                'lon_ubicacion', 
                'lat_ubicacion', 
                c='probabilidad',
                cmap='inferno', 
                size=80, 
                title='🗺️ Mapa de Calor Criminal',
                height=300,
                responsive=True,
                colorbar=True,
                xaxis=None, 
                yaxis=None
            ).opts(bgcolor='#1a1a1a')
            
            row_plots.objects = [bar_plot, heatmap_plot]

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
        pn.Card(deck_pane, title="📍 Visualización Geoespacial en Tiempo Real", sizing_mode='stretch_both', min_height=600, header_background='#111'),
        pn.Card(row_plots, title="📊 Analítica Predictiva", sizing_mode='stretch_width', header_background='#111')
    ]
)

template.servable()
if __name__ == '__main__':
    pn.serve(template, show=True, port=5006)
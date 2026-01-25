import panel as pn
import pydeck as pdk
import pandas as pd
import numpy as np
import sys
import os
import hvplot.pandas

# Importar el servicio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prediction_service import PredictionService

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
    zoom=12, 
    pitch=45, 
    bearing=0
)

# Definir tooltips
TOOLTIP_CONFIG = {
    "html": "<b>Tipo:</b> {tipo}<br><b>ID:</b> {id_visual}",
    "style": {"backgroundColor": "steelblue", "color": "white"}
}

# Preparar capas iniciales
if service:
    # Extraemos datos a CPU para visualización
    # Asumiendo que service.data tiene los tensores en GPU/CPU
    x_pers = service.data['Persona'].x.cpu().numpy()
    x_locs = service.data['Ubicacion'].x.cpu().numpy()
    
    # Índices: 0=Risk/Danger, 1=Lat, 2=Lon (según tu ETL)
    # Creamos DataFrames para PyDeck
    df_personas_base = pd.DataFrame({
        'lat': x_pers[:, 1],
        'lon': x_pers[:, 2],
        'tipo': 'Persona',
        'id_visual': [f'P_{i}' for i in range(len(x_pers))]
    })
    
    df_ubicaciones_base = pd.DataFrame({
        'lat': x_locs[:, 1],
        'lon': x_locs[:, 2],
        'tipo': 'Ubicación',
        'id_visual': [f'U_{i}' for i in range(len(x_locs))]
    })
    
    layer_personas = pdk.Layer(
        "ScatterplotLayer",
        data=df_personas_base,
        get_position=["lon", "lat"],
        get_color=[0, 100, 255, 100], # Azul translúcido
        get_radius=15,
        pickable=True
    )
    
    layer_ubicaciones = pdk.Layer(
        "ScatterplotLayer",
        data=df_ubicaciones_base,
        get_position=["lon", "lat"],
        get_color=[255, 200, 0, 150], # Amarillo translúcido
        get_radius=30,
        pickable=True
    )
    
    initial_layers = [layer_personas, layer_ubicaciones]
else:
    initial_layers = []

# --- CORRECCIÓN AQUÍ: Usamos pdk.Deck objeto, NO un diccionario ---
initial_deck = pdk.Deck(
    initial_view_state=INITIAL_VIEW_STATE,
    layers=initial_layers,
    map_style=MAP_STYLE,
    tooltip=TOOLTIP_CONFIG
)

deck_pane = pn.pane.DeckGL(
    initial_deck,
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
    height=50,
    sizing_mode='stretch_width'
)

kpi_total = pn.indicators.Number(name='Amenazas Activas', value=0, format='{value}', colors=[(10, 'green'), (50, 'orange'), (100, 'red')])
kpi_max_prob = pn.indicators.Number(name='Probabilidad Máxima', value=0.0, format='{value:.1%}')
kpi_danger_zone = pn.indicators.String(name='Zona Crítica', value='N/A')

row_plots = pn.Row(min_height=300, sizing_mode='stretch_width')

# ==============================================================================
# 4. LÓGICA DE CALLBACK
# ==============================================================================

def run_prediction(event):
    if not service:
        pn.state.notifications.error("El servicio de IA no está disponible.")
        return

    pn.state.notifications.info("Iniciando escaneo de la ciudad...", duration=2000)
    btn_predict.loading = True
    
    try:
        # Llamada al servicio
        df_pred = service.predict_threats(risk_threshold=0.6, danger_threshold=0.5)
        
        if df_pred.empty:
            pn.state.notifications.success("Ciudad Segura: No se detectaron amenazas.")
            
            # Restaurar mapa original (creando nuevo Deck)
            deck_pane.object = pdk.Deck(
                initial_view_state=INITIAL_VIEW_STATE,
                layers=initial_layers,
                map_style=MAP_STYLE,
                tooltip=TOOLTIP_CONFIG
            )
            
            kpi_total.value = 0
            kpi_max_prob.value = 0.0
            kpi_danger_zone.value = "N/A"
            row_plots.objects = [pn.pane.Markdown("### No hay datos de riesgo.")]
            
        else:
            count = len(df_pred)
            pn.state.notifications.warning(f"ALERTA: Se detectaron {count} posibles crímenes.")
            
            # Actualizar KPIs
            kpi_total.value = count
            kpi_max_prob.value = df_pred['probabilidad'].max()
            top_zone = df_pred['id_ubicacion'].mode()[0]
            kpi_danger_zone.value = str(top_zone)
            
            # Nueva Capa de Predicciones
            layer_predictions = pdk.Layer(
                "ArcLayer",
                data=df_pred,
                get_source_position=["lon_sujeto", "lat_sujeto"],
                get_target_position=["lon_ubicacion", "lat_ubicacion"],
                get_source_color=[0, 255, 255, 120], # Cyan origen
                get_target_color=[255, 0, 0, 255],   # Rojo neón destino
                get_width=4,
                pickable=True
            )
            
            # Combinar capas
            new_layers = initial_layers + [layer_predictions]
            
            # --- CORRECCIÓN AQUÍ: Asignamos un nuevo pdk.Deck ---
            deck_pane.object = pdk.Deck(
                initial_view_state=INITIAL_VIEW_STATE,
                layers=new_layers,
                map_style=MAP_STYLE,
                tooltip=TOOLTIP_CONFIG
            )
            
            # Gráficos
            hist_plot = df_pred.hvplot.hist(
                'probabilidad', 
                bins=10, 
                title='Distribución de Probabilidad', 
                color='red',
                height=300,
                responsive=True
            )
            
            heatmap_plot = df_pred.hvplot.points(
                'lon_ubicacion', 
                'lat_ubicacion', 
                c='probabilidad',
                cmap='plasma', 
                size=100, 
                title='Zonas Calientes',
                height=300,
                responsive=True,
                colorbar=True
            )
            
            row_plots.objects = [hist_plot, heatmap_plot]

    except Exception as e:
        error_msg = f"Error crítico: {str(e)}"
        print(error_msg)
        pn.state.notifications.error(error_msg)
        
    finally:
        btn_predict.loading = False

btn_predict.on_click(run_prediction)

# ==============================================================================
# 5. LAYOUT FINAL
# ==============================================================================

template = pn.template.MaterialTemplate(
    title='MINORITY REPORT MADRID | PRE-CRIME DASHBOARD',
    theme='dark',
    sidebar=[
        pn.pane.Markdown("## Controles"),
        btn_predict,
        pn.pane.Markdown("---"),
        pn.pane.Markdown("### Estado del Sistema"),
        pn.indicators.BooleanStatus(name='IA Online', value=True, color='success'),
        pn.indicators.BooleanStatus(name='Neo4j Conectado', value=True, color='success')
    ],
    main=[
        pn.Row(kpi_total, kpi_max_prob, kpi_danger_zone),
        pn.Card(deck_pane, title="Visualización Geoespacial 3D", sizing_mode='stretch_both', min_height=600),
        pn.Card(row_plots, title="Analítica Avanzada", sizing_mode='stretch_width')
    ]
)

template.servable()
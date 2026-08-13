import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ==========================================
# CONFIGURATION INITIALE
# ==========================================
st.set_page_config(
    page_title="ODIN ERP - Heatmap & Action Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# DESIGN SYSTEM (CSS GLOBAL) - PREMIUM DARK THEME
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }
    
    .stApp { 
        background: #0f172a; 
        color: #f8fafc;
    }
    
    /* Header Premium */
    .header-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem 2.5rem; 
        border-radius: 1.25rem; 
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5); 
        border: 1px solid #334155;
        display: flex; 
        align-items: center; 
        gap: 1.5rem;
    }
    .header-container h1 { 
        font-weight: 800; 
        font-size: 2.2rem; 
        margin: 0; 
        color: #f8fafc;
    }
    .header-container p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
        font-weight: 500;
    }
    .logo-box {
        font-size: 3rem; 
        background: rgba(249, 115, 22, 0.1); 
        width: 4.5rem; 
        height: 4.5rem;
        display: flex; 
        align-items: center; 
        justify-content: center; 
        border-radius: 1rem;
        border: 1px solid rgba(249, 115, 22, 0.3);
    }
    
    /* Cartes de contenu */
    .dashboard-card {
        background: #1e293b;
        border-radius: 1rem; 
        padding: 1.5rem 2rem; 
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); 
        border: 1px solid #334155;
    }
    .card-title { 
        font-weight: 700; 
        font-size: 1.2rem; 
        color: #f8fafc; 
        border-bottom: 2px solid #334155; 
        padding-bottom: 0.75rem; 
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Métriques Premium */
    .metric-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: left;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #10b981;
        font-weight: 500;
    }
    
    /* Boutons Orange */
    .stButton>button {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: white; 
        font-weight: 600; 
        border-radius: 0.5rem; 
        padding: 0.75rem 1rem; 
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3);
        color: white;
    }
    
    /* Style pour les sliders et inputs en mode dark */
    .stSlider > div > div > div > div {
        background-color: #f97316 !important;
    }
    
    /* Top Bar stat highlight */
    .highlight-bar {
        background: rgba(249, 115, 22, 0.1);
        border: 1px solid rgba(249, 115, 22, 0.3);
        border-radius: 0.75rem;
        padding: 1rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    .highlight-text {
        color: #f8fafc;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .highlight-sub {
        color: #f97316;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# API URL
API_BASE_URL = "http://127.0.0.1:8000"

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/8112/8112465.png", width=60)
st.sidebar.markdown("## ⚙️ ODIN ERP")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Nos Modèles (Oussema) :",
    [
        "🗺️ Heatmap Spatiale",
        "🎯 Prédiction de Succès (XGBoost)",
        "📈 Évolution de performance",
        "⚖️ Afficher les deux (Original)"
    ]
)
st.sidebar.markdown("---")
st.sidebar.caption("Tactical Heatmap & Success Predictor v1.0")

# Header principal
st.markdown('''
<div class="header-container">
    <div class="logo-box">
        <img src="https://cdn-icons-png.flaticon.com/512/8112/8112465.png" width="45">
    </div>
    <div>
        <h1>Heatmap — Zones d'activité & Prédiction</h1>
        <p>Analyse Spatio-Temporelle des Performances et Probabilité de Succès (XGBoost)</p>
    </div>
</div>
''', unsafe_allow_html=True)

# Filtres temporels factices comme dans la maquette
col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns([1, 1, 1, 1, 4])
with col_f1:
    st.button("Saison", use_container_width=True)
with col_f2:
    st.button("Dernier Match", use_container_width=True)
with col_f3:
    st.button("5 Matchs", use_container_width=True)
with col_f4:
    st.button("10 Matchs", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Layout principal : Heatmap à gauche, Stats à droite
col_main, col_side = st.columns([2.5, 1], gap="large")

# Fonction pour dessiner le terrain
def draw_pitch():
    fig = go.Figure()
    
    # Dimensions StatsBomb: 120 x 80
    length = 120
    width = 80
    
    # Vert Terrain
    pitch_color = '#228B22'
    line_color = 'rgba(255,255,255,0.4)'

    fig.update_layout(
        xaxis=dict(range=[0, length], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, width], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor=pitch_color,
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=600
    )

    # Lignes extérieures
    fig.add_shape(type="rect", x0=0, y0=0, x1=length, y1=width, line=dict(color=line_color, width=2))
    
    # Ligne médiane
    fig.add_shape(type="line", x0=length/2, y0=0, x1=length/2, y1=width, line=dict(color=line_color, width=2))
    
    # Rond central
    fig.add_shape(type="circle", x0=length/2 - 9.15, y0=width/2 - 9.15, x1=length/2 + 9.15, y1=width/2 + 9.15, line=dict(color=line_color, width=2))
    fig.add_shape(type="circle", x0=length/2 - 0.5, y0=width/2 - 0.5, x1=length/2 + 0.5, y1=width/2 + 0.5, fillcolor=line_color, line_color=line_color)

    # Surfaces de réparation
    fig.add_shape(type="rect", x0=0, y0=width/2 - 20.15, x1=16.5, y1=width/2 + 20.15, line=dict(color=line_color, width=2))
    fig.add_shape(type="rect", x0=length - 16.5, y0=width/2 - 20.15, x1=length, y1=width/2 + 20.15, line=dict(color=line_color, width=2))

    # Surfaces de but
    fig.add_shape(type="rect", x0=0, y0=width/2 - 9.16, x1=5.5, y1=width/2 + 9.16, line=dict(color=line_color, width=2))
    fig.add_shape(type="rect", x0=length - 5.5, y0=width/2 - 9.16, x1=length, y1=width/2 + 9.16, line=dict(color=line_color, width=2))

    # Points de penalty
    fig.add_shape(type="circle", x0=11 - 0.5, y0=width/2 - 0.5, x1=11 + 0.5, y1=width/2 + 0.5, fillcolor=line_color, line_color=line_color)
    fig.add_shape(type="circle", x0=length - 11 - 0.5, y0=width/2 - 0.5, x1=length - 11 + 0.5, y1=width/2 + 0.5, fillcolor=line_color, line_color=line_color)

    # Arcs de cercle (simplifiés)
    fig.add_shape(type="path", path=f"M 16.5 {width/2 - 7} Q 23 {width/2} 16.5 {width/2 + 7}", line=dict(color=line_color, width=2))
    fig.add_shape(type="path", path=f"M {length - 16.5} {width/2 - 7} Q {length - 23} {width/2} {length - 16.5} {width/2 + 7}", line=dict(color=line_color, width=2))

    # Buts
    fig.add_shape(type="rect", x0=-2, y0=width/2 - 3.66, x1=0, y1=width/2 + 3.66, line=dict(color=line_color, width=2))
    fig.add_shape(type="rect", x0=length, y0=width/2 - 3.66, x1=length+2, y1=width/2 + 3.66, line=dict(color=line_color, width=2))
    
    # Texte d'orientation
    fig.add_annotation(x=length/4, y=width-5, text="MI-TERRAIN → SURFACE", showarrow=False, font=dict(color=line_color, size=12))

    return fig

with col_main:
    if page in ["🗺️ Heatmap Spatiale", "⚖️ Afficher les deux (Original)"]:
        st.markdown('<div class="highlight-bar"><div class="highlight-text">🔥 ZONE PRÉFÉRÉE<br><span style="font-size: 1.4rem;">Demi-espace droit / Surface adverse</span></div><div style="text-align: right;"><div class="metric-value">128 actions</div><div class="highlight-sub">+15% vs mois dernier</div></div></div>', unsafe_allow_html=True)
    
        # Récupération des données Heatmap depuis l'API
        with st.spinner("Chargement des données spatiales..."):
            try:
                res = requests.get(f"{API_BASE_URL}/player-season-heatmap")
                if res.status_code == 200:
                    data = res.json()
                    events = data["events"]
                    df_events = pd.DataFrame(events)
                
                    # Tracé du terrain
                    fig_pitch = draw_pitch()
                
                    # Ajout de la Density Map (Heatmap)
                    # Plotly Express density_contour ou histogram2dcontour
                
                    fig_heat = px.density_contour(
                        df_events, x="x", y="y", 
                        z=None, 
                        color_discrete_sequence=['#f97316']
                    )
                
                    # On utilise une heatmap basée sur les probabilités de densité
                    heatmap_trace = go.Histogram2dContour(
                        x=df_events['x'],
                        y=df_events['y'],
                        colorscale=[[0, 'rgba(0,0,0,0)'], [0.3, 'rgba(234, 179, 8, 0.4)'], [0.6, 'rgba(249, 115, 22, 0.6)'], [1, 'rgba(220, 38, 38, 0.8)']],
                        showscale=False,
                        ncontours=20,
                        line=dict(width=0),
                        contours=dict(coloring='fill')
                    )
                
                    fig_pitch.add_trace(heatmap_trace)
                
                    # Ajout des points d'action discrets
                    scatter_trace = go.Scatter(
                        x=df_events['x'], y=df_events['y'],
                        mode='markers',
                        marker=dict(size=4, color='rgba(255,255,255,0.3)', line=dict(width=0)),
                        hoverinfo='text',
                        text=[f"{row.action_type} - Succès: {'Oui' if row.success else 'Non'}" for _, row in df_events.iterrows()]
                    )
                    fig_pitch.add_trace(scatter_trace)
                
                    st.plotly_chart(fig_pitch, use_container_width=True, config={'displayModeBar': False})
                
                    st.markdown("""
                    <div style="display: flex; gap: 2rem; justify-content: center; font-size: 0.85rem; color: #94a3b8;">
                        <div><span style="color: rgba(234, 179, 8, 0.8);">●</span> 0-20 actions</div>
                        <div><span style="color: rgba(249, 115, 22, 0.8);">●</span> 20-50 actions</div>
                        <div><span style="color: rgba(220, 38, 38, 0.8);">●</span> 50-100 actions</div>
                        <div><span style="color: #991b1b;">●</span> 100+ actions</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    st.error("Erreur lors de la récupération des données de la Heatmap.")
            except Exception as e:
                st.error(f"L'API Joueur est injoignable ({e}). Vérifiez que le backend tourne sur le port 8001.")

with col_side:
    if page in ["🎯 Prédiction de Succès (XGBoost)", "⚖️ Afficher les deux (Original)"]:
        st.markdown('''
        <div class="metric-box" style="margin-bottom: 1rem;">
            <div class="metric-label">⚽ Position Principale</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Avant-centre (BU) / Ailier Droit</div>
        </div>
    
        <div class="metric-box" style="margin-bottom: 1.5rem;">
            <div class="metric-label">📍 Zone Favorite</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Demi-espace droit</div>
        </div>
        ''', unsafe_allow_html=True)
    
        st.markdown('<div class="dashboard-card" style="padding: 1.25rem;"><div class="card-title" style="font-size: 1rem;">🤖 Action Success Predictor</div>', unsafe_allow_html=True)
    
        st.caption("Simulez la probabilité de succès d'une action à un endroit donné du terrain.")
    
        col_x, col_y = st.columns(2)
        with col_x:
            x_val = st.number_input("Position X (0-120)", min_value=0.0, max_value=120.0, value=95.0, step=1.0)
        with col_y:
            y_val = st.number_input("Position Y (0-80)", min_value=0.0, max_value=80.0, value=65.0, step=1.0)
        
        action_type = st.selectbox("Type d'Action", ["Pass", "Shot", "Carry"])
        play_pattern = st.selectbox("Situation", ["Regular Play", "From Kick Off", "From Corner", "From Free Kick"])
        under_pressure = st.checkbox("Sous pression adverse", value=True)
    
        if st.button("🔮 Prédire le succès", use_container_width=True):
            payload = {
                "playerId": 10,
                "x": x_val,
                "y": y_val,
                "action_type": action_type,
                "under_pressure": under_pressure,
                "play_pattern": play_pattern
            }
        
            with st.spinner("Calcul spatial en cours..."):
                try:
                    res = requests.post(f"{API_BASE_URL}/predict-action-success", json=payload)
                    if res.status_code == 200:
                        result = res.json()
                        prob = result["success_probability"] * 100
                        dist = result["distance_to_goal"]
                        source = result["source"]
                    
                        color = "#10b981" if prob > 70 else ("#f59e0b" if prob > 40 else "#ef4444")
                    
                        st.markdown(f"""
                        <div style="margin-top: 1rem; text-align: center; background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 0.5rem; border: 1px solid #334155;">
                            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Probabilité de réussite</div>
                            <div style="font-size: 2.5rem; font-weight: 800; color: {color}; line-height: 1.2;">{prob:.1f}%</div>
                            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">Distance au but : {dist}m</div>
                            <div style="font-size: 0.7rem; color: #64748b; margin-top: 0.2rem;">(Moteur : {source})</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Erreur de l'API de prédiction.")
                except Exception as e:
                    st.error("API injoignable.")
                
        st.markdown('</div>', unsafe_allow_html=True)


if page == "📈 Évolution de performance":
    st.markdown("## 📈 Évolution de performance")
    st.caption("Prévision de l'indice de forme mensuel avec le modèle de série temporelle StatsBomb.")

    left, right = st.columns([2, 1], gap="large")
    with left:
        history_text = st.text_area(
            "Historique des scores mensuels (du plus ancien au plus récent)",
            value="82, 84, 86, 85, 88, 89",
            help="Séparez les scores par des virgules, espaces ou retours à la ligne. Minimum : 3 scores.",
        )
    with right:
        performance_player_id = st.number_input("ID joueur", min_value=1, value=10, step=1)
        forecast_steps = st.slider("Mois à prévoir", min_value=1, max_value=6, value=3)
        matches_played = st.slider("Matchs prévus / mois", min_value=0, max_value=12, value=3)

    if st.button("Prédire l'évolution", type="primary", use_container_width=True):
        try:
            raw_values = history_text.replace(";", ",").replace("\n", ",").split(",")
            history = [float(value.strip()) for value in raw_values if value.strip()]
            if len(history) < 3:
                raise ValueError("Ajoutez au moins trois scores mensuels.")
            if any(value < 0 or value > 100 for value in history):
                raise ValueError("Les scores doivent être compris entre 0 et 100.")
        except ValueError as exc:
            st.error(f"Historique invalide : {exc}")
        else:
            payload = {
                "playerId": int(performance_player_id),
                "history": history,
                "steps": forecast_steps,
                "matches_played": matches_played,
            }
            try:
                response = requests.post(
                    f"{API_BASE_URL}/predict-player-performance", json=payload, timeout=20
                )
                if response.status_code != 200:
                    detail = response.json().get("detail", response.text)
                    st.error(f"Erreur API : {detail}")
                else:
                    result = response.json()
                    predictions = result["predictions"]
                    last_score = history[-1]
                    next_score = predictions[0]
                    delta = next_score - last_score

                    metric_a, metric_b, metric_c = st.columns(3)
                    metric_a.metric("Dernier score", f"{last_score:.1f}/100")
                    metric_b.metric("Prévision du prochain mois", f"{next_score:.1f}/100", f"{delta:+.1f} pts")
                    metric_c.metric("Horizon", f"{len(predictions)} mois")

                    history_x = [f"M-{len(history) - index - 1}" if index < len(history) - 1 else "Actuel"
                                 for index in range(len(history))]
                    forecast_x = ["Actuel"] + [f"M+{index}" for index in range(1, len(predictions) + 1)]
                    figure = go.Figure()
                    figure.add_trace(go.Scatter(
                        x=history_x, y=history, mode="lines+markers", name="Historique",
                        line=dict(color="#ff6b57", width=3), marker=dict(size=9),
                    ))
                    figure.add_trace(go.Scatter(
                        x=forecast_x, y=[last_score] + predictions, mode="lines+markers", name="Prévision",
                        line=dict(color="#38bdf8", width=3, dash="dash"), marker=dict(size=9),
                    ))
                    figure.update_layout(
                        height=420, yaxis=dict(range=[60, 100], title="Score de performance"),
                        xaxis_title="Période", plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                        font=dict(color="#f8fafc"), legend=dict(orientation="h", y=1.12),
                        margin=dict(l=20, r=20, t=55, b=20),
                    )
                    st.plotly_chart(figure, use_container_width=True)
                    st.caption(f"Source : {result['source']} · R² enregistré : {result['model_metrics'].get('r2', 'n/a')}")
            except requests.RequestException as exc:
                st.error(f"API Joueur injoignable : {exc}")

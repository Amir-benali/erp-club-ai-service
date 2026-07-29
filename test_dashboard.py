import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURATION INITIALE
# ==========================================
st.set_page_config(
    page_title="ERP Club AI - Hub Analytique",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. DESIGN SYSTEM (CSS GLOBAL)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }
    
    .stApp { 
        background: #f8fafc; 
    }
    
    /* Header Premium */
    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.5rem 2.5rem; 
        border-radius: 1.25rem; 
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.1); 
        color: white;
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
        background: rgba(255,255,255,0.1); 
        width: 4.5rem; 
        height: 4.5rem;
        display: flex; 
        align-items: center; 
        justify-content: center; 
        border-radius: 1rem;
        box-shadow: inset 0 2px 4px rgba(255,255,255,0.1);
    }
    
    /* Cartes de contenu */
    .dashboard-card {
        background: #ffffff;
        border-radius: 1rem; 
        padding: 1.5rem 2rem; 
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03); 
        border: 1px solid #e2e8f0;
    }
    .card-title { 
        font-weight: 700; 
        font-size: 1.2rem; 
        color: #0f172a; 
        border-bottom: 2px solid #f1f5f9; 
        padding-bottom: 0.75rem; 
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Viiv GX17 Live Card */
    .viiv-card {
        background: linear-gradient(135deg, #0f172a 0%, #1a1035 100%);
        border-radius: 1.25rem;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        border: 1px solid rgba(139, 92, 246, 0.3);
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.15), 0 4px 12px rgba(0,0,0,0.3);
        color: white;
    }
    .viiv-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.1rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        padding-bottom: 0.75rem;
    }
    .viiv-live-badge {
        background: #ef4444;
        color: white;
        font-size: 0.65rem;
        font-weight: 800;
        padding: 2px 8px;
        border-radius: 999px;
        letter-spacing: 0.05em;
        animation: pulse 1.5s infinite;
        text-transform: uppercase;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .viiv-brand {
        font-weight: 800;
        font-size: 1.05rem;
        color: #c4b5fd;
        letter-spacing: 0.03em;
    }
    .viiv-metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
    }
    .viiv-metric {
        background: rgba(255,255,255,0.06);
        border-radius: 0.75rem;
        padding: 0.65rem 0.75rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.07);
    }
    .viiv-metric-label {
        font-size: 0.6rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    .viiv-metric-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1;
    }
    .viiv-metric-unit {
        font-size: 0.6rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 2px;
    }
    .viiv-metric.fc .viiv-metric-value { color: #f87171; }
    .viiv-metric.spo2 .viiv-metric-value { color: #60a5fa; }
    .viiv-metric.hrv .viiv-metric-value { color: #34d399; }
    .viiv-metric.stress .viiv-metric-value { color: #fb923c; }
    .viiv-metric.energy .viiv-metric-value { color: #facc15; }
    .viiv-metric.sleep .viiv-metric-value { color: #a78bfa; }
    .viiv-metric.recovery .viiv-metric-value { color: #2dd4bf; }
    .viiv-metric.strain .viiv-metric-value { color: #f472b6; }
    
    .viiv-derived-tag {
        display: inline-block;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.3);
        color: #c4b5fd;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 999px;
        margin-bottom: 0.5rem;
    }

    /* Alertes et Métriques */
    .danger-alert {
        background: #fef2f2; 
        border-left: 6px solid #ef4444; 
        border-radius: 0.5rem;
        padding: 1.25rem; 
        color: #991b1b; 
        margin: 1.5rem 0;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.05);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
    }
    
    /* Heatmap Anatomique */
    .body-part {
        border: 1px solid #cbd5e1; 
        display: flex; 
        justify-content: center; 
        align-items: center;
        font-size: 0.65rem; 
        font-weight: 700; 
        color: #334155; 
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: white;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Boutons */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white; 
        font-weight: 600; 
        border-radius: 0.5rem; 
        padding: 0.75rem 1rem; 
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. ÉTAT SESSION — Données Viiv GX17
# ==========================================
if "viiv_data" not in st.session_state:
    st.session_state.viiv_data = None  # Aucune donnée Viiv encore chargée


# ==========================================
# 4. NAVIGATION ET ROUTES API
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/8112/8112465.png", width=60)
st.sidebar.markdown("## 🧭 Navigation IA")
page = st.sidebar.radio(
    "Sélectionnez un microservice :",
    [
        "🩺 M1 - Risque de Blessure (Global)", 
        "🗺️ M2 - Cartographie (Zones)", 
        "⏳ M3 - Analyse de Survie (Rechute)"
    ]
)
st.sidebar.markdown("---")

# ------------------------------------------
# PANNEAU VIIV GX17 — Sidebar (global)
# ------------------------------------------
st.sidebar.markdown("### 📡 Viiv GX17 — Données Capteur")
st.sidebar.caption("Saisissez les données reçues par l'app mobile depuis le capteur Viiv GX17 via Bluetooth.")

with st.sidebar.expander("🔴 Saisir / Mettre à jour les données Viiv", expanded=st.session_state.viiv_data is None):
    v_heart_rate = st.number_input("❤️ FC — Fréquence Cardiaque (bpm)", min_value=30.0, max_value=220.0, value=97.0, step=1.0)
    v_spo2 = st.number_input("🫁 SpO₂ — Saturation O₂ (%)", min_value=70.0, max_value=100.0, value=98.0, step=0.1)
    v_hrv = st.number_input("💚 HRV (ms)", min_value=0.0, max_value=200.0, value=42.0, step=0.5)
    v_stress = st.number_input("🧘 Stress (0–100)", min_value=0.0, max_value=100.0, value=35.0, step=1.0)
    v_energy = st.number_input("⚡ Énergie (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
    v_sleep = st.number_input("💤 Sommeil (score Viiv)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    v_recovery = st.number_input("🔋 Recovery (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    v_strain = st.number_input("🔥 Strain (0–21)", min_value=0.0, max_value=21.0, value=0.0, step=0.1)

    if st.button("✅ Valider les données Viiv GX17", use_container_width=True):
        st.session_state.viiv_data = {
            "heart_rate": v_heart_rate,
            "spo2": v_spo2,
            "hrv_ms": v_hrv,
            "stress_score": v_stress,
            "energy_pct": v_energy,
            "sleep_score": v_sleep,
            "recovery_pct": v_recovery,
            "strain": v_strain,
        }
        st.success("✅ Données Viiv chargées ! Les modules IA sont maintenant pré-remplis.")

if st.session_state.viiv_data:
    if st.sidebar.button("🗑️ Réinitialiser les données Viiv", use_container_width=True):
        st.session_state.viiv_data = None
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("ERP Club AI v5.0 — Intégration Viiv GX17")

# Endpoints de l'API FastAPI
API_URL_GLOBAL = "http://localhost:8000/predict-injury"
API_URL_ZONE = "http://localhost:8000/predict-injury-zone"
API_URL_RELAPSE = "http://localhost:8000/predict-relapse"


# ==========================================
# HELPER — Calcul des champs dérivés depuis Viiv
# ==========================================
def viiv_derive_sommeil(v):
    if v is None or v.get("sleep_score") is None: return None
    return round(max(1.0, min(10.0, float(v["sleep_score"]))), 2)

def viiv_derive_stress(v):
    if v is None or v.get("stress_score") is None: return None
    return round(max(1.0, min(10.0, v["stress_score"] / 10.0)), 2)

def viiv_derive_fatigue(v):
    if v is None or v.get("hrv_ms") is None: return None
    hrv = max(10.0, min(100.0, v["hrv_ms"]))
    return round(max(1.0, min(10.0, 10.0 - ((hrv - 10.0) / 90.0) * 9.0)), 2)

def viiv_derive_acute_load(v, base_load=5950.0):
    if v is None or v.get("strain") is None:
        return base_load
    return round(base_load + (float(v["strain"]) * 200.0), 0)

def viiv_derive_recovery(v):
    if v is None: return None
    return v.get("recovery_pct")

def viiv_derive_fatigue_index(v):
    fat = viiv_derive_fatigue(v)
    return round(fat * 10.0, 2) if fat else None

def viiv_derive_stress_level(v):
    if v is None or v.get("stress_score") is None: return None
    return round(max(0.0, min(1.0, v["stress_score"] / 100.0)), 3)


# ==========================================
# HELPER — Carte Live Viiv GX17
# ==========================================
def render_viiv_card(v):
    if not v:
        return
    st.markdown(f"""
    <div class="viiv-card">
        <div class="viiv-header">
            <span class="viiv-brand">Viiv GX17</span>
            <span class="viiv-live-badge">● Live</span>
            <span style="color:#64748b; font-size:0.75rem; margin-left: auto;">Données capteur Bluetooth</span>
        </div>
        <div class="viiv-metric-grid">
            <div class="viiv-metric fc">
                <div class="viiv-metric-label">FC</div>
                <div class="viiv-metric-value">{int(v.get('heart_rate', 0))}</div>
                <div class="viiv-metric-unit">bpm</div>
            </div>
            <div class="viiv-metric spo2">
                <div class="viiv-metric-label">SpO₂</div>
                <div class="viiv-metric-value">{v.get('spo2', 0):.1f}</div>
                <div class="viiv-metric-unit">%</div>
            </div>
            <div class="viiv-metric hrv">
                <div class="viiv-metric-label">HRV</div>
                <div class="viiv-metric-value">{v.get('hrv_ms', 0):.0f}</div>
                <div class="viiv-metric-unit">ms</div>
            </div>
            <div class="viiv-metric stress">
                <div class="viiv-metric-label">Stress</div>
                <div class="viiv-metric-value">{v.get('stress_score', 0):.0f}</div>
                <div class="viiv-metric-unit">/ 100</div>
            </div>
            <div class="viiv-metric energy">
                <div class="viiv-metric-label">Énergie</div>
                <div class="viiv-metric-value">{v.get('energy_pct', 0):.0f}</div>
                <div class="viiv-metric-unit">%</div>
            </div>
            <div class="viiv-metric sleep">
                <div class="viiv-metric-label">Sommeil</div>
                <div class="viiv-metric-value">{v.get('sleep_score', 0):.1f}</div>
                <div class="viiv-metric-unit">/ 10</div>
            </div>
            <div class="viiv-metric recovery">
                <div class="viiv-metric-label">Recovery</div>
                <div class="viiv-metric-value">{v.get('recovery_pct', 0):.0f}</div>
                <div class="viiv-metric-unit">%</div>
            </div>
            <div class="viiv-metric strain">
                <div class="viiv-metric-label">Strain</div>
                <div class="viiv-metric-value">{v.get('strain', 0):.1f}</div>
                <div class="viiv-metric-unit">/ 21</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 1 : RISQUE GLOBAL (Modèle 1)
# ==========================================
if page == "🩺 M1 - Risque de Blessure (Global)":
    st.markdown('''
    <div class="header-container">
        <div class="logo-box">🩺</div>
        <div>
            <h1>Prédiction Globale de Blessure</h1>
            <p>Classification Binaire XGBoost : Analyse des charges et facteurs de fatigue — Viiv GX17 intégré</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Carte Viiv Live en haut de page
    v = st.session_state.viiv_data
    render_viiv_card(v)
    if not v:
        st.warning("💡 Chargez les données Viiv GX17 dans le panneau de gauche pour activer l'analyse.")

    default_acute_load = viiv_derive_acute_load(v)

    with st.sidebar: 
        st.markdown("### ⚙️ Configuration")
        player_id = st.number_input("ID Joueur", value=10, step=1)

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown('<div class="dashboard-card"><div class="card-title">📊 Paramètres Cliniques & GPS</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if v:
                st.markdown('<span class="viiv-derived-tag">🔴 Viiv GX17</span>', unsafe_allow_html=True)
            acuteLoad = st.number_input("Acute Load (7j)", value=float(default_acute_load), step=100.0)
        with c2:
            douleur = st.slider("🦵 Douleurs Musculaires", 1.0, 10.0, 3.0)
            if v:
                st.markdown('<span class="viiv-derived-tag">🔴 Viiv GX17</span>', unsafe_allow_html=True)
            chronicLoad = st.number_input("Chronic Load (28j)", value=5100, step=100)
            
        totalLoad = st.number_input("Charge de travail prévue (Aujourd'hui)", value=850)
        c3, c4 = st.columns(2)
        with c3:
            sommeil_7d_mean = st.number_input("Sommeil moyen 7j", value=7.0, step=0.1)
            douleur_musculaire_7d_mean = st.number_input("Douleurs musculaires moyennes 7j", value=3.0, step=0.1)
        with c4:
            fatigue_7d_mean = st.number_input("Fatigue moyenne 7j", value=4.0, step=0.1)
            stress_7d_mean = st.number_input("Stress moyen 7j", value=4.0, step=0.1)
        acwr = float(acuteLoad / chronicLoad) if chronicLoad > 0 else 0
        
        acwr_color = "normal"
        if acwr > 1.5 or acwr < 0.8: acwr_color = "inverse"
        st.metric("Ratio ACWR (Acute:Chronic)", f"{acwr:.2f}", delta="Danger" if acwr>1.5 else "Optimal", delta_color=acwr_color)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dashboard-card"><div class="card-title">🤖 Diagnostic de l\'IA</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Lancer l'Analyse Prédictive", use_container_width=True, disabled=not v):
            payload = {
                "playerId": player_id,
                "totalLoad": totalLoad,
                "douleurMusculaire": douleur,
                "acuteLoad": acuteLoad,
                "chronicLoad": chronicLoad,
                "ACWR": acwr,
                "sommeil_7d_mean": sommeil_7d_mean,
                "fatigue_7d_mean": fatigue_7d_mean,
                "douleurMusculaire_7d_mean": douleur_musculaire_7d_mean,
                "stress_7d_mean": stress_7d_mean,
                "viiv": v,
            }

            with st.spinner("Analyse des arbres de décision en cours..."):
                try:
                    res = requests.post(API_URL_GLOBAL, json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        risk_prob = data['riskScore'] * 100
                        color = "#ef4444" if risk_prob > 60 else ("#f59e0b" if risk_prob > 30 else "#22c55e")
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem;">
                            <p style="margin:0; color: #64748b; font-weight: 600; text-transform: uppercase;">Probabilité de Blessure Imminente</p>
                            <h1 style="font-size: 4rem; color: {color}; margin: 0; font-weight: 800;">{risk_prob:.0f}%</h1>
                            <p style="margin:0; font-weight: bold; color: {color};">Niveau : {data['riskLevel']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.progress(data['riskScore'])

                        # Afficher les inputs résolus (Viiv → IA)
                        if data.get('resolved_inputs') and v:
                            with st.expander("🔍 Champs IA résolus depuis Viiv GX17"):
                                ri = data['resolved_inputs']
                                cols = st.columns(4)
                                cols[0].metric("Sommeil", f"{ri.get('sommeil', '-'):.2f}")
                                cols[1].metric("Fatigue", f"{ri.get('fatigue', '-'):.2f}")
                                cols[2].metric("Stress", f"{ri.get('stress', '-'):.2f}")
                                cols[3].metric("ACWR", f"{ri.get('ACWR', '-'):.2f}")
                        
                        if data.get('factors'):
                            st.markdown("#### 🔍 Principaux Facteurs (Explicabilité)")
                            df_factors = pd.DataFrame(data['factors']).sort_values(by="contribution", ascending=True)
                            fig_factors = px.bar(df_factors, x="contribution", y="feature", orientation='h', color="impact", color_discrete_map={"négatif": "#ef4444", "positif": "#22c55e"})
                            fig_factors.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250)
                            st.plotly_chart(fig_factors, use_container_width=True)
                    else: 
                        st.error(f"Erreur API ({res.status_code}): {res.text}")
                except Exception as e: 
                    st.error(f"Impossible de contacter l'API. Vérifiez que FastAPI tourne sur le port 8000.\nErreur: {e}")
        else:
            st.info("👈 Ajustez les paramètres et lancez l'analyse pour évaluer le risque du joueur.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 2 : CARTOGRAPHIE DES ZONES (Modèle 2)
# ==========================================
elif page == "🗺️ M2 - Cartographie (Zones)":
    st.markdown('''
    <div class="header-container">
        <div class="logo-box">🗺️</div>
        <div>
            <h1>Cartographie Anatomique</h1>
            <p>Modèle Multi-classe Random Forest : Prédiction des zones de vulnérabilité corporelle</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    v = st.session_state.viiv_data
    render_viiv_card(v)

    with st.sidebar:
        st.markdown("### 📋 Morphologie & Poste")
        player_id = st.number_input("ID Joueur", value=10, step=1)
        position = st.selectbox("Position", ["Attaquant", "Milieu", "Défenseur", "Gardien"])
        foot = st.selectbox("Pied Fort", ["Droitier", "Gaucher", "Ambidextre"])
        age = st.slider("Âge", 16, 40, 24)
        fifa = st.slider("Note Générale (Fifa Rating)", 50, 95, 75)

    col1, col2 = st.columns([1, 1.3], gap="large")
    
    with col1:
        st.markdown('<div class="dashboard-card"><div class="card-title">🏃‍♂️ Biomecanique & Charge</div>', unsafe_allow_html=True)
        acute_zone = viiv_derive_acute_load(v, 6000)
        chronic_zone = 4500
        acwr_zone = float(acute_zone / chronic_zone) if chronic_zone > 0 else 0
        st.metric("ACWR", f"{acwr_zone:.2f}")
        
        st.markdown("---")
        # douleurMusculaire enrichie par Viiv HRV si disponible
        default_douleur_z = 4.0
        if v:
            fat = viiv_derive_fatigue(v)
            if fat:
                default_douleur_z = round((4.0 + fat) / 2.0, 1)
            st.markdown('<span class="viiv-derived-tag">🔴 Viiv GX17 (HRV enrichit douleur)</span>', unsafe_allow_html=True)

        douleur_z = st.slider("Douleurs Actuelles (1-10)", 1.0, 10.0, default_douleur_z, key="z_doul")
        souplesse_z = st.slider("Souplesse Globale (1-10)", 1.0, 10.0, 6.0, key="z_soup")
        agilite_z = st.slider("Test d'Agilité (1-10)", 1.0, 10.0, 8.0, key="z_agil")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dashboard-card"><div class="card-title">🧬 Analyse des Points de Rupture</div>', unsafe_allow_html=True)
        if st.button("🔥 Générer la Cartographie Corporelle", use_container_width=True, disabled=not v):
            payload_zone = {
                "playerId": player_id, "position": position, "foot": foot, "age": age, 
                "fifa_rating": fifa, "acuteLoad": acute_zone, "chronicLoad": chronic_zone, 
                "ACWR": acwr_zone, "douleurMusculaire": douleur_z, "souplesse": souplesse_z, 
                "agilite": agilite_z
            }
            payload_zone["viiv"] = v

            with st.spinner("Cartographie des probabilités par zone..."):
                try:
                    res_zone = requests.post(API_URL_ZONE, json=payload_zone)
                    if res_zone.status_code == 200:
                        predictions = res_zone.json()["predictions"]
                        
                        df_res = pd.DataFrame(list(predictions.items()), columns=["Zone", "Proba"])
                        df_res["Probabilité"] = df_res["Proba"] * 100
                        df_res = df_res.sort_values(by='Probabilité', ascending=False)
                        top_zone = df_res.iloc[0]
                        
                        st.markdown(f'<div class="danger-alert">🚨 ZONE CRITIQUE DÉTECTÉE : {top_zone["Zone"].upper()} ({top_zone["Probabilité"]:.1f}%)</div>', unsafe_allow_html=True)
                        
                        def get_heat_color(zone_name):
                            val = predictions.get(zone_name, 0)
                            intensity = min(1.0, val * 3) 
                            return f"rgba({int(148 + (239-148)*intensity)}, {int(163 - 95*intensity)}, {int(184 - 116*intensity)}, 1)"
                        
                        tab1, tab2 = st.tabs(["🧍‍♂️ Vue Anatomique", "📊 Graphique Radar"])
                        
                        with tab1:
                            heatmap_html = f"""
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 6px; margin: 20px 0;">
                                <div class="body-part" style="width: 55px; height: 55px; border-radius: 50%; background: {get_heat_color('TETE')};">TÊTE</div>
                                <div style="display: flex; gap: 6px; align-items: flex-start;">
                                    <div class="body-part" style="width: 45px; height: 35px; border-radius: 12px 0 0 12px; background: {get_heat_color('EPAULE')};">ÉPAULE</div>
                                    <div class="body-part" style="width: 90px; height: 100px; border-radius: 8px; background: {get_heat_color('DOS')};">TORSE / DOS</div>
                                    <div class="body-part" style="width: 45px; height: 35px; border-radius: 0 12px 12px 0; background: {get_heat_color('EPAULE')};">ÉPAULE</div>
                                </div>
                                <div style="display: flex; gap: 6px; margin-top: -65px;">
                                    <div class="body-part" style="width: 35px; height: 90px; border-radius: 15px; margin-right: 52px; background: {get_heat_color('BRAS')}; writing-mode: vertical-rl;">BRAS</div>
                                    <div class="body-part" style="width: 80px; height: 45px; border-radius: 8px; margin-top: 65px; background: {get_heat_color('HANCHE')};">HANCHE / AINE</div>
                                    <div class="body-part" style="width: 35px; height: 90px; border-radius: 15px; margin-left: 52px; background: {get_heat_color('BRAS')}; writing-mode: vertical-rl;">BRAS</div>
                                </div>
                                <div style="display: flex; gap: 6px; align-items: flex-start;">
                                    <div class="body-part" style="width: 25px; height: 35px; border-radius: 50%; margin-right: 12px; background: {get_heat_color('MAIN')};">MAIN</div>
                                    <div class="body-part" style="width: 37px; height: 80px; border-radius: 8px; background: {get_heat_color('CUISSE')};">CUISSE</div>
                                    <div class="body-part" style="width: 37px; height: 80px; border-radius: 8px; background: {get_heat_color('CUISSE')};">CUISSE</div>
                                    <div class="body-part" style="width: 25px; height: 35px; border-radius: 50%; margin-left: 12px; background: {get_heat_color('MAIN')};">MAIN</div>
                                </div>
                                <div style="display: flex; gap: 6px;">
                                    <div class="body-part" style="width: 35px; height: 35px; border-radius: 50%; background: {get_heat_color('GENOU')};">GENOU</div>
                                    <div class="body-part" style="width: 35px; height: 35px; border-radius: 50%; background: {get_heat_color('GENOU')};">GENOU</div>
                                </div>
                                <div style="display: flex; gap: 6px;">
                                    <div class="body-part" style="width: 30px; height: 70px; border-radius: 8px; background: {get_heat_color('JAMBE')};">MOLLET</div>
                                    <div class="body-part" style="width: 30px; height: 70px; border-radius: 8px; background: {get_heat_color('JAMBE')};">MOLLET</div>
                                </div>
                                <div style="display: flex; gap: 6px;">
                                    <div class="body-part" style="width: 28px; height: 25px; border-radius: 8px; background: {get_heat_color('CHEVILLE')};">CHEV.</div>
                                    <div class="body-part" style="width: 28px; height: 25px; border-radius: 8px; background: {get_heat_color('CHEVILLE')};">CHEV.</div>
                                </div>
                                <div style="display: flex; gap: 6px;">
                                    <div class="body-part" style="width: 40px; height: 20px; border-radius: 10px 10px 0 0; background: {get_heat_color('PIED')};">PIED</div>
                                    <div class="body-part" style="width: 40px; height: 20px; border-radius: 10px 10px 0 0; background: {get_heat_color('PIED')};">PIED</div>
                                </div>
                            </div>
                            """
                            st.markdown(heatmap_html.replace('\n', ''), unsafe_allow_html=True)
                            
                        with tab2:
                            fig_radar = px.line_polar(
                                df_res.head(8), r='Probabilité', theta='Zone', 
                                line_close=True, color_discrete_sequence=['#ef4444']
                            )
                            fig_radar.update_traces(fill='toself', fillcolor='rgba(239, 68, 68, 0.3)', line=dict(width=3))
                            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, df_res['Probabilité'].max() + 5])), margin=dict(l=40, r=40, t=20, b=20))
                            st.plotly_chart(fig_radar, use_container_width=True)
                            
                    else: st.error("Erreur avec le modèle de zone.")
                except Exception as e: st.error(f"API injoignable : {e}")
        else:
            st.info("Sélectionnez la biométrie du joueur et lancez le modèle pour cartographier les fragilités corporelles.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 3 : ANALYSE DE SURVIE (Modèle 3)
# ==========================================
elif page == "⏳ M3 - Analyse de Survie (Rechute)":
    st.markdown('''
    <div class="header-container" style="background: linear-gradient(135deg, #064e3b 0%, #0f766e 100%);">
        <div class="logo-box">⏳</div>
        <div>
            <h1>Analyse de Survie Post-Rééducation</h1>
            <p>Régression de Cox (Cox PH) : Estimation temporelle du risque de rechute après retour au jeu.</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    v = st.session_state.viiv_data
    render_viiv_card(v)
    if not v:
        st.warning("💡 Chargez les données Viiv GX17 dans le panneau de gauche pour activer l'analyse.")

    # Valeurs dérivées depuis Viiv pour ce module
    default_recovery = viiv_derive_recovery(v) or 75.0

    with st.sidebar:
        st.markdown("### 🏥 Bilan de Rééducation")
        player_id = st.number_input("ID Joueur", value=10, step=1)
        physio_adherence = st.slider("Adhérence au protocole Physio (%)", 0, 100, 85, help="Taux d'assiduité aux séances de kinésithérapie.")
        post_acwr = st.slider("ACWR projeté (Retour au jeu)", 0.5, 2.5, 1.1, step=0.1, help="Charge de travail accumulée prévue pour sa reprise.")

    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown('<div class="dashboard-card"><div class="card-title">🩺 État de Santé (Wellness)</div>', unsafe_allow_html=True)

        if v:
            st.markdown('<span class="viiv-derived-tag">🔴 Viiv GX17 — Recovery %</span>', unsafe_allow_html=True)
        st.metric("Score de Récupération (Viiv)", f"{default_recovery:.0f}")
        st.metric("Sommeil (Viiv)", f"{viiv_derive_sommeil(v) or 7.5:.1f}")
        st.metric("Stress (Viiv)", f"{(viiv_derive_stress_level(v) or 0.3):.3f}")
        st.metric("Index de Fatigue (Viiv)", f"{viiv_derive_fatigue_index(v) or 45.0:.0f}")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dashboard-card"><div class="card-title">📈 Projection de Survie au fil du temps</div>', unsafe_allow_html=True)
        
        if st.button("🔮 Calculer la Courbe de Survie (Cox PH)", use_container_width=True, disabled=not v):
            payload_surv = {
                "playerId": player_id,
                "physio_adherence": physio_adherence,
                "post_recovery_ACWR": post_acwr,
                "viiv": v,
            }
            
            with st.spinner("Modélisation de la survie temporelle (Kaplan-Meier estimé)..."):
                try:
                    res_surv = requests.post(API_URL_RELAPSE, json=payload_surv)
                    
                    if res_surv.status_code == 200:
                        data_surv = res_surv.json()
                        curve_data = data_surv.get("survival_curve", [])
                        
                        if curve_data:
                            df_curve = pd.DataFrame(curve_data)
                            
                            fig = px.line(
                                df_curve, x="day", y="probability",
                                labels={"day": "Jours de suivi après retour", "probability": "Probabilité de ne pas rechuter"},
                                color_discrete_sequence=['#10b981']
                            )
                            
                            fig.update_layout(
                                yaxis_range=[0, 1.05], 
                                margin=dict(l=20, r=20, t=40, b=20),
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            fig.add_hline(
                                y=0.5, line_dash="dot", 
                                annotation_text="Seuil de Danger Critique (50%)", 
                                annotation_position="bottom right", line_color="#ef4444"
                            )
                            
                            fig.update_traces(fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.15)', line=dict(width=3))
                            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
                            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            c_index = data_surv.get("c_index", 0.96)
                            st.success(f"**Concordance Index (Précision globale du modèle) : {c_index:.3f}**")

                            # Afficher les inputs résolus depuis Viiv
                            if data_surv.get('resolved_inputs') and v:
                                with st.expander("🔍 Champs IA résolus depuis Viiv GX17"):
                                    ri = data_surv['resolved_inputs']
                                    cols = st.columns(4)
                                    cols[0].metric("Recovery", f"{ri.get('recovery_score', '-'):.1f}")
                                    cols[1].metric("Sommeil", f"{ri.get('sleep_quality', '-'):.2f}")
                                    cols[2].metric("Stress", f"{ri.get('stress_level', '-'):.3f}")
                                    cols[3].metric("Fatigue", f"{ri.get('fatigue_index', '-'):.1f}")
                            
                            jours_30 = df_curve[df_curve['day'] >= 30]
                            prob_a_30_jours = jours_30.iloc[0]['probability'] if not jours_30.empty else df_curve.iloc[-1]['probability']
                            
                            if prob_a_30_jours < 0.6:
                                st.error(f"⚠️ **Alerte Médicale :** Le joueur a seulement **{prob_a_30_jours*100:.1f}%** de chance de ne pas rechuter le premier mois. Protocole de reprise inadapté ou trop agressif.")
                            else:
                                st.info(f"✅ **Bilan positif :** Le joueur a **{prob_a_30_jours*100:.1f}%** de chance de rester en bonne santé après 30 jours.")
                        else:
                            st.warning("L'API n'a retourné aucune donnée pour la courbe.")
                    else:
                        st.error(f"Erreur avec le modèle de survie ({res_surv.status_code}) : {res_surv.text}")
                except Exception as e:
                    st.error(f"API injoignable. Assurez-vous que FastAPI tourne sur le port 8000.\nDétails : {e}")
        else:
            st.info("👈 Ajustez les paramètres post-rééducation du patient (Adhérence et ACWR sont primordiaux) et lancez le modèle pour simuler son risque temporel de rechute.")
            
        st.markdown('</div>', unsafe_allow_html=True)
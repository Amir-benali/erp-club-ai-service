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
if "medical_nutrition" not in st.session_state:
    st.session_state.medical_nutrition = None

NUTRITION_DEFAULTS = {
    "vitamin_d": 24.0, "vitamin_b12": 512.0, "folate": 8.0,
    "vitamin_c": 0.0, "vitamin_a": 0.0, "vitamin_e": 0.0,
    "zinc": 82.0, "magnesium": 1.8, "iron": 0.0, "ferritin": 85.0,
    "calcium": 9.1, "hemoglobin": 14.2, "c_reactive_protein": 0.32,
    "fasting_glucose": 0.96, "total_cholesterol": 2.15,
    "ldl_cholesterol": 1.45, "hdl_cholesterol": 0.48, "triglycerides": 1.30,
}
GLOBAL_NUTRITION_FIELDS = {
    "vitamin_d", "ferritin", "hemoglobin", "vitamin_b12", "magnesium",
    "zinc", "iron", "c_reactive_protein",
}
ZONE_NUTRITION_FIELDS = {
    "vitamin_d", "ferritin", "hemoglobin", "magnesium", "calcium",
    "c_reactive_protein",
}
RELAPSE_NUTRITION_FIELDS = {
    "vitamin_d", "ferritin", "hemoglobin", "vitamin_b12", "magnesium",
    "iron", "c_reactive_protein",
}

def nutrition_for_model(nutrition, allowed_fields):
    """Send only OCR values relevant to the target model."""
    if not nutrition:
        return None
    selected = {key: value for key, value in nutrition.items() if key in allowed_fields}
    return selected or None


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
        "⏳ M3 - Analyse de Survie (Rechute)",
        "🧾 M4 - OCR Rapport Medical",
        "⚽ M5 - Possession Estimation"
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
st.sidebar.markdown("### 🧪 Nutrition / Bilan OCR")
st.sidebar.caption("Valeurs standardisées extraites d'un rapport médical français. Elles ajustent la préparation dans les 3 modèles.")

with st.sidebar.expander("🧪 Saisir / mettre à jour le bilan OCR", expanded=st.session_state.medical_nutrition is None):
    nutrition_enabled = st.checkbox(
        "Utiliser un bilan nutritionnel", value=st.session_state.medical_nutrition is not None,
        key="nutrition_enabled",
    )
    current_nutrition = st.session_state.medical_nutrition or NUTRITION_DEFAULTS
    if nutrition_enabled:
        n_col1, n_col2 = st.columns(2)
        with n_col1:
            n_vitd = st.number_input("Vitamine D (ng/mL)", value=float(current_nutrition.get("vitamin_d", 0.0)), key="n_vitd")
            n_b12 = st.number_input("Vitamine B12 (pg/mL)", value=float(current_nutrition.get("vitamin_b12", 0.0)), key="n_b12")
            n_folate = st.number_input("Folates B9 (ng/mL)", value=float(current_nutrition.get("folate", 0.0)), key="n_folate")
            n_vitc = st.number_input("Vitamine C (mg/L)", value=float(current_nutrition.get("vitamin_c", 0.0)), key="n_vitc")
            n_vita = st.number_input("Vitamine A", value=float(current_nutrition.get("vitamin_a", 0.0)), key="n_vita")
            n_vite = st.number_input("Vitamine E", value=float(current_nutrition.get("vitamin_e", 0.0)), key="n_vite")
            n_zinc = st.number_input("Zinc (µg/dL)", value=float(current_nutrition.get("zinc", 0.0)), key="n_zinc")
            n_magnesium = st.number_input("Magnésium (mg/dL)", value=float(current_nutrition.get("magnesium", 0.0)), key="n_magnesium")
            n_iron = st.number_input("Fer (µg/dL)", value=float(current_nutrition.get("iron", 0.0)), key="n_iron")
        with n_col2:
            n_ferritin = st.number_input("Ferritine (ng/mL)", value=float(current_nutrition.get("ferritin", 0.0)), key="n_ferritin")
            n_calcium = st.number_input("Calcium (mg/dL)", value=float(current_nutrition.get("calcium", 0.0)), key="n_calcium")
            n_hemoglobin = st.number_input("Hémoglobine (g/dL)", value=float(current_nutrition.get("hemoglobin", 0.0)), key="n_hemoglobin")
            n_crp = st.number_input("CRP (mg/L)", value=float(current_nutrition.get("c_reactive_protein", 0.0)), key="n_crp")
            n_glucose = st.number_input("Glycémie à jeun (g/L)", value=float(current_nutrition.get("fasting_glucose", 0.0)), key="n_glucose")
            n_total_chol = st.number_input("Cholestérol total (g/L)", value=float(current_nutrition.get("total_cholesterol", 0.0)), key="n_total_chol")
            n_ldl = st.number_input("LDL (g/L)", value=float(current_nutrition.get("ldl_cholesterol", 0.0)), key="n_ldl")
            n_hdl = st.number_input("HDL (g/L)", value=float(current_nutrition.get("hdl_cholesterol", 0.0)), key="n_hdl")
            n_triglycerides = st.number_input("Triglycérides (g/L)", value=float(current_nutrition.get("triglycerides", 0.0)), key="n_triglycerides")
        if st.button("✅ Utiliser ce bilan dans les modèles", use_container_width=True):
            nutrition_payload = {
                "vitamin_d": n_vitd, "vitamin_b12": n_b12, "folate": n_folate,
                "vitamin_c": n_vitc, "vitamin_a": n_vita, "vitamin_e": n_vite,
                "zinc": n_zinc, "magnesium": n_magnesium, "iron": n_iron,
                "ferritin": n_ferritin, "calcium": n_calcium, "hemoglobin": n_hemoglobin,
                "c_reactive_protein": n_crp, "fasting_glucose": n_glucose,
                "total_cholesterol": n_total_chol, "ldl_cholesterol": n_ldl,
                "hdl_cholesterol": n_hdl, "triglycerides": n_triglycerides,
            }
            # A zero in the form means "not available in this report", not a
            # clinical zero. Omit it so FastAPI receives only known OCR values.
            st.session_state.medical_nutrition = {
                key: value for key, value in nutrition_payload.items() if value > 0
            }
            st.success("Bilan OCR chargé pour les prédictions.")

if st.session_state.medical_nutrition and st.sidebar.button("🗑️ Réinitialiser le bilan OCR", use_container_width=True):
    st.session_state.medical_nutrition = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("ERP Club AI v5.0 — Intégration Viiv GX17")

# Endpoints de l'API FastAPI
API_URL_GLOBAL = "http://localhost:8000/predict-injury"
API_URL_ZONE = "http://localhost:8000/predict-injury-zone"
API_URL_RELAPSE = "http://localhost:8000/predict-relapse"
API_URL_MEDICAL_OCR = "http://localhost:8000/extract-medical-nutrients"


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
    nutrition = st.session_state.medical_nutrition
    render_viiv_card(v)
    if not v:
        st.warning("💡 Chargez les données Viiv GX17 dans le panneau de gauche pour activer l'analyse.")
    if nutrition:
        st.info("🧪 Bilan nutritionnel OCR actif : les valeurs biologiques ajusteront fatigue, douleur, stress et sommeil.")

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
            if nutrition:
                payload["medical_nutrition"] = nutrition_for_model(nutrition, GLOBAL_NUTRITION_FIELDS)

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
    nutrition = st.session_state.medical_nutrition
    render_viiv_card(v)
    if nutrition:
        st.info("🧪 Bilan nutritionnel OCR actif : la douleur musculaire est ajustée avant la prédiction de zone.")

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
            if nutrition:
                payload_zone["medical_nutrition"] = nutrition_for_model(nutrition, ZONE_NUTRITION_FIELDS)

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
    nutrition = st.session_state.medical_nutrition
    render_viiv_card(v)
    if not v:
        st.warning("💡 Chargez les données Viiv GX17 dans le panneau de gauche pour activer l'analyse.")
    if nutrition:
        st.info("🧪 Bilan nutritionnel OCR actif : récupération, fatigue et stress seront ajustés.")

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
            if nutrition:
                payload_surv["medical_nutrition"] = nutrition_for_model(nutrition, RELAPSE_NUTRITION_FIELDS)
            
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

# ==========================================
# PAGE 4 : OCR RAPPORT MEDICAL (Vitamines/Mineraux)
# ==========================================
elif page == "🧾 M4 - OCR Rapport Medical":
    st.markdown('''
    <div class="header-container" style="background: linear-gradient(135deg, #0b3b2e 0%, #125a4a 100%);">
        <div class="logo-box">🧾</div>
        <div>
            <h1>OCR des Rapports Medicaux</h1>
            <p>Extraction automatique des vitamines et mineraux utiles a l'analyse de risque</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="dashboard-card"><div class="card-title">📥 Import du rapport</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Importer un rapport medical (image / pdf / docx)",
        type=["png", "jpg", "jpeg", "pdf", "docx"],
        help="Formats supportes: PNG, JPG, JPEG, PDF, DOCX.",
    )

    example_report = (
        "Patient: John Doe\n"
        "Date: 2026-07-20\n"
        "Vitamin D (25-OH D): 18 ng/mL\n"
        "Vitamin B12: 265 pg/mL\n"
        "Zinc: 62 ug/dL\n"
        "Magnesium serum: 1.5 mg/dL\n"
        "Ferritin: 410 ng/mL\n"
        "Clinical note: low energy and recurrent muscle fatigue.\n"
    )

    raw_text = st.text_area(
        "Ou coller le texte du rapport (exemple pre-rempli)",
        value=example_report,
        height=220,
    )

    analyze = st.button("🔍 Extraire Vitamines / Mineraux", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="dashboard-card"><div class="card-title">🧠 Resultat d\'analyse</div>', unsafe_allow_html=True)

    if analyze:
        files = None
        use_ocr_flag = "true"
        if uploaded_file is not None:
            fname = uploaded_file.name or "uploaded"
            ext = fname.rsplit('.', 1)[-1].lower() if "." in fname else ""
            # For document formats we prefer native text extraction (pdf/docx)
            if ext in ("pdf", "doc", "docx"):
                use_ocr_flag = "false"
            files = {
                "file": (
                    fname,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            }

        form_data = {
            "raw_text": raw_text,
            "use_ocr": use_ocr_flag,
        }

        with st.spinner("Lecture OCR + extraction des micronutriments en cours..."):
            try:
                res = requests.post(API_URL_MEDICAL_OCR, files=files, data=form_data)
                if res.status_code == 200:
                    payload = res.json()
                    nutrients_found = payload.get("nutrients_found", [])
                    mentions = payload.get("mentions", [])
                    flagged = payload.get("flagged", [])

                    # Sync OCR values into the global nutrition block used by
                    # all prediction pages, exactly like the Viiv session data.
                    ocr_nutrition = {
                        item["nutrient"]: item["value"]
                        for item in mentions
                        if item.get("nutrient") in NUTRITION_DEFAULTS
                        and item.get("value") is not None
                    }
                    if ocr_nutrition:
                        st.session_state.medical_nutrition = ocr_nutrition
                        st.success("🧪 Bilan OCR synchronisé avec les 3 modèles de prédiction.")

                    st.success(f"Extraction terminee. Nutriments detectes: {len(nutrients_found)}")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Nutriments detectes", len(nutrients_found))
                    c2.metric("Mentions extraites", len(mentions))
                    c3.metric("Anomalies (low/high)", len(flagged))

                    if nutrients_found:
                        st.markdown("**Nutriments detectes:** " + ", ".join(nutrients_found))

                    if mentions:
                        df_mentions = pd.DataFrame(mentions)
                        st.dataframe(df_mentions, use_container_width=True)
                    else:
                        st.info("Aucune mention nutriment/valeur detectee dans le texte fourni.")

                    if flagged:
                        st.markdown("### ⚠️ Micronutriments potentiellement a risque")
                        for item in flagged:
                            st.warning(
                                f"{item['nutrient']} = {item.get('value')} {item.get('unit') or ''} ({item['status']})"
                            )

                    with st.expander("Voir le texte analyse"):
                        st.text(payload.get("extracted_text", ""))
                else:
                    st.error(f"Erreur API ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Impossible de contacter FastAPI sur le port 8000. Détails: {e}")
    else:
        st.info("Importez une image medicale ou utilisez l'exemple texte, puis lancez l'extraction.")

    st.markdown('</div>', unsafe_allow_html=True)


# ====================================================================
# M5 — POSSESSION ESTIMATION
# ====================================================================
elif page == "⚽ M5 - Possession Estimation":
    st.markdown("""
    <div class="header-container">
        <div class="logo-box">⚽</div>
        <div>
            <h1>M5 — Possession Estimation</h1>
            <p>AI-powered ball possession analysis from match video footage</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📹 Upload Match Video</div>', unsafe_allow_html=True)

    st.info(
        "Upload a football match broadcast video (MP4, MKV, AVI). "
        "The AI will detect players, referees, and the ball, then compute Team A vs Team B possession."
    )

    col_upload, col_cfg = st.columns([2, 1])
    with col_upload:
        video_file = st.file_uploader(
            "📂 Select video file",
            type=["mp4", "mkv", "avi", "mov"],
            help="Supports MP4, MKV, AVI, MOV. Videos up to a few minutes work best."
        )

    with col_cfg:
        st.markdown("**⚙️ Analysis Settings**")
        max_frames = st.slider("Max frames to analyze", 50, 1000, 300, 50,
                               help="Fewer frames = faster. More = better accuracy.")
        conf_thresh = st.slider("Person detection threshold", 0.10, 0.80, 0.20, 0.05,
                                help="YOLO confidence threshold for player/referee detection.")
        ball_thresh = st.slider("Ball detection threshold", 0.05, 0.50, 0.10, 0.05,
                                help="YOLO confidence threshold for ball detection.")
        poss_radius = st.slider("Possession radius (px)", 30, 300, 100, 10,
                                help="Max pixel distance from ball to claim possession.")
        smooth_win  = st.slider("Smoothing window (frames)", 1, 30, 5, 1,
                                help="Majority-vote window to smooth possession labels.")

    st.markdown('</div>', unsafe_allow_html=True)

    if video_file is not None:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🚀 Run Analysis</div>', unsafe_allow_html=True)

        if st.button("⚽ Analyze Possession", use_container_width=True, type="primary"):
            with st.spinner("🔄 Uploading and processing video — this may take several minutes..."):
                try:
                    import requests as _req
                    files = {"video": (video_file.name, video_file.getvalue(),
                                       f"video/{video_file.type.split('/')[-1]}")}
                    data = {
                        "max_frames":       str(max_frames),
                        "conf_thresh":      str(conf_thresh),
                        "ball_conf_thresh": str(ball_thresh),
                        "poss_radius_px":   str(poss_radius),
                        "smoothing_window": str(smooth_win),
                    }
                    resp = _req.post(
                        "http://localhost:8000/possession/analyze",
                        files=files,
                        data=data,
                        timeout=600,
                    )

                    if resp.status_code == 200:
                        r = resp.json()
                        st.session_state["possession_result"] = r
                        st.success("✅ Analysis complete!")
                    else:
                        st.error(f"API error ({resp.status_code}): {resp.text}")
                except Exception as exc:
                    st.error(f"Could not reach FastAPI at localhost:8000. Details: {exc}")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Results Panel ──
    if "possession_result" in st.session_state and st.session_state.possession_result:
        r = st.session_state.possession_result

        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Possession Results</div>', unsafe_allow_html=True)

        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("⚽ Team A Possession",     f"{r.get('team_a_pct', 0):.1f}%")
        k2.metric("⚽ Team B Possession",     f"{r.get('team_b_pct', 0):.1f}%")
        k3.metric("🎯 Classifier Accuracy",   f"{r.get('classifier_val_acc', 0)*100:.1f}%")
        k4.metric("🖼️ Frames Processed",     str(r.get('total_frames_processed', '—')))

        st.markdown("---")

        # Detection breakdown
        d1, d2, d3 = st.columns(3)
        d1.metric("👕 Team A players detected", str(r.get('n_team_a', 0)))
        d2.metric("👕 Team B players detected", str(r.get('n_team_b', 0)))
        d3.metric("🦺 Referees detected",       str(r.get('n_referee', 0)))

        st.markdown("---")

        # Possession gauge chart
        pct_a = r.get('team_a_pct', 50.0)
        pct_b = r.get('team_b_pct', 50.0)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct_a,
            title={"text": "Team A Possession %", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#4f46e5"},
                "steps": [
                    {"range": [0, pct_a], "color": "rgba(79,70,229,0.15)"},
                    {"range": [pct_a, 100], "color": "rgba(239,68,68,0.10)"},
                ],
                "threshold": {"line": {"color": "#ef4444", "width": 4}, "value": 50},
            },
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20),
                                paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Bar chart
        fig_bar = go.Figure(data=[
            go.Bar(name="Team A", x=["Team A"], y=[pct_a],
                   marker_color="#4f46e5", text=[f"{pct_a}%"], textposition="outside"),
            go.Bar(name="Team B", x=["Team B"], y=[pct_b],
                   marker_color="#ef4444", text=[f"{pct_b}%"], textposition="outside"),
        ])
        fig_bar.update_layout(
            title="Overall Possession Split",
            yaxis_range=[0, 110],
            height=280,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Per-frame timeline (last 20 rows returned)
        records = r.get('per_frame_records', [])
        if records:
            df_pf = pd.DataFrame(records)
            fig_line = px.line(
                df_pf, x="time_sec", y=["team_a_pct", "team_b_pct"],
                labels={"value": "Possession %", "time_sec": "Time (s)", "variable": "Team"},
                title="Cumulative Possession Timeline",
                color_discrete_map={"team_a_pct": "#4f46e5", "team_b_pct": "#ef4444"},
            )
            fig_line.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_line, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Download Annotated Video ──
        job_id = r.get('job_id')
        if job_id:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🎬 Annotated Video</div>',
                        unsafe_allow_html=True)
            st.info(
                "Click the button below to download the annotated match video. "
                "Bounding boxes show: 🔴 Team A · 🔵 Team B · 🟡 Referee · 🟢 Ball. "
                "A real-time possession bar is overlaid at the top of each frame."
            )
            try:
                import requests as _req2
                vid_resp = _req2.get(
                    f"http://localhost:8000/possession/download/{job_id}",
                    stream=True, timeout=120)
                if vid_resp.status_code == 200:
                    st.download_button(
                        label="⬇️ Download Annotated Video",
                        data=vid_resp.content,
                        file_name=f"possession_{job_id}_annotated.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                else:
                    st.warning("Annotated video not yet available.")
            except Exception as exc:
                st.warning(f"Could not fetch video: {exc}")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🗑️ Clear Results", key="clear_poss"):
            del st.session_state["possession_result"]
            st.rerun()

    elif video_file is None:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; padding: 3rem 2rem; color:#64748b;">
            <div style="font-size:4rem; margin-bottom:1rem;">⚽</div>
            <h3 style="color:#1e293b;">Upload a Match Video to Begin</h3>
            <p>Supported formats: MP4, MKV, AVI, MOV</p>
            <p style="font-size:0.85rem; margin-top:1rem; color:#94a3b8;">
                The AI detects all on-pitch persons, filters out spectators, classifies<br>
                players into Team A / Team B, identifies the referee, and tracks ball possession
                frame-by-frame.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

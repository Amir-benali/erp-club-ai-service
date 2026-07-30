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
# 3. NAVIGATION ET ROUTES API
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/8112/8112465.png", width=60)
st.sidebar.markdown("## 🧭 Navigation IA")
page = st.sidebar.radio(
    "Sélectionnez un microservice :",
    [
        "🩺 M1 - Risque de Blessure (Global)", 
        "🗺️ M2 - Cartographie (Zones)", 
        "⏳ M3 - Analyse de Survie (Rechute)",
        "⚽ M4 - Analyse Post-Match (LLM)"
    ]
)
st.sidebar.markdown("---")
st.sidebar.caption("ERP Club AI v5.1.0 - Déploiement Local")

# Endpoints de l'API FastAPI
API_URL_GLOBAL = "http://localhost:8000/predict-injury"
API_URL_ZONE = "http://localhost:8000/predict-injury-zone"
API_URL_RELAPSE = "http://localhost:8000/predict-relapse"
API_URL_LLM_MATCH = "http://localhost:8000/generate-match-analysis"
API_URL_LLM_PLAYER = "http://localhost:8000/generate-player-insight"
API_URL_LLM_TACTICAL = "http://localhost:8000/generate-tactical-suggestion"



# ==========================================
# PAGE 1 : RISQUE GLOBAL (Modèle 1)
# ==========================================
if page == "🩺 M1 - Risque de Blessure (Global)":
    st.markdown('''
    <div class="header-container">
        <div class="logo-box">🩺</div>
        <div>
            <h1>Prédiction Globale de Blessure</h1>
            <p>Classification Binaire XGBoost : Analyse des charges et facteurs de fatigue</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    with st.sidebar: 
        st.markdown("### ⚙️ Configuration")
        model_choice = st.selectbox("Algorithme", ["XGBoost", "LightGBM"])
        player_id = st.number_input("ID Joueur", value=10, step=1)

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown('<div class="dashboard-card"><div class="card-title">📊 Paramètres Cliniques & GPS</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            sommeil = st.slider("💤 Sommeil (Qualité)", 1.0, 10.0, 7.5)
            fatigue = st.slider("⚡ Fatigue (RPE)", 1.0, 10.0, 4.0)
            acuteLoad = st.number_input("Acute Load (7j)", value=5950, step=100)
        with c2:
            douleur = st.slider("🦵 Douleurs Musculaires", 1.0, 10.0, 3.0)
            stress = st.slider("🧘 Niveau de Stress", 1.0, 10.0, 4.5)
            chronicLoad = st.number_input("Chronic Load (28j)", value=5100, step=100)
            
        totalLoad = st.number_input("Charge de travail prévue (Aujourd'hui)", value=850)
        acwr = float(acuteLoad / chronicLoad) if chronicLoad > 0 else 0
        
        acwr_color = "normal"
        if acwr > 1.5 or acwr < 0.8: acwr_color = "inverse"
        st.metric("Ratio ACWR (Acute:Chronic)", f"{acwr:.2f}", delta="Danger" if acwr>1.5 else "Optimal", delta_color=acwr_color)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dashboard-card"><div class="card-title">🤖 Diagnostic de l\'IA</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Lancer l'Analyse Prédictive", use_container_width=True):
            payload = {
                "playerId": player_id, "totalLoad": totalLoad, "sommeil": sommeil, 
                "fatigue": fatigue, "douleurMusculaire": douleur, "stress": stress, 
                "acuteLoad": acuteLoad, "chronicLoad": chronicLoad, "ACWR": acwr, 
                "model": model_choice
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
            <h1>Cartographie Anatomique des Risques</h1>
            <p>Modèle Multi-classe Random Forest : Prédiction des zones de vulnérabilité corporelle</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
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
        acute_zone = st.number_input("Acute Load (7j)", value=6000, step=100, key="z_acute")
        chronic_zone = st.number_input("Chronic Load (28j)", value=4500, step=100, key="z_chronic")
        acwr_zone = float(acute_zone / chronic_zone) if chronic_zone > 0 else 0
        st.metric("ACWR", f"{acwr_zone:.2f}")
        
        st.markdown("---")
        douleur_z = st.slider("Douleurs Actuelles (1-10)", 1.0, 10.0, 4.0, key="z_doul")
        souplesse_z = st.slider("Souplesse Globale (1-10)", 1.0, 10.0, 6.0, key="z_soup")
        agilite_z = st.slider("Test d'Agilité (1-10)", 1.0, 10.0, 8.0, key="z_agil")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dashboard-card"><div class="card-title">🧬 Analyse des Points de Rupture</div>', unsafe_allow_html=True)
        if st.button("🔥 Générer la Cartographie Corporelle", use_container_width=True):
            payload_zone = {
                "playerId": player_id, "position": position, "foot": foot, "age": age, 
                "fifa_rating": fifa, "acuteLoad": acute_zone, "chronicLoad": chronic_zone, 
                "ACWR": acwr_zone, "douleurMusculaire": douleur_z, "souplesse": souplesse_z, 
                "agilite": agilite_z
            }
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

    with st.sidebar:
        st.markdown("### 🏥 Bilan de Rééducation")
        player_id = st.number_input("ID Joueur", value=10, step=1)
        physio_adherence = st.slider("Adhérence au protocole Physio (%)", 0, 100, 85, help="Taux d'assiduité aux séances de kinésithérapie.")
        post_acwr = st.slider("ACWR projeté (Retour au jeu)", 0.5, 2.5, 1.1, step=0.1, help="Charge de travail accumulée prévue pour sa reprise.")

    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown('<div class="dashboard-card"><div class="card-title">🩺 État de Santé (Wellness)</div>', unsafe_allow_html=True)
        recovery = st.slider("Score de Récupération (0-100)", 0, 100, 75)
        sleep = st.slider("Qualité du Sommeil (1-10)", 1.0, 10.0, 7.5)
        stress = st.slider("Niveau de Stress (0-1)", 0.0, 1.0, 0.3)
        fatigue = st.slider("Index de Fatigue (0-100)", 0, 100, 45)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dashboard-card"><div class="card-title">📈 Projection de Survie au fil du temps</div>', unsafe_allow_html=True)
        
        if st.button("🔮 Calculer la Courbe de Survie (Cox PH)", use_container_width=True):
            payload_surv = {
                "playerId": player_id, "recovery_score": recovery, "sleep_quality": sleep,
                "stress_level": stress, "fatigue_index": fatigue, "physio_adherence": physio_adherence,
                "post_recovery_ACWR": post_acwr
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
# PAGE 4 : AUTOMATED POST-MATCH (MODULE 4)
# ==========================================
elif page == "⚽ M4 - Analyse Post-Match (LLM)":
    st.markdown("""
    <div class="header-container" style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);">
        <div class="logo-box">⚽</div>
        <div>
            <h1>Analyse Tactique & Rapports Génératifs</h1>
            <p>Module M4 : Intelligence Artificielle (GPT-4o-mini) à sorties JSON strictes et respect de la philosophie de jeu</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📊 Données de Rencontre")
        opponent = st.text_input("Adversaire", value="Olympique de Marseille")
        goals_for = st.number_input("Buts Marqués", value=2, min_value=0)
        goals_against = st.number_input("Buts Encaissés", value=1, min_value=0)
        result_text = "Victoire" if goals_for > goals_against else ("Défaite" if goals_for < goals_against else "Nul")
        formation = st.selectbox("Schéma Initial", ["4-3-3", "4-2-3-1", "3-5-2", "4-4-2"])
        philosophy = st.selectbox("Philosophie Tactique du Club", ["Jeu de Position (Possession)", "Gegenpressing (Klopp)", "Bloc Bas & Contre"])
        tactical_notes = st.text_area("Notes", "Pressing haut au niveau du rond central, relancer rapidement par les ailes.")

    col_inputs, col_results = st.columns([1, 1.2], gap="large")

    with col_inputs:
        st.markdown('<div class="dashboard-card"><div class="card-title">📊 Statistiques Collectives</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            possession = st.slider("Possession (%)", 30, 70, 58)
            pass_acc = st.slider("Précision Passes (%)", 50, 95, 84)
            shot_acc = st.slider("Tirs Cadrés (%)", 10, 90, 45)
            pressure_idx = st.slider("Pression Collective", 10, 100, 72)
        with c2:
            xg = st.number_input("Expected Goals (xG)", value=2.15, step=0.1)
            xga = st.number_input("Expected Goals Against (xGA)", value=1.05, step=0.1)
            ppda = st.number_input("PPDA (Intensité Pressing)", value=9.2, step=0.5)
            field_tilt = st.slider("Field Tilt (Domination %)", 10, 90, 64)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="dashboard-card"><div class="card-title">🛡️ Profil Tactique Observé</div>', unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        with t1:
            pressing = st.selectbox("Intensité Pressing", ["Élevée", "Moyenne", "Basse"])
            buildup = st.selectbox("Construction", ["Mixte", "Lente/Patiente", "Rapide"])
            wing_play = st.selectbox("Jeu sur les ailes", ["Intense", "Modéré", "Faible"])
        with t2:
            compactness = st.selectbox("Compacité", ["Bloc Compact", "Bloc Étiré", "Fragilité Axiale"])
            transition = st.selectbox("Vitesse Transition", ["Rapide", "Standard", "Lente"])
            setpiece = st.selectbox("CPA", ["Forte", "Neutre", "Faible"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        st.markdown('<div class="dashboard-card"><div class="card-title">🏃‍♂️ Performance Individuelle (Joueur Clé)</div>', unsafe_allow_html=True)
        j1, j2 = st.columns(2)
        with j1:
            p_name = st.text_input("Joueur", "Amine Gouiri")
            p_pos = st.selectbox("Poste", ["Ailier Gauche", "Milieu Central", "Buteur", "Défenseur"])
            p_rating = st.slider("Note Match", 1.0, 10.0, 8.1, step=0.1)
        with j2:
            p_goals = st.number_input("Buts", value=1, min_value=0)
            p_assists = st.number_input("Passes Dec.", value=1, min_value=0)
            p_sprints = st.number_input("Sprints (>25km/h)", value=14, min_value=0)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🔮 Lancer l'Analyse Tactique Intégrale", use_container_width=True):
            # Préparation des requêtes JSON
            payload_match = {
                "matchId": 101, "opponent": opponent, "result": result_text,
                "goalsFor": goals_for, "goalsAgainst": goals_against, "formation": formation,
                "tacticalNotes": tactical_notes, "clubPhilosophy": philosophy,
                "teamAnalytics": {
                    "possession": possession, "passAccuracy": pass_acc, "shotAccuracy": shot_acc,
                    "pressureIndex": pressure_idx, "xg": xg, "xga": xga, "ppda": ppda, "fieldTilt": field_tilt
                },
                "tacticalAnalysis": {
                    "pressingIntensity": pressing, "buildupSpeed": buildup, "wingPlay": wing_play,
                    "counterAttackEfficiency": "Standard", "defensiveCompactness": compactness,
                    "transitionSpeed": transition, "setPieceEffectiveness": setpiece
                },
                "events": [
                    {"minute": 12, "type": "BUT", "player": p_name, "detail": "Enroulé du pied droit"},
                    {"minute": 34, "type": "BUT", "player": "Adversaire", "detail": "Transition rapide"},
                    {"minute": 75, "type": "PASSE_DECISIVE", "player": p_name, "detail": "Passe millimétrée"}
                ],
                "playerPerformances": []
            }
            
            payload_player = {
                "playerId": 45, "playerName": p_name, "position": p_pos,
                "goals": p_goals, "assists": p_assists, "rating": p_rating,
                "distanceCovered": 10.8, "sprintCount": p_sprints,
                "passAccuracy": 86.5, "touchCount": 58
            }
            
            tab1, tab2, tab3 = st.tabs(["📋 Rapport Post-Match Structuré", "👤 Insight Joueur Clé", "💡 Recommandations Coach"])
            
            with tab1:
                with st.spinner("Génération du rapport de match..."):
                    try:
                        res = requests.post(API_URL_LLM_MATCH, json=payload_match)
                        if res.status_code == 200:
                            data = res.json()
                            st.markdown(f"### 📊 Synthèse Globale\n{data['global_summary']}")
                            
                            st.markdown("#### ⚡ Cadre Tactique")
                            c_p, c_t, c_d = st.columns(3)
                            with c_p:
                                st.info(f"**Phase Offensive / Possession**\n\n{data['tactical_framework']['possession_phase']['assessment']}\n\n*Corrélation statistique : {data['tactical_framework']['possession_phase']['metric_correlation']}*")
                            with c_t:
                                st.warning(f"**Phase de Transition**\n\n{data['tactical_framework']['transition_phase']['assessment']}\n\n*Corrélation statistique : {data['tactical_framework']['transition_phase']['metric_correlation']}*")
                            with c_d:
                                st.error(f"**Phase Défensive**\n\n{data['tactical_framework']['defensive_phase']['assessment']}\n\n*Corrélation statistique : {data['tactical_framework']['defensive_phase']['metric_correlation']}*")
                                
                            st.markdown("#### 📝 Directives d'Entraînement Prioritaires")
                            for idx, directive in enumerate(data['immediate_directives'], 1):
                                st.success(f"{idx}. {directive}")
                        else:
                            st.error(res.text)
                    except Exception as e:
                        st.error(f"Injoignable : {e}")

            with tab2:
                with st.spinner("Génération de l'évaluation joueur..."):
                    try:
                        res = requests.post(API_URL_LLM_PLAYER, json=payload_player)
                        if res.status_code == 200:
                            data = res.json()
                            st.markdown(f"### 👤 {p_name} ({p_pos})")
                            st.markdown(f"**🎯 Impact Tactique :**\n{data['tactical_impact']}")
                            st.markdown(f"**⚡ Analyse de la Charge Physique :**\n{data['physical_assessment']}")
                            st.markdown(f"**⚠️ Faille Technique à corriger :**\n{data['technical_flaw']}")
                            st.markdown(f"**🛠️ Exercice Dédié Recommandé :**\n*{data['targeted_drill']}*")
                        else:
                            st.error(res.text)
                    except Exception as e:
                        st.error(f"Injoignable : {e}")

            with tab3:
                with st.spinner("Génération des suggestions tactiques..."):
                    try:
                        res = requests.post(API_URL_LLM_TACTICAL, json=payload_match)
                        if res.status_code == 200:
                            data = res.json()
                            st.error(f"**Faille Identifiée dans le Bloc :**\n{data['vulnerability_identified']}")
                            st.warning(f"**Preuve Statistique à l'Appui :**\n{data['statistical_proof']}")
                            st.success(f"**Correctif Tactique Proposé :**\n{data['tactical_fix']}")
                            st.info(f"**Résultat Attendu :**\n{data['expected_outcome']}")
                        else:
                            st.error(res.text)
                    except Exception as e:
                        st.error(f"Injoignable : {e}")
        else:
            st.info("💡 Ajustez les paramètres tactiques ou statistiques et lancez l'Analyse Tactique Intégrale pour obtenir les synthèses détaillées de l'IA.")
        st.markdown('</div>', unsafe_allow_html=True)
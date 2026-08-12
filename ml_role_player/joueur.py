from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import random
from typing import List

app = FastAPI(
    title="ERP Club AI - Player Action Prediction",
    description="Microservice pour la prédiction de réussite des actions (Passes, Tirs, Dribbles) et génération de Heatmap",
    version="1.0.0"
)

# ---------------------------------------------------------
# CHARGEMENT DU MODÈLE (OU FALLBACK MATHÉMATIQUE)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "player_heatmap_model.joblib")
PERFORMANCE_MODEL_PATH = os.path.join(BASE_DIR, "models", "player_performance_model.joblib")

model_artifact = None
try:
    if os.path.exists(MODEL_PATH):
        model_artifact = joblib.load(MODEL_PATH)
        print("[OK] [MODULE JOUEUR] Modèle Heatmap chargé avec succès.")
    else:
        print(f"[WARN] [MODULE JOUEUR] Modèle introuvable à {MODEL_PATH}. Le fallback mathématique sera utilisé.")
except Exception as e:
    print(f"[WARN] [MODULE JOUEUR] Erreur au chargement du modèle : {e}. Le fallback mathématique sera utilisé.")

# Chargement du modèle de série temporelle de performance.
performance_model_artifact = None
try:
    if os.path.exists(PERFORMANCE_MODEL_PATH):
        performance_model_artifact = joblib.load(PERFORMANCE_MODEL_PATH)
        print("[OK] [MODULE JOUEUR] Modèle de performance temporelle chargé.")
    else:
        print(f"[WARN] [MODULE JOUEUR] Modèle temporel introuvable : {PERFORMANCE_MODEL_PATH}")
except Exception as e:
    print(f"[WARN] [MODULE JOUEUR] Erreur de chargement du modèle temporel : {e}")

# ---------------------------------------------------------
# SCHÉMAS DE DONNÉES (Pydantic)
# ---------------------------------------------------------
class ActionSuccessInput(BaseModel):
    playerId: int = 10
    x: float = Field(..., ge=0, le=120)
    y: float = Field(..., ge=0, le=80)
    action_type: str = "Pass" # "Pass", "Shot", "Carry"
    under_pressure: bool = False
    play_pattern: str = "Regular Play"


class PerformanceForecastInput(BaseModel):
    """Historique mensuel de scores de forme, du plus ancien au plus récent."""
    playerId: int = 10
    history: List[float] = Field(..., min_length=3)
    steps: int = Field(3, ge=1, le=6)
    matches_played: int = Field(3, ge=0, le=20)

# ---------------------------------------------------------
# ROUTES DE L'API
# ---------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "message": "API Joueur est en ligne 🟢",
        "performance_model_loaded": performance_model_artifact is not None,
    }


@app.get("/player-performance-model-info")
def get_player_performance_model_info():
    if performance_model_artifact is None:
        raise HTTPException(status_code=503, detail="Le modèle de performance temporelle n'est pas disponible.")
    artifact = performance_model_artifact
    return {
        "model_type": artifact.get("model_type"),
        "task": artifact.get("task"),
        "data_source": artifact.get("data_source"),
        "min_months": artifact.get("min_months"),
        "metrics": artifact.get("metrics", {}),
    }


@app.post("/predict-player-performance")
def predict_player_performance(data: PerformanceForecastInput):
    """Prévoit les prochains scores mensuels à partir de l'historique du joueur."""
    if performance_model_artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle indisponible. Exécutez d'abord le notebook 05_player_performance_timeseries.ipynb.",
        )

    artifact = performance_model_artifact
    lags = artifact.get("lags", [1, 2, 3])
    if len(data.history) < max(lags):
        raise HTTPException(status_code=422, detail=f"Au moins {max(lags)} scores historiques sont requis.")
    if any(score < 0 or score > 100 for score in data.history):
        raise HTTPException(status_code=422, detail="Chaque score de l'historique doit être compris entre 0 et 100.")

    model = artifact["model"]
    features = artifact["features"]
    roll_window = int(artifact.get("roll_window", 3))
    score_min, score_max = artifact.get("score_range", (0.0, 100.0))
    scores = [float(score) for score in data.history]
    predictions = []

    for _ in range(data.steps):
        row = {f"lag_{lag}": scores[-lag] for lag in lags}
        recent = scores[-roll_window:]
        row.update({
            "roll_mean": float(np.mean(recent)),
            "roll_std": float(np.std(recent)) if len(recent) > 1 else 0.0,
            "expanding_mean": float(np.mean(scores)),
            "delta_prev": float(scores[-1] - scores[-2]) if len(scores) > 1 else 0.0,
            "month_index": len(scores),
            "matches_played": data.matches_played,
        })
        prediction = float(model.predict(pd.DataFrame([row])[features])[0])
        prediction = float(np.clip(prediction, score_min, score_max))
        predictions.append(round(prediction, 2))
        scores.append(prediction)

    return {
        "playerId": data.playerId,
        "history": [round(value, 2) for value in data.history],
        "predictions": predictions,
        "steps": data.steps,
        "score_range": [score_min, score_max],
        "source": "player_performance_timeseries_model",
        "model_metrics": artifact.get("metrics", {}),
    }

@app.post("/predict-action-success")
def predict_action_success(data: ActionSuccessInput):
    # Calculs géométriques de base
    goal_x, goal_y = 120.0, 40.0
    distance = np.sqrt((goal_x - data.x)**2 + (goal_y - data.y)**2)
    angle = np.arctan2(goal_y - data.y, goal_x - data.x) * (180 / np.pi)

    if model_artifact:
        try:
            model = model_artifact['model']
            scaler = model_artifact['scaler']
            encoders = model_artifact['label_encoders']
            features = model_artifact['features']
            
            # Préparation des données
            input_data = {
                'x': data.x,
                'y': data.y,
                'Distance': distance,
                'Angle': angle,
                'action_type': data.action_type,
                'play_pattern': data.play_pattern,
                'under_pressure': int(data.under_pressure)
            }
            
            # Encodage catégoriel
            for col in ['action_type', 'play_pattern']:
                if col in encoders:
                    le = encoders[col]
                    # Gérer les classes inconnues
                    if input_data[col] in le.classes_:
                        input_data[col] = le.transform([input_data[col]])[0]
                    else:
                        input_data[col] = 0 # Valeur par défaut
            
            df_input = pd.DataFrame([input_data])[features]
            
            # Scaling
            continuous_cols = ['x', 'y', 'Distance', 'Angle']
            df_input[continuous_cols] = scaler.transform(df_input[continuous_cols])
            
            # Prédiction
            prob_success = float(model.predict_proba(df_input)[0][1])
            
            return {
                "playerId": data.playerId,
                "success_probability": round(prob_success, 4),
                "distance_to_goal": round(distance, 2),
                "source": "ml_model"
            }
        except Exception as e:
            print(f"Erreur avec le modèle ML ({e}), bascule vers le fallback.")
            
    # Moteur de Fallback Mathématique de Secours
    # Formule : P(Succès) = clip(1.0 - 0.004 * Distance - (0.25 si sous_pression) + bruit, 0.05, 0.98)
    noise = random.uniform(-0.05, 0.05)
    pressure_penalty = 0.25 if data.under_pressure else 0.0
    
    # Ajustement selon le type d'action
    action_penalty = 0.0
    if data.action_type == "Shot":
        action_penalty = 0.2 # Les tirs sont globalement moins réussis que les passes
    elif data.action_type == "Carry":
        action_penalty = -0.1 # Les conduites de balle sont plus sûres
        
    p_success = 1.0 - (0.004 * distance) - pressure_penalty - action_penalty + noise
    p_success = max(0.05, min(0.98, p_success))
    
    return {
        "playerId": data.playerId,
        "success_probability": round(p_success, 4),
        "distance_to_goal": round(distance, 2),
        "source": "fallback_formula"
    }

@app.get("/player-season-heatmap")
def get_player_season_heatmap(playerId: int = 10):
    """
    Retourne un historique réaliste de 128 événements de jeu géolocalisés.
    Biaisé vers la zone offensive droite ("Demi-espace droit / Surface adverse").
    StatsBomb coords: X(0->120), Y(0->80).
    Offensive droite = X élevé (ex: 70-120), Y élevé (ex: 40-80 pour le côté droit en vue attaquante si 0 est à gauche, ou 0-40, ça dépend de la convention. Prenons 50-80 pour correspondre à "bas du terrain" sur un plot classique).
    """
    events = []
    
    # On génère 128 événements, avec 75% concentrés dans la zone favorite
    for i in range(128):
        if random.random() < 0.75:
            # Zone favorite: Demi-espace droit / Surface adverse
            # X dans le dernier tiers: 80 à 120
            # Y sur le côté droit: 50 à 80
            x = np.random.normal(100, 10)
            y = np.random.normal(65, 8)
        else:
            # Reste du terrain
            x = np.random.uniform(20, 120)
            y = np.random.uniform(0, 80)
            
        # S'assurer qu'on reste dans les limites du terrain StatsBomb (120x80)
        x = max(0, min(120, x))
        y = max(0, min(80, y))
        
        # Déterminer un type d'action et un succès aléatoire logique
        action = random.choices(["Pass", "Shot", "Carry"], weights=[0.6, 0.1, 0.3])[0]
        
        events.append({
            "id": i,
            "x": round(x, 2),
            "y": round(y, 2),
            "action_type": action,
            "success": 1 if random.random() > 0.3 else 0
        })
        
    return {
        "playerId": playerId,
        "total_actions": len(events),
        "events": events
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

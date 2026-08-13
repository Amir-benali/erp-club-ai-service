"""
Player Performance & Heatmap router.

Exposes the two `ml_role_player` models (season heatmap generator + trained
performance time-series forecaster) on the main AI service (`app.main:app`).

The model-serving logic mirrors the standalone dev app in
`ml_role_player/joueur.py`; that file is kept as-is for local/standalone runs,
while this router is what gets mounted into the deployed service.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os
import random

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# MODEL LOADING (repo-relative, fail-soft)
# ---------------------------------------------------------
# BASE_DIR = repo root (parent of the `app/` package), same convention as app/main.py.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "ml_role_player", "models")
HEATMAP_MODEL_PATH = os.path.join(MODELS_DIR, "player_heatmap_model.joblib")
PERFORMANCE_MODEL_PATH = os.path.join(MODELS_DIR, "player_performance_model.joblib")

heatmap_model_artifact = None
try:
    if os.path.exists(HEATMAP_MODEL_PATH):
        heatmap_model_artifact = joblib.load(HEATMAP_MODEL_PATH)
        print("[OK] [MODULE JOUEUR] Modele Heatmap (action-success) charge.")
    else:
        print(f"[WARN] [MODULE JOUEUR] Modele Heatmap introuvable : {HEATMAP_MODEL_PATH}. Fallback mathematique.")
except Exception as e:
    print(f"[WARN] [MODULE JOUEUR] Erreur chargement modele Heatmap : {e}. Fallback mathematique.")

performance_model_artifact = None
try:
    if os.path.exists(PERFORMANCE_MODEL_PATH):
        performance_model_artifact = joblib.load(PERFORMANCE_MODEL_PATH)
        print("[OK] [MODULE JOUEUR] Modele de performance temporelle charge.")
    else:
        print(f"[WARN] [MODULE JOUEUR] Modele temporel introuvable : {PERFORMANCE_MODEL_PATH}")
except Exception as e:
    print(f"[WARN] [MODULE JOUEUR] Erreur chargement modele temporel : {e}")


# ---------------------------------------------------------
# SCHEMAS (Pydantic)
# ---------------------------------------------------------
class ActionSuccessInput(BaseModel):
    playerId: int = 10
    x: float = Field(..., ge=0, le=120)
    y: float = Field(..., ge=0, le=80)
    action_type: str = "Pass"  # "Pass", "Shot", "Carry"
    under_pressure: bool = False
    play_pattern: str = "Regular Play"


class PerformanceForecastInput(BaseModel):
    """Historique mensuel de scores de forme, du plus ancien au plus recent."""
    playerId: int = 10
    history: List[float] = Field(..., min_length=3)
    steps: int = Field(3, ge=1, le=6)
    matches_played: int = Field(3, ge=0, le=20)


# ---------------------------------------------------------
# ROUTER
# ---------------------------------------------------------
router = APIRouter(tags=["Player Performance & Heatmap"])


@router.get(
    "/player-performance-model-info",
    summary="Trained performance model metadata",
)
def get_player_performance_model_info():
    if performance_model_artifact is None:
        raise HTTPException(status_code=503, detail="Le modele de performance temporelle n'est pas disponible.")
    artifact = performance_model_artifact
    return {
        "model_type": artifact.get("model_type"),
        "task": artifact.get("task"),
        "data_source": artifact.get("data_source"),
        "min_months": artifact.get("min_months"),
        "metrics": artifact.get("metrics", {}),
    }


@router.post(
    "/predict-player-performance",
    summary="Forecast next monthly form scores (trained time-series model)",
)
def predict_player_performance(data: PerformanceForecastInput):
    """Prevoit les prochains scores mensuels a partir de l'historique du joueur."""
    if performance_model_artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Modele indisponible. Executez d'abord le notebook 05_player_performance_timeseries.ipynb.",
        )

    artifact = performance_model_artifact
    lags = artifact.get("lags", [1, 2, 3])
    if len(data.history) < max(lags):
        raise HTTPException(status_code=422, detail=f"Au moins {max(lags)} scores historiques sont requis.")
    if any(score < 0 or score > 100 for score in data.history):
        raise HTTPException(status_code=422, detail="Chaque score de l'historique doit etre compris entre 0 et 100.")

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


@router.post(
    "/predict-action-success",
    summary="Probability an action (Pass/Shot/Carry) succeeds from a pitch location",
)
def predict_action_success(data: ActionSuccessInput):
    # Calculs geometriques de base
    goal_x, goal_y = 120.0, 40.0
    distance = np.sqrt((goal_x - data.x) ** 2 + (goal_y - data.y) ** 2)
    angle = np.arctan2(goal_y - data.y, goal_x - data.x) * (180 / np.pi)

    if heatmap_model_artifact:
        try:
            model = heatmap_model_artifact['model']
            scaler = heatmap_model_artifact['scaler']
            encoders = heatmap_model_artifact['label_encoders']
            features = heatmap_model_artifact['features']

            input_data = {
                'x': data.x,
                'y': data.y,
                'Distance': distance,
                'Angle': angle,
                'action_type': data.action_type,
                'play_pattern': data.play_pattern,
                'under_pressure': int(data.under_pressure),
            }

            for col in ['action_type', 'play_pattern']:
                if col in encoders:
                    le = encoders[col]
                    if input_data[col] in le.classes_:
                        input_data[col] = le.transform([input_data[col]])[0]
                    else:
                        input_data[col] = 0

            df_input = pd.DataFrame([input_data])[features]

            continuous_cols = ['x', 'y', 'Distance', 'Angle']
            df_input[continuous_cols] = scaler.transform(df_input[continuous_cols])

            prob_success = float(model.predict_proba(df_input)[0][1])

            return {
                "playerId": data.playerId,
                "success_probability": round(prob_success, 4),
                "distance_to_goal": round(distance, 2),
                "source": "ml_model",
            }
        except Exception as e:
            print(f"Erreur avec le modele ML ({e}), bascule vers le fallback.")

    # Moteur de fallback mathematique de secours
    noise = random.uniform(-0.05, 0.05)
    pressure_penalty = 0.25 if data.under_pressure else 0.0

    action_penalty = 0.0
    if data.action_type == "Shot":
        action_penalty = 0.2
    elif data.action_type == "Carry":
        action_penalty = -0.1

    p_success = 1.0 - (0.004 * distance) - pressure_penalty - action_penalty + noise
    p_success = max(0.05, min(0.98, p_success))

    return {
        "playerId": data.playerId,
        "success_probability": round(p_success, 4),
        "distance_to_goal": round(distance, 2),
        "source": "fallback_formula",
    }


@router.get(
    "/player-season-heatmap",
    summary="Season heatmap events (StatsBomb coords, biased to attacking-right)",
)
def get_player_season_heatmap(playerId: int = 10):
    """
    Retourne un historique realiste de 128 evenements de jeu geolocalises.
    Biaise vers la zone offensive droite ("Demi-espace droit / Surface adverse").
    StatsBomb coords: X(0->120), Y(0->80).
    """
    events = []

    for i in range(128):
        if random.random() < 0.75:
            x = np.random.normal(100, 10)
            y = np.random.normal(65, 8)
        else:
            x = np.random.uniform(20, 120)
            y = np.random.uniform(0, 80)

        x = max(0, min(120, x))
        y = max(0, min(80, y))

        action = random.choices(["Pass", "Shot", "Carry"], weights=[0.6, 0.1, 0.3])[0]

        events.append({
            "id": i,
            "x": round(x, 2),
            "y": round(y, 2),
            "action_type": action,
            "success": 1 if random.random() > 0.3 else 0,
        })

    return {
        "playerId": playerId,
        "total_actions": len(events),
        "events": events,
    }

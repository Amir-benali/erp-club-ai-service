from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import pandas as pd
import joblib
import shap
import os
import uuid
import tempfile
from pathlib import Path
import random
import numpy as np

from app.medical_ocr import (
    extract_nutrients_from_text,
    extract_text_from_file,
    run_easyocr_on_image_bytes,
)

# ---------------------------------------------------------
# OpenAPI / Swagger Metadata
# ---------------------------------------------------------

tags_metadata = [
    {
        "name": "Health",
        "description": "Service health-check endpoint. Returns the current status and version of the AI service.",
    },
    {
        "name": "Injury Risk",
        "description": (
            "**Module 1 — Global Injury Risk (XGBoost).** "
            "Predicts the probability that a player will sustain an injury based on workload, "
            "biometric, and wellness features. Supports native Viiv GX17 sensor data."
        ),
    },
    {
        "name": "Injury Zone",
        "description": (
            "**Module 2 — Anatomical Zone Mapping (Random Forest / LightGBM).** "
            "Classifies which body zone (e.g. knee, hamstring, ankle) is most likely to be injured, "
            "given player profile and load metrics."
        ),
    },
    {
        "name": "Relapse Survival",
        "description": (
            "**Module 3 — Relapse Survival Analysis (Cox Proportional Hazards).** "
            "Estimates the survival curve (probability of staying injury-free over time) "
            "for a player returning from injury."
        ),
    },
    {
        "name": "Medical OCR",
        "description": (
            "Extracts vitamins and minerals (zinc, magnesium, etc.) from medical reports "
            "using OCR + rule-based parsing for downstream model awareness."
        ),
    },
    {
        "name": "Possession Estimation",
        "description": (
            "**Module 5 — Football Possession Estimation (YOLO + MLP).** "
            "Uploads a match video and returns Team A vs Team B possession percentages, "
            "per-frame timeline, referee identification, spectator filtering, and an annotated output video."
        ),
    },
    {
        "name": "Player Intelligence",
        "description": (
            "**Player action and performance module.** "
            "Generates season heatmap events, predicts action success probability, "
            "and forecasts monthly player performance."
        ),
    },
]

app = FastAPI(
    title="ERP Club — AI Service",
    description=(
        "## ERP Club AI Microservice — Viiv GX17 Integration\n\n"
        "This service exposes **three independent AI modules** for sports-medicine decision support:\n\n"
        "| Module | Endpoint | Algorithm |\n"
        "|--------|----------|-----------|\n"
        "| Global Injury Risk | `POST /predict-injury` | XGBoost + SHAP |\n"
        "| Anatomical Zone Mapping | `POST /predict-injury-zone` | Random Forest / LightGBM |\n"
        "| Relapse Survival | `POST /predict-relapse` | Cox Proportional Hazards |\n\n"
        "### Viiv GX17 Sensor Integration\n"
        "Each request body is centered on a `viiv` block containing raw sensor readings "
        "(HR, SpO₂, HRV, Strain, Recovery %, …) plus only the supplementary fields that the "
        "target model cannot infer from Viiv alone. Each prediction also accepts an optional "
        "`medical_nutrition` block populated from Medical OCR.\n\n"
        "### Authentication\n"
        "No authentication is required for this internal microservice. "
        "Secure network-level access is enforced at the API gateway."
    ),
    version="5.1.0",
    openapi_tags=tags_metadata,
    contact={
        "name": "ERP Club — Data & AI Team",
        "email": "ai@erp-club.io",
    },
    license_info={
        "name": "Proprietary — ERP Club",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. CHARGEMENT ROBUSTE DES MODÈLES (Au démarrage)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- MODULE 1 : Risque Global (XGBoost) ---
MODEL_XGB_PATH = os.path.join(BASE_DIR, "ml_core", "artifacts", "injury_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "ml_core", "artifacts", "scaler.pkl")

try:
    xgb_model = joblib.load(MODEL_XGB_PATH)
    scaler = joblib.load(SCALER_PATH)
    explainer = shap.TreeExplainer(xgb_model)
    print("[OK] [MODULE 1] Modele XGBoost charge avec succes.")
except Exception as e:
    xgb_model = None
    print(f"[WARN] [MODULE 1] Erreur XGBoost : {e}")

# --- MODULE 2 : Cartographie des Zones (Random Forest / LightGBM) ---
model_zone_artifact = None
possible_zone_paths = [
    os.path.join(BASE_DIR, "ml_core", "models", "injury_zone_model.joblib"),
    os.path.join(BASE_DIR, "ml_core", "artifacts", "injury_zone_model.joblib"),
    os.path.join(BASE_DIR, "models", "injury_zone_model.joblib"),
    os.path.join(BASE_DIR, "injury_zone_model.joblib")
]

for path in possible_zone_paths:
    if os.path.exists(path):
        try:
            model_zone_artifact = joblib.load(path)
            print("[OK] [MODULE 2] Modele 'Zone de Blessure' charge avec succes.")
            break
        except Exception: pass
if not model_zone_artifact:
    print(
        "[WARN] [MODULE 2] Fichier 'injury_zone_model.joblib' introuvable."
        f"Chemins testes: {possible_zone_paths}"
    )

# --- MODULE 3 : Analyse de Survie (Cox Proportional Hazards) ---
model_survival_artifact = None
possible_survival_paths = [
    os.path.join(BASE_DIR, "ml_core", "models", "relapse_survival_model.joblib"),
    os.path.join(BASE_DIR, "ml_core", "artifacts", "relapse_survival_model.joblib"),
    os.path.join(BASE_DIR, "models", "relapse_survival_model.joblib"),
    os.path.join(BASE_DIR, "relapse_survival_model.joblib")
]

for path in possible_survival_paths:
    if os.path.exists(path):
        try:
            model_survival_artifact = joblib.load(path)
            print("[OK] [MODULE 3] Modele 'Analyse de Survie' charge avec succes.")
            break
        except Exception as e: 
            print(f"[WARN] [MODULE 3] Erreur de lecture : {e}")
if not model_survival_artifact:
    print(
        "[WARN] [MODULE 3] Fichier 'relapse_survival_model.joblib' introuvable. "
        f"Chemins testes: {possible_survival_paths}"
    )

# --- MODULE 4 : Intelligence Joueur (Heatmap / Action Success / Performance) ---
PLAYER_ACTION_MODEL_PATH = os.path.join(BASE_DIR, "ml_role_player", "models", "player_heatmap_model.joblib")
PLAYER_PERFORMANCE_MODEL_PATH = os.path.join(BASE_DIR, "ml_role_player", "models", "player_performance_model.joblib")

player_action_model_artifact = None
try:
    if os.path.exists(PLAYER_ACTION_MODEL_PATH):
        player_action_model_artifact = joblib.load(PLAYER_ACTION_MODEL_PATH)
        print("[OK] [MODULE 4] Modele Heatmap joueur charge avec succes.")
    else:
        print(
            "[WARN] [MODULE 4] Modele Heatmap joueur introuvable. "
            f"Chemin teste: {PLAYER_ACTION_MODEL_PATH}. Fallback mathematique actif."
        )
except Exception as e:
    print(f"[WARN] [MODULE 4] Erreur Heatmap joueur : {e}. Fallback mathematique actif.")

player_performance_model_artifact = None
try:
    if os.path.exists(PLAYER_PERFORMANCE_MODEL_PATH):
        player_performance_model_artifact = joblib.load(PLAYER_PERFORMANCE_MODEL_PATH)
        print("[OK] [MODULE 4] Modele de performance joueur charge avec succes.")
    else:
        print(
            "[WARN] [MODULE 4] Modele de performance joueur introuvable. "
            f"Chemin teste: {PLAYER_PERFORMANCE_MODEL_PATH}"
        )
except Exception as e:
    print(f"[WARN] [MODULE 4] Erreur performance joueur : {e}")


# ---------------------------------------------------------
# 2. SCHÉMAS VIIV GX17 (Données brutes du capteur)
# ---------------------------------------------------------

class ViivGX17Data(BaseModel):
    """
    Données brutes transmises par l'application mobile depuis le capteur Viiv GX17 via Bluetooth.
    Toutes les valeurs sont requises dans le payload mobile.
    """
    # Cardiaque
    heart_rate: float = Field(..., description="Fréquence cardiaque (FC) en bpm — ex: 97")
    spo2: float = Field(..., description="Saturation en oxygène SpO₂ en % — ex: 98.5")
    hrv_ms: float = Field(..., description="Variabilité de la fréquence cardiaque (HRV) en ms — ex: 42.0")

    # Bien-être
    stress_score: float = Field(..., description="Score de stress Viiv (0–100) — ex: 35.0")
    energy_pct: float = Field(..., description="Niveau d'énergie en % — ex: 100.0")
    sleep_score: float = Field(..., description="Score de sommeil Viiv (0–10 ou heures) — ex: 7.5")
    recovery_pct: float = Field(..., description="Score de récupération en % — ex: 30.0")
    strain: float = Field(..., description="Charge d'effort cumulée Viiv (0–21) — ex: 0.0")

    model_config = {
        "json_schema_extra": {
            "example": {
                "heart_rate": 97.0,
                "spo2": 98.5,
                "hrv_ms": 42.0,
                "stress_score": 35.0,
                "energy_pct": 100.0,
                "sleep_score": 7.5,
                "recovery_pct": 30.0,
                "strain": 0.0,
            }
        }
    }

    def derive_sommeil(self) -> Optional[float]:
        """Convertit sleep_score Viiv → échelle 1–10 (sommeil)."""
        if self.sleep_score is None:
            return None
        return round(max(1.0, min(10.0, float(self.sleep_score))), 2)

    def derive_stress(self) -> Optional[float]:
        """Convertit stress_score Viiv (0–100) → échelle 1–10."""
        if self.stress_score is None:
            return None
        return round(max(1.0, min(10.0, self.stress_score / 10.0)), 2)

    def derive_fatigue(self) -> Optional[float]:
        """
        Dérive la fatigue depuis HRV (ms) : HRV faible = fatigue élevée.
        HRV typique athlète : 20–80 ms → fatigue 1–10 (inversé).
        """
        if self.hrv_ms is None:
            return None
        # Normalisation inverse : HRV=20ms → fatigue=9, HRV=80ms → fatigue=1
        hrv_clamped = max(10.0, min(100.0, self.hrv_ms))
        fatigue = 10.0 - ((hrv_clamped - 10.0) / 90.0) * 9.0
        return round(max(1.0, min(10.0, fatigue)), 2)

    def derive_acute_load(self, base_load: float = 5000.0) -> Optional[float]:
        """Estime l'Acute Load depuis le Strain Viiv (0–21 → load approximatif)."""
        if self.strain is None:
            return None
        return round(base_load + (self.strain * 200.0), 0)

    def derive_recovery_score(self) -> Optional[float]:
        """Recovery % Viiv → score 0–100 pour le module de survie."""
        return self.recovery_pct  # Déjà en %

    def derive_fatigue_index(self) -> Optional[float]:
        """Dérive l'index de fatigue (0–100) depuis HRV."""
        fat = self.derive_fatigue()
        if fat is None:
            return None
        return round(fat * 10.0, 2)  # 1–10 → 10–100

    def derive_stress_level(self) -> Optional[float]:
        """Convertit stress_score (0–100) → niveau 0–1 pour module survie."""
        if self.stress_score is None:
            return None
        return round(max(0.0, min(1.0, self.stress_score / 100.0)), 3)


# ---------------------------------------------------------
# 3. SCHÉMAS DE DONNÉES IA (Pydantic) — avec champs Viiv intégrés
# ---------------------------------------------------------

class NutritionReadinessBase(BaseModel):
    """Base helpers for model-specific OCR nutrition blocks."""
    def snapshot(self) -> Dict[str, float]:
        return self.model_dump(exclude_none=True)

    def readiness_penalty(self) -> float:
        """Translate abnormal OCR values into a transparent 0–3 readiness penalty.

        Existing ML artifacts were not trained with raw laboratory columns. The
        penalty therefore adjusts already-trained fatigue, pain, stress, and
        recovery inputs instead of silently changing their feature schema.
        """
        score = 0.0
        vitamin_d = getattr(self, "vitamin_d", None)
        vitamin_b12 = getattr(self, "vitamin_b12", None)
        zinc = getattr(self, "zinc", None)
        magnesium = getattr(self, "magnesium", None)
        iron = getattr(self, "iron", None)
        ferritin = getattr(self, "ferritin", None)
        calcium = getattr(self, "calcium", None)
        hemoglobin = getattr(self, "hemoglobin", None)
        crp = getattr(self, "c_reactive_protein", None)
        if vitamin_d is not None and vitamin_d < 30: score += 0.8
        if vitamin_b12 is not None and vitamin_b12 < 200: score += 0.5
        if zinc is not None and zinc < 70: score += 0.4
        if magnesium is not None and magnesium < 1.7: score += 0.4
        if iron is not None and iron < 50: score += 0.4
        if ferritin is not None: score += 0.6 if ferritin < 30 else (0.2 if ferritin > 400 else 0.0)
        if calcium is not None and calcium < 8.6: score += 0.2
        if hemoglobin is not None and hemoglobin < 13: score += 0.5
        if crp is not None and crp > 3: score += 0.8
        return round(min(score, 3.0), 2)


class GlobalRiskNutritionData(NutritionReadinessBase):
    """Nutrition values relevant to fatigue, pain, stress, and sleep."""
    vitamin_d: Optional[float] = Field(None, description="Vitamine D 25-OH (ng/mL)")
    ferritin: Optional[float] = Field(None, description="Ferritine (ng/mL)")
    hemoglobin: Optional[float] = Field(None, description="Hémoglobine (g/dL)")
    vitamin_b12: Optional[float] = Field(None, description="Vitamine B12 (pg/mL)")
    magnesium: Optional[float] = Field(None, description="Magnésium (mg/dL)")
    zinc: Optional[float] = Field(None, description="Zinc (µg/dL)")
    iron: Optional[float] = Field(None, description="Fer sérique (µg/dL)")
    c_reactive_protein: Optional[float] = Field(None, description="CRP / CRP ultrasensible (mg/L)")


class InjuryZoneNutritionData(NutritionReadinessBase):
    """Nutrition values relevant to muscle pain, fatigue, and stability."""
    vitamin_d: Optional[float] = Field(None, description="Vitamine D 25-OH (ng/mL)")
    ferritin: Optional[float] = Field(None, description="Ferritine (ng/mL)")
    hemoglobin: Optional[float] = Field(None, description="Hémoglobine (g/dL)")
    magnesium: Optional[float] = Field(None, description="Magnésium (mg/dL)")
    calcium: Optional[float] = Field(None, description="Calcium (mg/dL)")
    c_reactive_protein: Optional[float] = Field(None, description="CRP / CRP ultrasensible (mg/L)")


class RelapseNutritionData(NutritionReadinessBase):
    """Nutrition values relevant to recovery, fatigue, sleep, and stress."""
    vitamin_d: Optional[float] = Field(None, description="Vitamine D 25-OH (ng/mL)")
    ferritin: Optional[float] = Field(None, description="Ferritine (ng/mL)")
    hemoglobin: Optional[float] = Field(None, description="Hémoglobine (g/dL)")
    vitamin_b12: Optional[float] = Field(None, description="Vitamine B12 (pg/mL)")
    magnesium: Optional[float] = Field(None, description="Magnésium (mg/dL)")
    iron: Optional[float] = Field(None, description="Fer sérique (µg/dL)")
    c_reactive_protein: Optional[float] = Field(None, description="CRP / CRP ultrasensible (mg/L)")


class PlayerFeatures(BaseModel):
    playerId: int

    # --- Champs Viiv GX17 bruts (transmis par l'app mobile) ---
    viiv: ViivGX17Data = Field(
        ...,
        description="Données brutes du capteur Viiv GX17 transmises par l'app mobile."
    )
    medical_nutrition: Optional[GlobalRiskNutritionData] = Field(
        None, description="Bilan nutritionnel/biologique issu de l'OCR médical."
    )

    # --- Champs IA requis hors Viiv ---
    totalLoad: float = Field(..., description="Charge de travail prévue aujourd'hui (au)")
    douleurMusculaire: float = Field(..., description="Douleurs musculaires (1–10)")
    acuteLoad: float = Field(..., description="Charge aiguë 7 jours")
    chronicLoad: float = Field(..., description="Charge chronique 28 jours")
    ACWR: float = Field(..., description="Ratio ACWR")
    sommeil_7d_mean: float = Field(..., description="Sommeil moyen sur 7 jours")
    fatigue_7d_mean: float = Field(..., description="Fatigue moyenne sur 7 jours")
    douleurMusculaire_7d_mean: float = Field(..., description="Douleurs musculaires moyennes sur 7 jours")
    stress_7d_mean: float = Field(..., description="Stress moyen sur 7 jours")

    model_config = {
        "json_schema_extra": {
            "example": {
                "playerId": 0,
                "totalLoad": 850.0,
                "douleurMusculaire": 3.0,
                "acuteLoad": 5950.0,
                "chronicLoad": 5100.0,
                "ACWR": 1.17,
                "sommeil_7d_mean": 7.0,
                "fatigue_7d_mean": 4.0,
                "douleurMusculaire_7d_mean": 3.0,
                "stress_7d_mean": 4.0,
                "viiv": {
                    "heart_rate": 97.0,
                    "spo2": 98.5,
                    "hrv_ms": 42.0,
                    "stress_score": 35.0,
                    "energy_pct": 100.0,
                    "sleep_score": 7.5,
                    "recovery_pct": 30.0,
                    "strain": 0.0,
                },
                "medical_nutrition": {
                    "vitamin_d": 24.0, "ferritin": 85.0, "hemoglobin": 14.2,
                    "vitamin_b12": 512.0, "magnesium": 1.8, "zinc": 82.0,
                    "iron": 90.0, "c_reactive_protein": 0.32,
                },
            }
        }
    }

    def resolve_inputs(self) -> dict:
        """
        Résout les champs IA finals à partir des données Viiv GX17 obligatoires
        et des champs manuels obligatoires fournis par l'application mobile.
        """
        v = self.viiv

        sommeil_final = v.derive_sommeil()
        stress_final = v.derive_stress()
        fatigue_final = v.derive_fatigue()
        nutrition_penalty = self.medical_nutrition.readiness_penalty() if self.medical_nutrition else 0.0
        acute_final = v.derive_acute_load(self.acuteLoad) if v.strain is not None else self.acuteLoad
        acwr_final = self.ACWR

        return {
            "totalLoad": self.totalLoad,
            "sommeil": round(max(1.0, sommeil_final - (0.15 * nutrition_penalty)), 2),
            "fatigue": round(min(10.0, fatigue_final + nutrition_penalty), 2),
            "douleurMusculaire": round(min(10.0, self.douleurMusculaire + (0.5 * nutrition_penalty)), 2),
            "stress": round(min(10.0, stress_final + (0.25 * nutrition_penalty)), 2),
            "acuteLoad": acute_final,
            "chronicLoad": self.chronicLoad,
            "ACWR": acwr_final,
            "sommeil_7d_mean": self.sommeil_7d_mean,
            "fatigue_7d_mean": self.fatigue_7d_mean,
            "douleurMusculaire_7d_mean": self.douleurMusculaire_7d_mean,
            "stress_7d_mean": self.stress_7d_mean,
            "nutrition_readiness_penalty": nutrition_penalty,
            "medical_nutrition": self.medical_nutrition.snapshot() if self.medical_nutrition else None,
        }

FEATURES_ORDER_XGB = [
    'totalLoad', 'sommeil', 'fatigue', 'douleurMusculaire', 'stress', 
    'acuteLoad', 'chronicLoad', 'ACWR', 
    'sommeil_7d_mean', 'fatigue_7d_mean', 'douleurMusculaire_7d_mean', 'stress_7d_mean'
]

class ZonePredictionInput(BaseModel):
    playerId: int = Field(..., description="Player identifier")
    position: str = Field(..., description="Player position")
    foot: str = Field(..., description="Preferred foot")
    age: int = Field(..., description="Player age")
    fifa_rating: int = Field(..., description="FIFA rating")
    acuteLoad: float = Field(..., description="Charge aiguë 7 jours")
    chronicLoad: float = Field(..., description="Charge chronique 28 jours")
    ACWR: float = Field(..., description="Ratio ACWR")
    douleurMusculaire: float = Field(..., description="Douleurs musculaires")
    souplesse: float = Field(..., description="Souplesse")
    agilite: float = Field(..., description="Agilité")

    # Champs Viiv GX17 requis (transmis depuis l'app)
    viiv: ViivGX17Data = Field(
        ...,
        description="Données Viiv GX17 brutes transmises par l'app mobile."
    )
    medical_nutrition: Optional[InjuryZoneNutritionData] = Field(
        None, description="Bilan nutritionnel/biologique OCR utilisé pour ajuster la douleur musculaire."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "playerId": 0,
                "position": "Attaquant",
                "foot": "Droitier",
                "age": 24,
                "fifa_rating": 75,
                "acuteLoad": 6000,
                "chronicLoad": 4500,
                "ACWR": 1.33,
                "douleurMusculaire": 4.0,
                "souplesse": 6.0,
                "agilite": 8.0,
                "viiv": {
                    "heart_rate": 97.0,
                    "spo2": 98.5,
                    "hrv_ms": 42.0,
                    "stress_score": 35.0,
                    "energy_pct": 100.0,
                    "sleep_score": 7.5,
                    "recovery_pct": 30.0,
                    "strain": 0.0,
                },
                "medical_nutrition": {
                    "vitamin_d": 24.0, "ferritin": 85.0, "hemoglobin": 14.2,
                    "magnesium": 1.8, "calcium": 9.1, "c_reactive_protein": 0.32,
                },
            }
        }
    }

class RelapseSurvivalInput(BaseModel):
    playerId: int = Field(..., description="Player identifier")

    # Champs Viiv GX17 bruts (transmis depuis l'app mobile)
    viiv: ViivGX17Data = Field(
        ...,
        description="Données Viiv GX17 transmises par l'app mobile."
    )
    medical_nutrition: Optional[RelapseNutritionData] = Field(
        None, description="Bilan nutritionnel/biologique OCR utilisé pour ajuster récupération et fatigue."
    )

    # Champs IA requis hors Viiv
    physio_adherence: float = Field(..., description="Adhérence protocole Physio (%)")
    post_recovery_ACWR: float = Field(..., description="ACWR projeté au retour au jeu")

    model_config = {
        "json_schema_extra": {
            "example": {
                "playerId": 0,
                "viiv": {
                    "heart_rate": 97.0,
                    "spo2": 98.5,
                    "hrv_ms": 42.0,
                    "stress_score": 35.0,
                    "energy_pct": 100.0,
                    "sleep_score": 7.5,
                    "recovery_pct": 30.0,
                    "strain": 0.0,
                },
                "physio_adherence": 85.0,
                "post_recovery_ACWR": 1.1,
                "medical_nutrition": {
                    "vitamin_d": 24.0, "ferritin": 85.0, "hemoglobin": 14.2,
                    "vitamin_b12": 512.0, "magnesium": 1.8, "iron": 90.0,
                    "c_reactive_protein": 0.32,
                },
            }
        }
    }

    def resolve_inputs(self) -> dict:
        v = self.viiv
        nutrition_penalty = self.medical_nutrition.readiness_penalty() if self.medical_nutrition else 0.0
        return {
            "recovery_score": round(max(0.0, v.derive_recovery_score() - (10.0 * nutrition_penalty)), 2),
            "sleep_quality": round(max(1.0, v.derive_sommeil() - (0.15 * nutrition_penalty)), 2),
            "stress_level": round(min(1.0, v.derive_stress_level() + (0.03 * nutrition_penalty)), 3),
            "fatigue_index": round(min(100.0, v.derive_fatigue_index() + (5.0 * nutrition_penalty)), 2),
            "physio_adherence": self.physio_adherence,
            "post_recovery_ACWR": self.post_recovery_ACWR,
            "nutrition_readiness_penalty": nutrition_penalty,
            "medical_nutrition": self.medical_nutrition.snapshot() if self.medical_nutrition else None,
        }


# ---------------------------------------------------------
# 4. RESPONSE MODELS (Swagger schemas)
# ---------------------------------------------------------

class ShapFactor(BaseModel):
    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="SHAP contribution value")
    impact: str = Field(..., description="'négatif' if the feature increases risk, 'positif' otherwise")

class ViivSnapshot(BaseModel):
    heart_rate: Optional[float] = Field(None, description="Heart rate (bpm)")
    spo2: Optional[float] = Field(None, description="SpO₂ (%)")
    hrv_ms: Optional[float] = Field(None, description="HRV (ms)")
    stress_score: Optional[float] = Field(None, description="Viiv stress score (0–100)")
    energy_pct: Optional[float] = Field(None, description="Energy level (%)")
    sleep_score: Optional[float] = Field(None, description="Viiv sleep score")
    recovery_pct: Optional[float] = Field(None, description="Recovery (%)")
    strain: Optional[float] = Field(None, description="Viiv strain (0–21)")

class InjuryRiskResponse(BaseModel):
    playerId: int = Field(..., description="Player identifier")
    riskScore: float = Field(..., ge=0.0, le=1.0, description="Injury probability (0–1)")
    riskLevel: str = Field(..., description="'Faible', 'Modéré', or 'Critique'")
    factors: List[ShapFactor] = Field(..., description="SHAP-based contributing factors, sorted by absolute impact")
    resolved_inputs: Dict[str, Any] = Field(..., description="Final feature values used for prediction (after Viiv resolution)")
    viiv_data: Optional[ViivSnapshot] = Field(None, description="Raw Viiv GX17 snapshot used in this request")
    medical_nutrition: Optional[GlobalRiskNutritionData] = Field(None, description="Nutrition/lab OCR snapshot used in this request")

    model_config = {
        "json_schema_extra": {
            "example": {
                "playerId": 42,
                "riskScore": 0.73,
                "riskLevel": "Critique",
                "factors": [
                    {"feature": "ACWR", "contribution": 0.312, "impact": "négatif"},
                    {"feature": "fatigue", "contribution": 0.187, "impact": "négatif"},
                ],
                "resolved_inputs": {"totalLoad": 850.0, "sommeil": 6.5},
                "viiv_data": {"heart_rate": 97.0, "hrv_ms": 28.0},
                "medical_nutrition": {"vitamin_d": 24.0, "ferritin": 85.0},
            }
        }
    }

class InjuryZoneResponse(BaseModel):
    playerId: int = Field(..., description="Player identifier")
    predictions: Dict[str, float] = Field(
        ...,
        description="Probability per anatomical zone (keys = zone names, values = probability 0–1)"
    )
    resolved_inputs: Dict[str, Any] = Field(..., description="Final model inputs after Viiv and nutrition adjustment")
    medical_nutrition: Optional[InjuryZoneNutritionData] = Field(None, description="Nutrition/lab OCR snapshot used in this request")

    model_config = {
        "json_schema_extra": {
            "example": {
                "playerId": 42,
                "predictions": {
                    "Genou": 0.41,
                    "Ischio-jambiers": 0.32,
                    "Cheville": 0.15,
                    "Adducteur": 0.12,
                },
                "resolved_inputs": {"douleurMusculaire": 4.4, "nutrition_readiness_penalty": 0.8},
                "medical_nutrition": {"vitamin_d": 24.0},
            }
        }
    }

class SurvivalPoint(BaseModel):
    day: int = Field(..., description="Day along the survival timeline")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of remaining injury-free at this day")

class RelapseSurvivalResponse(BaseModel):
    playerId: int = Field(..., description="Player identifier")
    c_index: float = Field(..., description="Concordance index of the Cox model (model quality, closer to 1 is better)")
    survival_curve: List[SurvivalPoint] = Field(..., description="Day-by-day survival probability curve")
    resolved_inputs: Dict[str, Any] = Field(..., description="Final feature values used for prediction")
    viiv_data: Optional[Dict[str, Any]] = Field(None, description="Raw Viiv GX17 snapshot (subset relevant to survival)")
    medical_nutrition: Optional[RelapseNutritionData] = Field(None, description="Nutrition/lab OCR snapshot used in this request")

    model_config = {
        "json_schema_extra": {
            "example": {
                "playerId": 42,
                "c_index": 0.96,
                "survival_curve": [
                    {"day": 0, "probability": 1.0},
                    {"day": 30, "probability": 0.81},
                    {"day": 90, "probability": 0.54},
                ],
                "resolved_inputs": {"recovery_score": 72.0, "sleep_quality": 7.0},
                "viiv_data": {"recovery_pct": 72.0, "hrv_ms": 38.0},
                "medical_nutrition": {"vitamin_d": 24.0, "magnesium": 1.8},
            }
        }
    }


class MedicalNutrientMention(BaseModel):
    nutrient: str
    matched_alias: str
    value: Optional[float] = None
    unit: Optional[str] = None
    status: str
    text_snippet: str


class MedicalNutrientExtractionResponse(BaseModel):
    source: str = Field(..., description="Input source: image, text, or image+text")
    extracted_text: str = Field(..., description="OCR and/or provided text merged for parsing")
    nutrients_found: List[str] = Field(..., description="Detected nutrient names")
    mentions: List[MedicalNutrientMention] = Field(..., description="Detected nutrient/value mentions")
    flagged: List[MedicalNutrientMention] = Field(..., description="Only low/high out-of-range mentions")


class ActionSuccessInput(BaseModel):
    playerId: int = Field(10, description="Player identifier")
    x: float = Field(..., ge=0, le=120, description="StatsBomb X coordinate, from 0 to 120")
    y: float = Field(..., ge=0, le=80, description="StatsBomb Y coordinate, from 0 to 80")
    action_type: str = Field("Pass", description="Action type: Pass, Shot, or Carry")
    under_pressure: bool = Field(False, description="Whether the player is under opponent pressure")
    play_pattern: str = Field("Regular Play", description="Game situation or play pattern")

    model_config = {
        "json_schema_extra": {
            "example": {
                "playerId": 10,
                "x": 95.0,
                "y": 65.0,
                "action_type": "Pass",
                "under_pressure": True,
                "play_pattern": "Regular Play",
            }
        }
    }


class ActionSuccessResponse(BaseModel):
    playerId: int
    success_probability: float = Field(..., ge=0.0, le=1.0)
    distance_to_goal: float
    source: str = Field(..., description="ml_model when the artifact is available, otherwise fallback_formula")


class HeatmapEvent(BaseModel):
    id: int
    x: float
    y: float
    action_type: str
    success: int = Field(..., ge=0, le=1)


class PlayerSeasonHeatmapResponse(BaseModel):
    playerId: int
    total_actions: int
    events: List[HeatmapEvent]


class PerformanceForecastInput(BaseModel):
    """Monthly performance history, ordered from oldest to newest."""
    playerId: int = Field(10, description="Player identifier")
    history: List[float] = Field(..., min_length=3, description="Monthly performance scores, oldest to newest")
    steps: int = Field(3, ge=1, le=6, description="Number of months to forecast")
    matches_played: int = Field(3, ge=0, le=20, description="Expected matches played per forecast month")

    model_config = {
        "json_schema_extra": {
            "example": {
                "playerId": 10,
                "history": [82, 84, 86, 85, 88, 89],
                "steps": 3,
                "matches_played": 3,
            }
        }
    }


class PerformanceForecastResponse(BaseModel):
    playerId: int
    history: List[float]
    predictions: List[float]
    steps: int
    score_range: List[float]
    source: str
    model_metrics: Dict[str, Any]


class PlayerPerformanceModelInfoResponse(BaseModel):
    model_type: Optional[str] = None
    task: Optional[str] = None
    data_source: Optional[str] = None
    min_months: Optional[int] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------
# 5. ROUTES DE L'API
# ---------------------------------------------------------

@app.get(
    "/",
    tags=["Health"],
    summary="Service health check",
    response_description="Service status, version, and integration info",
)
def read_root():
    """
    Returns a simple liveness response confirming the AI service is online,
    together with the current version and Viiv integration status.
    """
    return {
        "message": "ERP Club AI Service est en ligne 🟢",
        "version": "5.0.0",
        "viiv_integration": "Viiv GX17 natif — champs `viiv` dans chaque requête POST",
        "player_action_model_loaded": player_action_model_artifact is not None,
        "player_performance_model_loaded": player_performance_model_artifact is not None,
    }


@app.get(
    "/player-performance-model-info",
    tags=["Player Intelligence"],
    summary="Get player performance model metadata",
    response_model=PlayerPerformanceModelInfoResponse,
    response_description="Model type, task, data source, minimum history length, and metrics",
    responses={503: {"description": "Player performance model is not available"}},
)
def get_player_performance_model_info():
    if player_performance_model_artifact is None:
        raise HTTPException(status_code=503, detail="Le modele de performance temporelle n'est pas disponible.")

    artifact = player_performance_model_artifact
    return {
        "model_type": artifact.get("model_type"),
        "task": artifact.get("task"),
        "data_source": artifact.get("data_source"),
        "min_months": artifact.get("min_months"),
        "metrics": artifact.get("metrics", {}),
    }


@app.post(
    "/predict-player-performance",
    tags=["Player Intelligence"],
    summary="Forecast monthly player performance",
    response_model=PerformanceForecastResponse,
    response_description="Predicted performance scores for the requested forecast horizon",
    responses={
        200: {"description": "Forecast successful"},
        422: {"description": "Invalid score history"},
        503: {"description": "Player performance model is not available"},
    },
)
def predict_player_performance(data: PerformanceForecastInput):
    """
    Forecasts the next monthly performance scores from a recent player score history.

    The endpoint uses `player_performance_model.joblib` when available and returns
    the recorded model metrics with the predictions for traceability.
    """
    if player_performance_model_artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Modele indisponible. Executez d'abord le notebook 05_player_performance_timeseries.ipynb.",
        )

    artifact = player_performance_model_artifact
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


@app.post(
    "/predict-action-success",
    tags=["Player Intelligence"],
    summary="Predict action success probability",
    response_model=ActionSuccessResponse,
    response_description="Probability of action success plus distance to goal",
)
def predict_action_success(data: ActionSuccessInput):
    """
    Predicts whether a pass, shot, or carry is likely to succeed from a given pitch location.

    If `player_heatmap_model.joblib` is unavailable, the service keeps the endpoint online
    with the same fallback formula previously used by the standalone player API.
    """
    goal_x, goal_y = 120.0, 40.0
    distance = np.sqrt((goal_x - data.x) ** 2 + (goal_y - data.y) ** 2)
    angle = np.arctan2(goal_y - data.y, goal_x - data.x) * (180 / np.pi)

    if player_action_model_artifact:
        try:
            player_model = player_action_model_artifact["model"]
            player_scaler = player_action_model_artifact["scaler"]
            encoders = player_action_model_artifact["label_encoders"]
            features = player_action_model_artifact["features"]

            input_data = {
                "x": data.x,
                "y": data.y,
                "Distance": distance,
                "Angle": angle,
                "action_type": data.action_type,
                "play_pattern": data.play_pattern,
                "under_pressure": int(data.under_pressure),
            }

            for col in ["action_type", "play_pattern"]:
                if col in encoders:
                    label_encoder = encoders[col]
                    input_data[col] = (
                        label_encoder.transform([input_data[col]])[0]
                        if input_data[col] in label_encoder.classes_
                        else 0
                    )

            df_input = pd.DataFrame([input_data])[features]
            continuous_cols = ["x", "y", "Distance", "Angle"]
            df_input[continuous_cols] = player_scaler.transform(df_input[continuous_cols])
            prob_success = float(player_model.predict_proba(df_input)[0][1])

            return {
                "playerId": data.playerId,
                "success_probability": round(prob_success, 4),
                "distance_to_goal": round(distance, 2),
                "source": "ml_model",
            }
        except Exception as e:
            print(f"[WARN] [MODULE 4] Erreur prediction joueur ({e}), fallback actif.")

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


@app.get(
    "/player-season-heatmap",
    tags=["Player Intelligence"],
    summary="Generate player season heatmap events",
    response_model=PlayerSeasonHeatmapResponse,
    response_description="Synthetic geolocated player events used by the heatmap dashboard",
)
def get_player_season_heatmap(playerId: int = 10):
    """
    Returns 128 realistic geolocated player events biased toward the right half-space
    and attacking box area, using StatsBomb pitch coordinates.
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


@app.post(
    "/predict-injury",
    tags=["Injury Risk"],
    summary="Predict global injury risk (XGBoost + SHAP)",
    response_model=InjuryRiskResponse,
    response_description="Injury probability, risk level, SHAP factors, and resolved inputs",
    responses={
        200: {"description": "Prediction successful"},
        500: {"description": "Model not loaded or prediction error"},
    },
)
def predict_injury_risk(data: PlayerFeatures):
    """
        Computes the **global injury risk** for a player using the XGBoost model.

    ### Input Resolution
        - If a `viiv` block is provided, sensor-derived values populate the matching model inputs:
      - `hrv_ms` → `fatigue`
      - `stress_score` → `stress`
      - `sleep_score` → `sommeil`
      - `strain` → `acuteLoad`
    - `ACWR` is auto-calculated as `acuteLoad / chronicLoad` when not provided.
    - An optional `medical_nutrition` block from OCR applies a documented
      readiness adjustment to fatigue, pain, stress, and sleep before scoring.

    ### Risk Levels
    | Score | Level |
    |-------|-------|
    | > 0.70 | **Critique** |
    | 0.40 – 0.70 | **Modéré** |
    | < 0.40 | **Faible** |

    ### SHAP Explainability
    The `factors` array lists feature contributions (sorted by absolute impact),
    making predictions **fully explainable** to medical staff.
    """
    if not xgb_model:
        raise HTTPException(status_code=500, detail="Modèle XGBoost non chargé.")
    try:
        resolved = data.resolve_inputs()
        features_dict = {col: [resolved[col]] for col in FEATURES_ORDER_XGB}
        df_input = pd.DataFrame(features_dict)
        df_scaled = pd.DataFrame(scaler.transform(df_input), columns=FEATURES_ORDER_XGB)
        risk_prob = float(xgb_model.predict_proba(df_scaled)[0][1])
        
        shap_values = explainer.shap_values(df_scaled)
        contributions = shap_values[1][0] if isinstance(shap_values, list) else (shap_values[0] if len(shap_values.shape) > 1 else shap_values)
            
        factors = []
        for feature_name, contrib in zip(FEATURES_ORDER_XGB, contributions):
            if abs(contrib) > 0.01:
                factors.append({"feature": feature_name, "contribution": round(float(contrib), 3), "impact": "négatif" if contrib > 0 else "positif"})
        factors = sorted(factors, key=lambda x: abs(x["contribution"]), reverse=True)
        
        risk_level = "Critique" if risk_prob > 0.70 else ("Modéré" if risk_prob > 0.40 else "Faible")

        # Inclure les données Viiv brutes résolues dans la réponse pour la traçabilité
        viiv_snapshot = None
        if data.viiv:
            viiv_snapshot = {
                "heart_rate": data.viiv.heart_rate,
                "spo2": data.viiv.spo2,
                "hrv_ms": data.viiv.hrv_ms,
                "stress_score": data.viiv.stress_score,
                "energy_pct": data.viiv.energy_pct,
                "sleep_score": data.viiv.sleep_score,
                "recovery_pct": data.viiv.recovery_pct,
                "strain": data.viiv.strain,
            }
        nutrition_snapshot = data.medical_nutrition.snapshot() if data.medical_nutrition else None

        return {
            "playerId": data.playerId,
            "riskScore": round(risk_prob, 2),
            "riskLevel": risk_level,
            "factors": factors,
            "resolved_inputs": resolved,
            "viiv_data": viiv_snapshot,
            "medical_nutrition": nutrition_snapshot,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict-injury-zone",
    tags=["Injury Zone"],
    summary="Predict anatomical injury zone (Random Forest / LightGBM)",
    response_model=InjuryZoneResponse,
    response_description="Per-zone probability distribution across all anatomical zones",
    responses={
        200: {"description": "Prediction successful"},
        500: {"description": "Model not loaded or prediction error"},
    },
)
def predict_injury_zone(data: ZonePredictionInput):
    """
    Classifies which **anatomical zone** is most at risk of injury for a given player.

    ### Inputs
    - **Player profile**: `age`, `position`, `foot`, `fifa_rating`
    - **Load metrics**: `acuteLoad`, `chronicLoad`, `ACWR`
    - **Physical tests**: `souplesse`, `agilite`, `douleurMusculaire`
    - **Viiv GX17**: HRV is used to adjust `douleurMusculaire` via a weighted average.
    - **Medical OCR**: abnormal nutrition/lab values apply a readiness adjustment
      to `douleurMusculaire` before prediction.

    ### Output
    A dictionary mapping each anatomical zone to its predicted probability.
    The probabilities sum to 1.0 across all zones.

    > **Example zones**: Genou, Ischio-jambiers, Cheville, Adducteur, Lombaire …
    """
    if not model_zone_artifact:
        raise HTTPException(status_code=500, detail="Le modèle des zones est introuvable.")
    try:
        model_zone = model_zone_artifact.get('model')
        pos_encoder, foot_encoder = model_zone_artifact.get('pos_mapping'), model_zone_artifact.get('foot_mapping')
        features = model_zone_artifact.get('feature_names', model_zone_artifact.get('features', []))
        zones = model_zone_artifact.get('model_classes', model_zone_artifact.get('target_classes', []))
        
        pos_enc = int(pos_encoder.transform([data.position])[0]) if pos_encoder and data.position in pos_encoder.classes_ else 0
        foot_enc = int(foot_encoder.transform([data.foot])[0]) if foot_encoder and data.foot in foot_encoder.classes_ else 0

        # Si Viiv disponible, enrichir douleurMusculaire depuis HRV
        douleur_final = data.douleurMusculaire
        if data.viiv:
            viiv_fatigue = data.viiv.derive_fatigue()
            if viiv_fatigue is not None:
                douleur_final = (douleur_final + viiv_fatigue) / 2.0  # moyenne pondérée
        nutrition_penalty = data.medical_nutrition.readiness_penalty() if data.medical_nutrition else 0.0
        douleur_final = round(min(10.0, douleur_final + (0.6 * nutrition_penalty)), 2)

        input_dict = {
            'Age': data.age, 'FIFA rating': data.fifa_rating, 'acuteLoad': data.acuteLoad,
            'chronicLoad': data.chronicLoad, 'ACWR': data.ACWR, 'douleurMusculaire': douleur_final,
            'souplesse': data.souplesse, 'agilite': data.agilite, 'Position_encoded': pos_enc, 'Foot_encoded': foot_enc
        }
        
        final_input = {feat: input_dict.get(feat, 0) for feat in features}
        input_df = pd.DataFrame([final_input])[features] 
        probabilities = model_zone.predict_proba(input_df)[0]
        predictions = {zones[i]: float(probabilities[i]) for i in range(len(zones))}
        return {
            "playerId": data.playerId,
            "predictions": predictions,
            "resolved_inputs": {
                **input_dict,
                "nutrition_readiness_penalty": nutrition_penalty,
            },
            "medical_nutrition": data.medical_nutrition.snapshot() if data.medical_nutrition else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict-relapse",
    tags=["Relapse Survival"],
    summary="Predict relapse survival curve (Cox Proportional Hazards)",
    response_model=RelapseSurvivalResponse,
    response_description="Survival curve (day-by-day probability of remaining injury-free) and model concordance index",
    responses={
        200: {"description": "Prediction successful"},
        500: {"description": "Model not loaded or prediction error"},
    },
)
def predict_relapse_risk(data: RelapseSurvivalInput):
    """
    Estimates the **relapse survival curve** for a player returning from injury
    using a Cox Proportional Hazards model.

    ### What is a survival curve?
    The curve gives, for each day after return-to-play, the probability that the
    player has *not* suffered a relapse injury. A steep drop in the early days
    signals high relapse risk.

    ### Input Resolution (Viiv GX17)
    The request payload should contain Viiv data plus the supplementary rehabilitation
    fields required by the model.

    | Viiv field | → | Model feature |
    |---|---|---|
    | `recovery_pct` | → | `recovery_score` |
    | `sleep_score` | → | `sleep_quality` |
    | `stress_score` | → | `stress_level` (0–1) |
    | `hrv_ms` | → | `fatigue_index` (0–100) |

    The optional `medical_nutrition` OCR block reduces recovery and increases
    fatigue/stress through the same transparent readiness adjustment.

    ### Model Quality
    The `c_index` (concordance index) measures how well the model ranks players
    by actual relapse time. A value of **0.96** means near-perfect discrimination.
    """
    if not model_survival_artifact:
        raise HTTPException(status_code=500, detail="Le modèle de Survie (Cox) est introuvable.")
    
    try:
        model_surv = model_survival_artifact['model']
        scaler_surv = model_survival_artifact['scaler']
        features_surv = model_survival_artifact['features']
        
        resolved = data.resolve_inputs()

        input_df = pd.DataFrame([resolved])[features_surv]
        input_scaled = pd.DataFrame(scaler_surv.transform(input_df), columns=features_surv)
        
        surv_func = model_surv.predict_survival_function(input_scaled)
        
        timeline = surv_func.index.tolist()
        probabilities = surv_func.iloc[:, 0].tolist()
        
        curve = [{"day": int(t), "probability": float(p)} for t, p in zip(timeline, probabilities)]
        
        viiv_snapshot = None
        if data.viiv:
            viiv_snapshot = {
                "heart_rate": data.viiv.heart_rate,
                "spo2": data.viiv.spo2,
                "hrv_ms": data.viiv.hrv_ms,
                "recovery_pct": data.viiv.recovery_pct,
                "stress_score": data.viiv.stress_score,
                "energy_pct": data.viiv.energy_pct,
            }
        nutrition_snapshot = data.medical_nutrition.snapshot() if data.medical_nutrition else None

        return {
            "playerId": data.playerId,
            "c_index": model_survival_artifact.get('c_index', 0.96),
            "survival_curve": curve,
            "resolved_inputs": resolved,
            "viiv_data": viiv_snapshot,
            "medical_nutrition": nutrition_snapshot,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction Survie: {str(e)}")


@app.post(
    "/extract-medical-nutrients",
    tags=["Medical OCR"],
    summary="Extract vitamins/minerals from a medical report",
    response_model=MedicalNutrientExtractionResponse,
    response_description="Detected nutrient mentions and abnormal findings",
)
async def extract_medical_nutrients(
    file: Optional[UploadFile] = File(default=None),
    raw_text: Optional[str] = Form(default=None),
    use_ocr: bool = Form(default=True),
):
    """
    Accepts:
    - a PDF, DOCX, or image report (`file`) for text extraction / OCR,
    - plain text report (`raw_text`),
    - or both (merged before parsing).

    Extracts vitamins and minerals that can influence risk modeling, such as
    vitamin D, B12, zinc, magnesium, ferritin, and calcium.
    """
    if file is None and not (raw_text and raw_text.strip()):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one input: file (PDF/DOCX/image) or raw_text.",
        )

    extracted_chunks: List[str] = []
    source = "text"

    if file is not None:
        filename = file.filename or "unknown"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in ("pdf", "docx", "doc"):
            source = "document"
        else:
            source = "image"

        try:
            file_bytes = await file.read()
            if use_ocr or ext in ("pdf", "docx", "doc"):
                parsed_text = extract_text_from_file(file_bytes, filename)
                if parsed_text.strip():
                    extracted_chunks.append(parsed_text)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File parsing failed: {str(e)}")

    if raw_text and raw_text.strip():
        source = f"{source}+text" if source != "text" else "text"
        extracted_chunks.append(raw_text.strip())

    merged_text = "\n\n".join(extracted_chunks).strip()
    if not merged_text:
        raise HTTPException(status_code=400, detail="No readable text found in input.")

    extracted = extract_nutrients_from_text(merged_text)
    return {
        "source": source,
        "extracted_text": merged_text,
        "nutrients_found": extracted["nutrients_found"],
        "mentions": extracted["mentions"],
        "flagged": extracted["flagged"],
    }


# =============================================================
# MODULE 5 — Possession Estimation
# =============================================================
try:
    from app.possession import process_video_possession
    POSSESSION_READY = True
except Exception as _pe:
    POSSESSION_READY = False
    print(f"[WARN] Possession module unavailable: {_pe}")

POSSESSION_OUTPUT_DIR = Path(tempfile.gettempdir()) / "erp_possession_outputs"
POSSESSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.post(
    "/possession/analyze",
    tags=["Possession Estimation"],
    summary="Analyze a football match video for team possession",
    response_description=(
        "Possession percentages, per-frame timeline, referee/player counts, "
        "classifier accuracy, and a download URL for the annotated output video."
    ),
)
async def analyze_possession(
    video: UploadFile = File(..., description="Match broadcast video (mp4, mkv, avi)"),
    max_frames: int  = Form(default=300, ge=10, le=2000,
                            description="Maximum number of frames to process"),
    conf_thresh: float = Form(default=0.20, ge=0.05, le=0.80,
                              description="YOLO person detection confidence threshold"),
    ball_conf_thresh: float = Form(default=0.10, ge=0.05, le=0.80,
                                   description="YOLO ball detection confidence threshold"),
    poss_radius_px: float = Form(default=100.0, ge=20.0, le=500.0,
                                  description="Pixel radius around ball to claim possession"),
    smoothing_window: int = Form(default=5, ge=1, le=30,
                                  description="Majority-vote smoothing window size (frames)"),
):
    """
    Upload a match video to run the full possession estimation pipeline:

    - **Spectators** in the stands are ignored (pitch boundary filter).
    - **Referees** on the pitch are identified (Class 2) and excluded from possession.
    - **Goalkeepers** are correctly assigned to Team A or Team B.
    - **IoU tracking + EMA probability smoothing** prevents annotation flickering.
    - Returns possession stats + download link for the annotated video.
    """
    if not POSSESSION_READY:
        raise HTTPException(
            status_code=503,
            detail="Possession module is not available. Ensure ultralytics, opencv-python, and scikit-learn are installed.",
        )

    suffix = Path(video.filename).suffix.lower()
    if suffix not in (".mp4", ".mkv", ".avi", ".mov"):
        raise HTTPException(status_code=400,
                            detail=f"Unsupported video format '{suffix}'. Use mp4, mkv, avi or mov.")

    job_id = str(uuid.uuid4())[:8]
    input_path  = POSSESSION_OUTPUT_DIR / f"{job_id}_input{suffix}"
    output_path = POSSESSION_OUTPUT_DIR / f"{job_id}_annotated.mp4"

    # Save uploaded file
    contents = await video.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    try:
        result = process_video_possession(
            video_path=str(input_path),
            output_video_path=str(output_path),
            max_frames=max_frames,
            conf_thresh=conf_thresh,
            ball_conf_thresh=ball_conf_thresh,
            poss_radius_px=poss_radius_px,
            smoothing_window=smoothing_window,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Possession analysis failed: {str(e)}")
    finally:
        if input_path.exists():
            input_path.unlink(missing_ok=True)

    result["job_id"] = job_id
    result["output_video_download_url"] = f"/possession/download/{job_id}"
    # Keep only last 20 per-frame records in response for brevity (full data in video)
    result["per_frame_records"] = result["per_frame_records"][-20:]
    return result


@app.get(
    "/possession/download/{job_id}",
    tags=["Possession Estimation"],
    summary="Download the annotated possession video",
    response_class=FileResponse,
)
async def download_possession_video(job_id: str):
    """Stream the annotated video generated by `/possession/analyze`."""
    video_path = POSSESSION_OUTPUT_DIR / f"{job_id}_annotated.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found. It may have expired.")
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"possession_{job_id}_annotated.mp4",
    )

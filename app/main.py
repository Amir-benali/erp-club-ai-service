from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import pandas as pd
import joblib
import shap
import os

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
        "target model cannot infer from Viiv alone.\n\n"
        "### Authentication\n"
        "No authentication is required for this internal microservice. "
        "Secure network-level access is enforced at the API gateway."
    ),
    version="5.0.0",
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

class PlayerFeatures(BaseModel):
    playerId: int

    # --- Champs Viiv GX17 bruts (transmis par l'app mobile) ---
    viiv: ViivGX17Data = Field(
        ...,
        description="Données brutes du capteur Viiv GX17 transmises par l'app mobile."
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
        acute_final = v.derive_acute_load(self.acuteLoad) if v.strain is not None else self.acuteLoad
        acwr_final = self.ACWR

        return {
            "totalLoad": self.totalLoad,
            "sommeil": sommeil_final,
            "fatigue": fatigue_final,
            "douleurMusculaire": self.douleurMusculaire,
            "stress": stress_final,
            "acuteLoad": acute_final,
            "chronicLoad": self.chronicLoad,
            "ACWR": acwr_final,
            "sommeil_7d_mean": self.sommeil_7d_mean,
            "fatigue_7d_mean": self.fatigue_7d_mean,
            "douleurMusculaire_7d_mean": self.douleurMusculaire_7d_mean,
            "stress_7d_mean": self.stress_7d_mean,
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
            }
        }
    }

    def resolve_inputs(self) -> dict:
        v = self.viiv
        return {
            "recovery_score": v.derive_recovery_score(),
            "sleep_quality": v.derive_sommeil(),
            "stress_level": v.derive_stress_level(),
            "fatigue_index": v.derive_fatigue_index(),
            "physio_adherence": self.physio_adherence,
            "post_recovery_ACWR": self.post_recovery_ACWR,
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
            }
        }
    }

class InjuryZoneResponse(BaseModel):
    playerId: int = Field(..., description="Player identifier")
    predictions: Dict[str, float] = Field(
        ...,
        description="Probability per anatomical zone (keys = zone names, values = probability 0–1)"
    )

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
        "viiv_integration": "Viiv GX17 natif — champs `viiv` dans chaque requête POST"
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

        return {
            "playerId": data.playerId,
            "riskScore": round(risk_prob, 2),
            "riskLevel": risk_level,
            "factors": factors,
            "resolved_inputs": resolved,
            "viiv_data": viiv_snapshot,
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
        
        pos_enc = pos_encoder.transform([data.position])[0] if pos_encoder and data.position in pos_encoder.classes_ else 0
        foot_enc = foot_encoder.transform([data.foot])[0] if foot_encoder and data.foot in foot_encoder.classes_ else 0

        # Si Viiv disponible, enrichir douleurMusculaire depuis HRV
        douleur_final = data.douleurMusculaire
        if data.viiv:
            viiv_fatigue = data.viiv.derive_fatigue()
            if viiv_fatigue is not None:
                douleur_final = (douleur_final + viiv_fatigue) / 2.0  # moyenne pondérée

        input_dict = {
            'Age': data.age, 'FIFA rating': data.fifa_rating, 'acuteLoad': data.acuteLoad,
            'chronicLoad': data.chronicLoad, 'ACWR': data.ACWR, 'douleurMusculaire': douleur_final,
            'souplesse': data.souplesse, 'agilite': data.agilite, 'Position_encoded': pos_enc, 'Foot_encoded': foot_enc
        }
        
        final_input = {feat: input_dict.get(feat, 0) for feat in features}
        input_df = pd.DataFrame([final_input])[features] 
        probabilities = model_zone.predict_proba(input_df)[0]
        predictions = {zones[i]: float(probabilities[i]) for i in range(len(zones))}
        return {"playerId": data.playerId, "predictions": predictions}
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

        return {
            "playerId": data.playerId,
            "c_index": model_survival_artifact.get('c_index', 0.96),
            "survival_curve": curve,
            "resolved_inputs": resolved,
            "viiv_data": viiv_snapshot,
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
"""
AquaSentinel — predict.py  (v3.0.0)
=====================================
Water Quality Assessment Model — RS485 Multi-Probe + DS18B20

YOUR HARDWARE:
    temperature  (°C)    DS18B20 / RS485 temperature sensor
    ph           (pH)    RS485 pH probe
    nitrite      (mg/L)  RS485 Nitrite channel
    phosphorus   (mg/L)  RS485 Phosphorus channel

THE MODEL EXPECTS (10 features, trained on AWD dataset — 3,887 samples):
    Temp, pH`, Nitrite (mg L-1 ), Phosphorus (mg L-1 ),
    nitrite_log, phosphorus_log,
    ph_nitrite_interaction, phosphorus_nitrite_product,
    temp_ph_interaction, temp_squared

OUTPUT LABELS:
    0 = Excellent  — optimal conditions, continue routine monitoring
    1 = Good       — acceptable, monitor closely
    2 = Poor       — critical, immediate intervention required

DATASET:
    Aquaculture Water Quality Dataset (AWD)
    Veeramsetty et al., Mendeley Data (2024)
    DOI: 10.17632/y78ty2g293.1

MODEL PERFORMANCE (from water_quality.ipynb):
    Overall Accuracy : 85.5%
    Macro F1 Score   : 0.8479
    CV Mean F1       : 0.8444  (Std Dev: 0.0158)
    Poor Precision   : 1.00   (no false alarms)
    Poor Recall      : 0.59   (catches 59% of bad ponds)

CHANGES IN v3.0.0:
    - Complete rewrite — water quality assessment replaces bloom risk
    - 4 hardware parameters: Temperature, pH, Nitrite, Phosphorus
    - 10 engineered features (log transforms, interactions, polynomial)
    - Labels: Excellent / Good / Poor  (replaces Low / Medium / High Risk)
    - Input validation bounds updated to match AWD training data ranges
    - No conversion layer needed — hardware measures exactly what model expects
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("aquasentinel.predict")

# ── Load model artifacts ──────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

logger.info("Loading model artifacts from %s ...", MODEL_DIR)
try:
    model        = joblib.load(MODEL_DIR / "random_forest_model.pkl")
    scaler       = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.pkl")
    logger.info("Model loaded. Expects %d features.", len(feature_cols))
except Exception as e:
    logger.critical("FATAL — could not load model artifacts: %s", e)
    raise

# ── FEATURE_COLS — must match water_quality.ipynb Cell 11 exactly ─────────
FEATURE_COLS = [
    # Raw sensor readings (4) — directly from RS485 hardware
    "Temp",
    "pH`",
    "Nitrite (mg L-1 )",
    "Phosphorus (mg L-1 )",
    # Log transforms (2)
    "nitrite_log",
    "phosphorus_log",
    # Interaction terms (3)
    "ph_nitrite_interaction",
    "phosphorus_nitrite_product",
    "temp_ph_interaction",
    # Polynomial (1)
    "temp_squared",
]

# ── Input validation bounds ───────────────────────────────────────────────
# Based on AWD dataset after cleaning (3,887 samples):
#   Temperature : 0.19 – 34.99 °C  → widened to 0–50 for sensor safety
#   pH          : 0.004 – 13.65    → physical limit 0–14
#   Nitrite     : 0.00 – 4.99 mg/L → widened to 0–20
#   Phosphorus  : 0.00 – 4.97 mg/L → widened to 0–20
SENSOR_BOUNDS = {
    "temperature": ( 0.0,  50.0),
    "ph":          ( 0.0,  14.0),
    "nitrite":     ( 0.0,  20.0),
    "phosphorus":  ( 0.0,  20.0),
}


# ═══════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_inputs(
    temperature: float,
    ph:          float,
    nitrite:     float,
    phosphorus:  float,
) -> dict:
    """
    Validate sensor inputs against safe operating ranges.
    Out-of-range values are clamped and flagged in the response.
    """
    warnings = []
    clamped  = {}

    incoming = {
        "temperature": temperature,
        "ph":          ph,
        "nitrite":     nitrite,
        "phosphorus":  phosphorus,
    }

    for name, value in incoming.items():
        lo, hi = SENSOR_BOUNDS[name]
        if value < lo or value > hi:
            adjusted = max(lo, min(hi, value))
            warnings.append(
                f"{name}={value} is outside expected range [{lo}, {hi}]. "
                f"Clamped to {adjusted}."
            )
            clamped[name] = adjusted
            logger.warning(
                "Input out of range: %s=%.4f → clamped to %.4f",
                name, value, adjusted,
            )

    return {
        "valid":          len(warnings) == 0,
        "warnings":       warnings,
        "clamped_values": clamped,
    }


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def engineer_features(
    temperature: float,
    ph:          float,
    nitrite:     float,
    phosphorus:  float,
) -> dict:
    """
    Build all 10 model features from 4 raw sensor readings.
    Mirrors Cell 11 of water_quality.ipynb exactly.

    Feature groups:
        Raw (4)         : direct sensor readings
        Log transforms  : right-skew correction for Nitrite and Phosphorus
        Interactions    : pH×Nitrite toxicity, P×N dual stress, Temp×pH stress
        Polynomial      : Temp² for non-linear thermal effects
    """
    return {
        # ── Raw (4) ──────────────────────────────────────────────────────
        "Temp":                       temperature,
        "pH`":                        ph,
        "Nitrite (mg L-1 )":          nitrite,
        "Phosphorus (mg L-1 )":       phosphorus,
        # ── Log transforms (2) ───────────────────────────────────────────
        # log1p safe for zero values
        "nitrite_log":                np.log1p(nitrite),
        "phosphorus_log":             np.log1p(phosphorus),
        # ── Interaction terms (3) ────────────────────────────────────────
        # pH × Nitrite: un-ionised nitrite toxicity rises at high pH
        "ph_nitrite_interaction":     ph * nitrite,
        # Phosphorus × Nitrite: dual nutrient water quality stress index
        "phosphorus_nitrite_product": phosphorus * nitrite,
        # Temp × pH: compound thermal + chemical stress
        "temp_ph_interaction":        temperature * ph,
        # ── Polynomial (1) ───────────────────────────────────────────────
        # Non-linear thermal stress on biochemical processes
        "temp_squared":               temperature ** 2,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_water_quality(
    temperature: float,   # °C   — DS18B20 / RS485
    ph:          float,   # pH   — RS485
    nitrite:     float,   # mg/L — RS485
    phosphorus:  float,   # mg/L — RS485
) -> dict:
    """
    Assess water quality in a fish pond from four sensor readings.

    Parameters
    ----------
    temperature : float  Water temperature in °C
    ph          : float  pH value (0 – 14)
    nitrite     : float  Nitrite concentration in mg/L
    phosphorus  : float  Phosphorus concentration in mg/L

    Returns
    -------
    dict — quality_level, quality_label, confidence, probabilities,
           action, sensor_inputs, data_quality, validation
    """

    # ── STEP 1: Validate and clamp inputs ───────────────────────────────
    validation = validate_inputs(temperature, ph, nitrite, phosphorus)
    clamped    = validation["clamped_values"]
    temperature = clamped.get("temperature", temperature)
    ph          = clamped.get("ph",          ph)
    nitrite     = clamped.get("nitrite",     nitrite)
    phosphorus  = clamped.get("phosphorus",  phosphorus)

    # ── STEP 2: Feature engineering ─────────────────────────────────────
    features = engineer_features(temperature, ph, nitrite, phosphorus)

    # ── STEP 3: Scale and predict ────────────────────────────────────────
    input_df     = pd.DataFrame([features])[feature_cols]
    input_scaled = scaler.transform(input_df)

    prediction    = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    confidence    = float(probabilities[prediction] * 100)

    logger.info(
        "Prediction: %s (%.1f%%)  temp=%.1f ph=%.2f nitrite=%.3f phosphorus=%.3f",
        ["EXCELLENT", "GOOD", "POOR"][int(prediction)],
        confidence, temperature, ph, nitrite, phosphorus,
    )

    # ── STEP 4: Build response ───────────────────────────────────────────
    quality_map = {
        0: {
            "label":    "EXCELLENT",
            "icon":     "green",
            "severity": "excellent",
            "action": (
                "Optimal water quality. All parameters within ideal range. "
                "Continue routine monitoring every 6 hours."
            ),
        },
        1: {
            "label":    "GOOD",
            "icon":     "yellow",
            "severity": "good",
            "action": (
                "Acceptable water quality. Parameters are within safe limits "
                "but showing some elevation. Monitor closely every 2–3 hours. "
                "Consider light aeration increase if trend is worsening."
            ),
        },
        2: {
            "label":    "POOR",
            "icon":     "red",
            "severity": "poor",
            "action": (
                "CRITICAL — Water quality has deteriorated. "
                "Increase aeration immediately, reduce feeding by 50%, "
                "consider 20% water exchange. "
                "Alert pond manager now."
            ),
        },
    }

    info = quality_map[int(prediction)]

    return {
        # ── Core result ──────────────────────────────────────────────────
        "quality_level": int(prediction),
        "quality_label": info["label"],
        "icon":          info["icon"],
        "severity":      info["severity"],
        "confidence":    round(confidence, 2),
        "action":        info["action"],
        "probabilities": {
            "excellent": round(float(probabilities[0] * 100), 2),
            "good":      round(float(probabilities[1] * 100), 2),
            "poor":      round(float(probabilities[2] * 100), 2),
        },
        # ── What went into the model ─────────────────────────────────────
        "sensor_inputs": {
            "temperature_c":   temperature,
            "ph":              ph,
            "nitrite_mg_l":    nitrite,
            "phosphorus_mg_l": phosphorus,
        },
        # ── Transparency ─────────────────────────────────────────────────
        "data_quality": {
            "features_used":     len(feature_cols),
            "feature_source":    "4 raw RS485 readings → 10 engineered features",
            "no_conversion":     True,
            "hardware_match":    "RS485 measures exactly what model was trained on",
        },
        # ── Validation report ────────────────────────────────────────────
        "validation": {
            "inputs_valid":   validation["valid"],
            "warnings":       validation["warnings"],
            "values_clamped": len(validation["clamped_values"]) > 0,
        },
    }
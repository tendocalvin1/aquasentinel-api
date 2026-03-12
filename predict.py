"""
AquaSentinel — predict.py  (v2.1.0)
=====================================
Hardware-to-Model conversion layer.

YOUR HARDWARE (RS485 Multi-Probe + DS18B20 + GPIO):
    total_nitrogen  (mg/kg)   Total Nitrogen
    phosphorus      (mg/kg)   logged only — not used by current model
    potassium       (mg/kg)   logged only — not used by current model
    ec              (µS/cm)   Electrical Conductivity
    ph              (pH)      direct
    temperature     (°C)      direct
    turbidity_high  (bool)    GPIO digital — True=High, False=Low

THE MODEL EXPECTS (21 features, trained on IoTPond6.csv):
    Temperature(C), Turbidity(NTU), Dissolved Oxygen(g/ml),
    PH, Ammonia(g/ml), Nitrate(g/ml),
    hour, month, day_of_week, season,
    ammonia_rolling_mean, nitrate_rolling_mean, temp_rolling_mean,
    ammonia_rolling_std,  nitrate_rolling_std,
    ammonia_log, nitrate_log, ammonia_spike,
    nitrate_ph_interaction, ammonia_ph_interaction, temp_nitrate_interaction

CONVERSION SCIENCE:
    Nitrogen  → Ammonia   : pH-adjusted TN fractionation (Boyd 1998)
    EC        → Nitrate   : blended ionic + TN cross-check (Rhoades 1989,
                            FAO 2015) — 55% EC-based / 45% TN-based
    Turbidity → NTU       : digital proxy (High=100, Low=20 NTU)
    DO        → estimated : Henry's Law solubility + pH bloom correction
                            (APHA Standard Methods, 2017)

CHANGES IN v2.1.0:
    - AMMONIA_SPIKE_THRESHOLD corrected to 23.8816 (exact p75 from training data)
    - Input validation added — clamps out-of-range sensor values and warns caller
    - Rolling buffer seeding function added — restores state after restart from DB
    - Structured logging added throughout
    - Module-level model load wrapped in try/except with clear error message
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from collections import deque

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

# ── FEATURE_COLS — must match notebook Cell 6 exactly ────────────────────
FEATURE_COLS = [
    "Temperature(C)", "Turbidity(NTU)", "Dissolved Oxygen(g/ml)",
    "PH", "Ammonia(g/ml)", "Nitrate(g/ml)",
    "hour", "month", "day_of_week", "season",
    "ammonia_rolling_mean", "nitrate_rolling_mean", "temp_rolling_mean",
    "ammonia_rolling_std",  "nitrate_rolling_std",
    "ammonia_log", "nitrate_log",
    "ammonia_spike",
    "nitrate_ph_interaction", "ammonia_ph_interaction", "temp_nitrate_interaction",
]

# ── Rolling window buffers (last 6 readings) ──────────────────────────────
# Persist across requests within one process.
# Reset on container restart — call seed_rolling_buffers() on startup
# once the database is connected to restore the last 6 readings.
WINDOW              = 6
_ammonia_buffer     = deque(maxlen=WINDOW)
_nitrate_buffer     = deque(maxlen=WINDOW)
_temperature_buffer = deque(maxlen=WINDOW)

# ── Constants from training data ──────────────────────────────────────────
# Exact p75 of Ammonia(g/ml) on the clean IoTPond6 training set.
# Calculated in notebook Cell 6: df_feat['Ammonia(g/ml)'].quantile(0.75)
# Corrected from earlier hardcoded 10.08 — actual value is 23.8816
AMMONIA_SPIKE_THRESHOLD = 23.8816

# Safe operating ranges for each hardware input.
# Used by validate_inputs() to detect and clamp impossible readings.
# pH and temperature use physical/biological limits, not just training bounds,
# so the API stays robust even with sensor drift.
SENSOR_BOUNDS = {
    "ph":             (0.0,    14.0),
    "temperature":    (10.0,   40.0),
    "ec":             (0.0,  8000.0),
    "total_nitrogen": (0.0,  2000.0),
    "phosphorus":     (0.0,  1000.0),
    "potassium":      (0.0,  1000.0),
}


# ═══════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_inputs(
    total_nitrogen: float,
    ec:             float,
    ph:             float,
    temperature:    float,
    phosphorus:     float,
    potassium:      float,
) -> dict:
    """
    Validate hardware sensor inputs against safe operating ranges.

    Why this matters:
        The Random Forest was calibrated on IoTPond6 data with specific
        value distributions. Inputs far outside that range produce
        predictions the model was never trained for. Rather than silently
        returning unreliable results we log warnings and clamp extremes.

    Returns
    -------
    dict:
        valid          (bool)  — True if all inputs were within bounds
        warnings       (list)  — one string per out-of-range parameter
        clamped_values (dict)  — only the parameters that were adjusted
    """
    warnings = []
    clamped  = {}

    incoming = {
        "ph":             ph,
        "temperature":    temperature,
        "ec":             ec,
        "total_nitrogen": total_nitrogen,
        "phosphorus":     phosphorus,
        "potassium":      potassium,
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
# ROLLING BUFFER SEEDING  (call once on startup from main.py)
# ═══════════════════════════════════════════════════════════════════════════

def seed_rolling_buffers(last_readings: list) -> None:
    """
    Re-populate rolling buffers from the last N database records after restart.

    Without this, every container restart forces rolling_std_active=False
    until 6 new readings arrive. Call this in main.py startup:

        from database import get_last_n_predictions
        from predict  import seed_rolling_buffers
        seed_rolling_buffers(get_last_n_predictions(6))

    Parameters
    ----------
    last_readings : list of dicts, oldest first, each containing:
        "ammonia_g_ml"   (float)
        "nitrate_g_ml"   (float)
        "temperature_c"  (float)
    """
    for r in last_readings:
        _ammonia_buffer.append(r["ammonia_g_ml"])
        _nitrate_buffer.append(r["nitrate_g_ml"])
        _temperature_buffer.append(r["temperature_c"])

    logger.info(
        "Rolling buffers seeded with %d historical readings. "
        "rolling_std_active=%s",
        len(last_readings),
        len(_ammonia_buffer) > 1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CONVERSION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def nitrogen_to_ammonia(total_nitrogen: float, ph: float) -> float:
    """
    Total Nitrogen (mg/kg) → Ammonia (g/ml)

    Science:
        Total Nitrogen = TAN + Nitrate-N + Organic-N
        TAN (Total Ammonia Nitrogen) fraction = 15–25% depending on pH.
        Un-ionised NH3 (the toxic form) dominates at high pH (Boyd 1998).
        Conservative upper-bound fraction avoids under-predicting toxicity.

    mg/kg ÷ 1000 ≈ g/ml in dilute aqueous solution (density ≈ 1 g/ml)
    """
    if ph > 8.0:
        fraction = 0.25    # High pH — NH3 dominates, max toxic fraction
    elif ph > 7.5:
        fraction = 0.20    # Moderate pH — mixed NH3 / NH4+
    else:
        fraction = 0.15    # Low pH — NH4+ dominates, lower toxic fraction

    ammonia = round((total_nitrogen * fraction) / 1000.0, 6)
    logger.debug(
        "N→Ammonia: TN=%.2f ph=%.2f fraction=%.2f → %.6f g/ml",
        total_nitrogen, ph, fraction, ammonia,
    )
    return ammonia


def ec_and_nitrogen_to_nitrate(
    ec: float,
    total_nitrogen: float,
    ph: float,
) -> float:
    """
    EC (µS/cm) + Total Nitrogen (mg/kg) → Nitrate (g/ml)

    Two independent estimates blended 55 / 45:

    1. EC-based  (55% weight):
       Nitrate-N is the dominant ion in fish pond water.
       nitrate_mg/L = EC_µS/cm × 0.7  (Rhoades et al. 1989)
       ÷ 1000 → g/ml

    2. TN-based  (45% weight):
       Nitrate-N ≈ 60–65% of TN after subtracting ammonia fraction.
       (FAO Aquaculture Guidelines, 2015)

    EC gets higher weight — it is the more direct ionic measurement.
    TN-based estimate cross-checks and dampens EC sensor noise.
    """
    nitrate_from_ec = (ec * 0.7) / 1000.0

    if ph > 8.0:
        nitrate_fraction = 0.60
    elif ph > 7.5:
        nitrate_fraction = 0.63
    else:
        nitrate_fraction = 0.65

    nitrate_from_tn = (total_nitrogen * nitrate_fraction) / 1000.0

    blended = round((nitrate_from_ec * 0.55) + (nitrate_from_tn * 0.45), 6)

    logger.debug(
        "EC+TN→Nitrate: ec=%.2f TN=%.2f ph=%.2f "
        "ec_est=%.6f tn_est=%.6f blended=%.6f g/ml",
        ec, total_nitrogen, ph,
        nitrate_from_ec, nitrate_from_tn, blended,
    )
    return blended


def turbidity_digital_to_ntu(turbidity_high: bool) -> float:
    """
    GPIO digital turbidity signal → NTU approximation.

    IoTPond6 training data: 70.3% of turbidity readings = 100 NTU.
    The original pond sensor was effectively digital in practice.
    Mapping is consistent with the distribution the model was trained on.
        HIGH signal (cloudy, bloom conditions) → 100 NTU
        LOW  signal (clear, safe conditions)   →  20 NTU
    """
    ntu = 100.0 if turbidity_high else 20.0
    logger.debug("Turbidity: digital=%s → %.1f NTU", turbidity_high, ntu)
    return ntu


def estimate_dissolved_oxygen(temperature: float, ph: float) -> float:
    """
    Estimate Dissolved Oxygen (g/ml) from temperature and pH.

    Base: Henry's Law solubility (APHA Standard Methods, 2017)
        DO_sat = 14.62 - 0.3898T + 0.006969T² - 0.00005896T³
        Valid: 0–35°C

    pH correction models algal bloom dynamics:
        High pH = algae consuming CO2 via daytime photosynthesis.
        Same algae consume O2 overnight via respiration.
        High pH reliably predicts low overnight DO — the danger period.

    Converted mg/L → g/ml (× 0.001). Floored at 0.001 (no zero inputs).
    Conservative lower-bound — safer to over-flag than miss an O2 crash.
    """
    base_do = (
        14.62
        - (0.3898    * temperature)
        + (0.006969  * temperature ** 2)
        - (0.00005896 * temperature ** 3)
    )

    if ph > 8.5:
        base_do *= 0.50
    elif ph > 8.0:
        base_do *= 0.65
    elif ph > 7.5:
        base_do *= 0.82

    do_gml = round(max(base_do * 0.001, 0.001), 6)
    logger.debug(
        "DO estimate: temp=%.2f ph=%.2f base_do=%.4f → %.6f g/ml",
        temperature, ph, base_do, do_gml,
    )
    return do_gml


def get_temporal_context() -> dict:
    """
    Auto-fill temporal features from system clock.
    Matches the feature engineering in notebook Cell 6.

    Uganda seasons:
        Dry  (season=0): June – September
        Wet  (season=1): all other months
    """
    now    = datetime.now()
    month  = now.month
    season = 0 if month in [6, 7, 8, 9] else 1
    return {
        "hour":        now.hour,
        "month":       month,
        "day_of_week": now.weekday(),
        "season":      season,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_bloom_risk(
    total_nitrogen:   float,
    ec:               float,
    ph:               float,
    temperature:      float,
    turbidity_high:   bool,
    phosphorus:       float = 0.0,
    potassium:        float = 0.0,
    dissolved_oxygen: float = None,
) -> dict:
    """
    Predict algal bloom risk from hardware sensor readings.

    Parameters
    ----------
    total_nitrogen   : float   Total Nitrogen mg/kg  (RS485)
    ec               : float   Electrical Conductivity µS/cm  (RS485)
    ph               : float   pH 0–14  (RS485)
    temperature      : float   Water temperature °C  (DS18B20)
    turbidity_high   : bool    True = GPIO turbidity sensor HIGH
    phosphorus       : float   Phosphorus mg/kg — logged only
    potassium        : float   Potassium  mg/kg — logged only
    dissolved_oxygen : float   Optional real DO g/ml — bypasses estimation

    Returns
    -------
    dict — risk_level, risk_label, confidence, probabilities,
           action, model_inputs, hardware_inputs, data_quality, validation
    """

    # ── STEP 0: Validate and clamp inputs ───────────────────────────────
    validation = validate_inputs(
        total_nitrogen=total_nitrogen,
        ec=ec,
        ph=ph,
        temperature=temperature,
        phosphorus=phosphorus,
        potassium=potassium,
    )
    clamped        = validation["clamped_values"]
    total_nitrogen = clamped.get("total_nitrogen", total_nitrogen)
    ec             = clamped.get("ec",             ec)
    ph             = clamped.get("ph",             ph)
    temperature    = clamped.get("temperature",    temperature)
    phosphorus     = clamped.get("phosphorus",     phosphorus)
    potassium      = clamped.get("potassium",      potassium)

    # ── STEP 1: Temporal features ────────────────────────────────────────
    time_ctx = get_temporal_context()

    # ── STEP 2: Hardware → Model unit conversions ────────────────────────
    ammonia   = nitrogen_to_ammonia(total_nitrogen, ph)
    nitrate   = ec_and_nitrogen_to_nitrate(ec, total_nitrogen, ph)
    turbidity = turbidity_digital_to_ntu(turbidity_high)

    do_was_estimated = dissolved_oxygen is None
    if do_was_estimated:
        dissolved_oxygen = estimate_dissolved_oxygen(temperature, ph)

    # ── STEP 3: Update and read rolling buffers ──────────────────────────
    _ammonia_buffer.append(ammonia)
    _nitrate_buffer.append(nitrate)
    _temperature_buffer.append(temperature)

    ammonia_list     = list(_ammonia_buffer)
    nitrate_list     = list(_nitrate_buffer)
    temperature_list = list(_temperature_buffer)

    ammonia_rolling_mean = float(np.mean(ammonia_list))
    nitrate_rolling_mean = float(np.mean(nitrate_list))
    temp_rolling_mean    = float(np.mean(temperature_list))

    ammonia_rolling_std  = float(np.std(ammonia_list)) if len(ammonia_list) > 1 else 0.0
    nitrate_rolling_std  = float(np.std(nitrate_list)) if len(nitrate_list) > 1 else 0.0

    # ── STEP 4: Build all 21 features ───────────────────────────────────
    raw = {
        "Temperature(C)":           temperature,
        "Turbidity(NTU)":           turbidity,
        "Dissolved Oxygen(g/ml)":   dissolved_oxygen,
        "PH":                       ph,
        "Ammonia(g/ml)":            ammonia,
        "Nitrate(g/ml)":            nitrate,
        "hour":                     time_ctx["hour"],
        "month":                    time_ctx["month"],
        "day_of_week":              time_ctx["day_of_week"],
        "season":                   time_ctx["season"],
        "ammonia_rolling_mean":     ammonia_rolling_mean,
        "nitrate_rolling_mean":     nitrate_rolling_mean,
        "temp_rolling_mean":        temp_rolling_mean,
        "ammonia_rolling_std":      ammonia_rolling_std,
        "nitrate_rolling_std":      nitrate_rolling_std,
        "ammonia_log":              np.log1p(ammonia),
        "nitrate_log":              np.log1p(nitrate),
        "ammonia_spike":            int(ammonia > AMMONIA_SPIKE_THRESHOLD),
        "nitrate_ph_interaction":   nitrate    * ph,
        "ammonia_ph_interaction":   ammonia    * ph,
        "temp_nitrate_interaction": temperature * nitrate,
    }

    # ── STEP 5: Scale and predict ────────────────────────────────────────
    input_df     = pd.DataFrame([raw])[feature_cols]
    input_scaled = scaler.transform(input_df)

    prediction    = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    confidence    = float(probabilities[prediction] * 100)

    logger.info(
        "Prediction: %s (%.1f%%)  TN=%.1f EC=%.1f ph=%.2f temp=%.1f turb=%s",
        ["LOW", "MEDIUM", "HIGH"][int(prediction)],
        confidence, total_nitrogen, ec, ph, temperature,
        "HIGH" if turbidity_high else "LOW",
    )

    # ── STEP 6: Build and return response ───────────────────────────────
    risk_map = {
        0: {
            "label":    "LOW RISK",
            "icon":     "green",
            "severity": "low",
            "action":   (
                "Normal conditions. Continue routine monitoring. "
                "All parameters within safe operating range."
            ),
        },
        1: {
            "label":    "MEDIUM RISK",
            "icon":     "yellow",
            "severity": "medium",
            "action":   (
                "Elevated risk detected. Increase aeration and reduce "
                "feeding by 30%. Test water manually. "
                "Monitor every 30 minutes."
            ),
        },
        2: {
            "label":    "HIGH RISK",
            "icon":     "red",
            "severity": "high",
            "action":   (
                "CRITICAL — Immediate intervention required. "
                "Maximum aeration, stop feeding immediately, "
                "consider 20% water exchange. "
                "Alert pond manager immediately."
            ),
        },
    }

    info = risk_map[int(prediction)]

    return {
        "risk_level":    int(prediction),
        "risk_label":    info["label"],
        "icon":          info["icon"],
        "severity":      info["severity"],
        "confidence":    round(confidence, 2),
        "action":        info["action"],
        "probabilities": {
            "low_risk":    round(float(probabilities[0] * 100), 2),
            "medium_risk": round(float(probabilities[1] * 100), 2),
            "high_risk":   round(float(probabilities[2] * 100), 2),
        },
        "model_inputs": {
            "ammonia_g_ml":          ammonia,
            "nitrate_g_ml":          nitrate,
            "ph":                    ph,
            "temperature_c":         temperature,
            "turbidity_ntu":         turbidity,
            "dissolved_oxygen_g_ml": dissolved_oxygen,
        },
        "hardware_inputs": {
            "total_nitrogen_mg_kg": total_nitrogen,
            "ec_us_cm":             ec,
            "ph":                   ph,
            "temperature_c":        temperature,
            "turbidity_digital":    "HIGH" if turbidity_high else "LOW",
            "phosphorus_mg_kg":     phosphorus,
            "potassium_mg_kg":      potassium,
        },
        "data_quality": {
            "do_source": (
                "sensor"
                if not do_was_estimated
                else "estimated (Henry Law + pH correction)"
            ),
            "ammonia_source":      "pH-adjusted TN fractionation (Boyd 1998)",
            "nitrate_source":      "blended EC + TN estimate (55% EC / 45% TN)",
            "turbidity_source":    "digital proxy (High=100 NTU, Low=20 NTU)",
            "rolling_window_size": len(_ammonia_buffer),
            "rolling_std_active":  len(_ammonia_buffer) > 1,
        },
        "validation": {
            "inputs_valid":   validation["valid"],
            "warnings":       validation["warnings"],
            "values_clamped": len(validation["clamped_values"]) > 0,
        },
    }
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR     = Path(__file__).parent
MODEL_DIR    = BASE_DIR / "model"

print("Loading model artifacts...")
model        = joblib.load(MODEL_DIR / "random_forest_model.pkl")
scaler       = joblib.load(MODEL_DIR / "scaler.pkl")
feature_cols = joblib.load(MODEL_DIR / "feature_columns.pkl")
print(f"Model loaded. Expects {len(feature_cols)} features.")


def nitrogen_to_ammonia_nitrate(
    total_nitrogen: float,
    ph: float
) -> tuple:
    """
    Split Total Nitrogen (mg/kg) into Ammonia and Nitrate fractions.

    In fish pond water, TN is composed of:
      - Total Ammonia Nitrogen (TAN): 15–25% depending on pH
      - Nitrate-Nitrogen: ~65%
      - Organic Nitrogen: remainder (not used by model)

    The ammonia fraction is pH-dependent because un-ionised ammonia
    (NH3) — the toxic form — dominates at high pH. At low pH, ammonia
    is mostly ionised (NH4+) and less toxic. This matches Boyd (1998).
    """
    if ph > 8.0:
        ammonia_fraction = 0.25
    elif ph > 7.5:
        ammonia_fraction = 0.20
    else:
        ammonia_fraction = 0.15

    nitrate_fraction = 0.65

    # mg/kg ÷ 1000 = g/kg ≈ g/ml in dilute aqueous solution
    ammonia = (total_nitrogen * ammonia_fraction) / 1000.0
    nitrate  = (total_nitrogen * nitrate_fraction) / 1000.0

    return round(ammonia, 6), round(nitrate, 6)


def estimate_dissolved_oxygen(temperature: float, ph: float) -> float:
    """
    Estimate dissolved oxygen from temperature and pH.

    Uses Henry's Law solubility approximation:
      DO_sat ≈ 14.62 - 0.3898T + 0.006969T² - 0.0000590T³
    (APHA Standard Methods for the Examination of Water, 2017)

    pH correction reflects active algal O2 consumption during blooms.
    High pH = algae actively photosynthesising = O2 depleted at night.
    """
    base_do = (14.62
               - (0.3898  * temperature)
               + (0.006969 * temperature ** 2)
               - (0.00005896 * temperature ** 3))

    if ph > 8.5:
        base_do *= 0.55
    elif ph > 8.0:
        base_do *= 0.70
    elif ph > 7.5:
        base_do *= 0.85

    # mg/L → g/ml
    return round(max(base_do * 0.001, 0.001), 6)


def get_temporal_context() -> dict:
    """Auto-fill temporal features from system clock."""
    now = datetime.now()
    month = now.month
    # Uganda: June–September = Dry (0), all other months = Wet (1)
    season = 0 if month in [6, 7, 8, 9] else 1
    return {
        'hour':        now.hour,
        'month':       month,
        'day_of_week': now.weekday(),
        'season':      season,
    }


def predict_bloom_risk(
    # ── Hardware parameters (exactly what your sensors give you) ──
    total_nitrogen:   float,          # mg/kg  — RS485 Multi-Probe
    ec:               float,          # µS/cm  — RS485 Multi-Probe
    ph:               float,          # pH     — RS485 Multi-Probe
    temperature:      float,          # °C     — RS485 + DS18B20
    turbidity_high:   bool,           # True/False — GPIO Digital Sensor
    # ── Optional: pass real value when you add a DO sensor ────────
    dissolved_oxygen: float = None,   # g/ml   — future hardware
) -> dict:

    # ── STEP 1: Auto-fill time from system clock ──────────────
    time_ctx = get_temporal_context()

    # ── STEP 2: Convert hardware readings to model inputs ─────

    # Total Nitrogen → Ammonia + Nitrate (pH-adjusted fractions)
    ammonia, nitrate = nitrogen_to_ammonia_nitrate(total_nitrogen, ph)

    # EC (µS/cm) → Nitrate refinement
    # EC gives us an independent nitrate estimate to cross-check.
    # Standard pond water: nitrate_mg/L ≈ EC_µS/cm × 0.7
    # We blend both estimates: 60% TN-derived, 40% EC-derived
    nitrate_from_ec = (ec * 0.7) / 1000.0
    nitrate = round((nitrate * 0.60) + (nitrate_from_ec * 0.40), 6)

    # Turbidity: Digital → NTU proxy
    turbidity = 100.0 if turbidity_high else 20.0

    # DO: real sensor takes priority, otherwise estimate
    do_was_estimated = dissolved_oxygen is None
    if do_was_estimated:
        dissolved_oxygen = estimate_dissolved_oxygen(temperature, ph)

    # ── STEP 3: Build all 21 features ────────────────────────
    raw = {
        'Temperature(C)':           temperature,
        'Turbidity(NTU)':           turbidity,
        'PH':                       ph,
        'Ammonia(g/ml)':            ammonia,
        'Nitrate(g/ml)':            nitrate,
        'Dissolved Oxygen(g/ml)':   dissolved_oxygen,
        'hour':                     time_ctx['hour'],
        'month':                    time_ctx['month'],
        'day_of_week':              time_ctx['day_of_week'],
        'season':                   time_ctx['season'],
        'ammonia_rolling_mean':     ammonia,
        'nitrate_rolling_mean':     nitrate,
        'temp_rolling_mean':        temperature,
        'ammonia_rolling_std':      0.0,
        'nitrate_rolling_std':      0.0,
        'ammonia_log':              np.log1p(ammonia),
        'nitrate_log':              np.log1p(nitrate),
        'ammonia_spike':            1 if ammonia > 10 else 0,
        'nitrate_ph_interaction':   nitrate * ph,
        'ammonia_ph_interaction':   ammonia * ph,
        'temp_nitrate_interaction': temperature * nitrate,
    }

    # ── STEP 4: Scale and predict ─────────────────────────────
    input_df     = pd.DataFrame([raw])[feature_cols]
    input_scaled = scaler.transform(input_df)

    prediction    = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    confidence    = float(probabilities[prediction] * 100)

    risk_map = {
        0: {
            'label':    'LOW RISK',
            'icon':     'green',
            'action':   'Normal conditions. Continue routine monitoring.',
            'severity': 'low'
        },
        1: {
            'label':    'MEDIUM RISK',
            'icon':     'yellow',
            'action':   'Elevated risk detected. Increase aeration and '
                        'reduce feeding by 30%. Monitor every 30 minutes.',
            'severity': 'medium'
        },
        2: {
            'label':    'HIGH RISK',
            'icon':     'red',
            'action':   'CRITICAL — Immediate intervention required. '
                        'Maximum aeration, stop feeding, consider partial '
                        'water exchange. Alert pond manager immediately.',
            'severity': 'high'
        }
    }

    return {
        'risk_level':    int(prediction),
        'risk_label':    risk_map[prediction]['label'],
        'icon':          risk_map[prediction]['icon'],
        'severity':      risk_map[prediction]['severity'],
        'confidence':    round(confidence, 2),
        'action':        risk_map[prediction]['action'],
        'probabilities': {
            'low_risk':    round(float(probabilities[0] * 100), 2),
            'medium_risk': round(float(probabilities[1] * 100), 2),
            'high_risk':   round(float(probabilities[2] * 100), 2)
        },
        'model_inputs': {
            'ammonia':          ammonia,
            'nitrate':          nitrate,
            'ph':               ph,
            'temperature':      temperature,
            'turbidity':        turbidity,
            'dissolved_oxygen': dissolved_oxygen,
        },
        'hardware_inputs': {
            'total_nitrogen_mg_kg': total_nitrogen,
            'ec_us_cm':             ec,
            'ph':                   ph,
            'temperature_c':        temperature,
            'turbidity_digital':    'HIGH' if turbidity_high else 'LOW',
        },
        'data_quality': {
            'do_estimated':         do_was_estimated,
            'nitrogen_split_method':'pH-adjusted TN fractionation (Boyd 1998)',
            'turbidity_method':     'digital proxy (High=100 NTU, Low=20 NTU)',
            'temporal_auto_filled': True,
        }
    }
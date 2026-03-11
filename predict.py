import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

print("Loading model artifacts...")
model        = joblib.load(MODEL_DIR / "random_forest_model.pkl")
scaler       = joblib.load(MODEL_DIR / "scaler.pkl")
feature_cols = joblib.load(MODEL_DIR / "feature_columns.pkl")
print(f"Model loaded. Expects {len(feature_cols)} features.")


def estimate_dissolved_oxygen(temperature: float, ph: float) -> float:
    """
    Estimate dissolved oxygen from temperature and pH.
    Based on: warm water holds less O2 (solubility curve),
    and high pH during blooms indicates active O2 consumption.

    Returns a conservative estimate in g/ml.
    """
    # Base DO from temperature (standard solubility curve approximation)
    # At 20°C → ~9.1 mg/L, at 30°C → ~7.5 mg/L
    base_do = 14.62 - (0.3898 * temperature) + (0.006969 * temperature ** 2) - (0.00005896 * temperature ** 3)

    # pH correction: high pH (active bloom) means O2 is being consumed fast
    # pH above 8.0 → reduce estimated DO
    if ph > 8.5:
        base_do *= 0.55   # severe bloom → very low DO
    elif ph > 8.0:
        base_do *= 0.70   # active bloom → reduced DO
    elif ph > 7.5:
        base_do *= 0.85   # mild concern → slightly reduced
    # below 7.5 → no correction, pond chemistry normal

    # Convert mg/L to g/ml (1 mg/L = 0.001 g/L = 0.001 g/ml)
    return round(max(base_do * 0.001, 0.001), 5)


def predict_bloom_risk(
    # ── Your actual hardware parameters ──────────────────────
    nitrogen:    float,        # mg/kg  from RS485 Multi-Probe
    ec:          float,        # µS/cm  from RS485 Multi-Probe
    ph:          float,        # pH     from RS485 Multi-Probe
    temperature: float,        # °C     from RS485 + DS18B20
    turbidity_high: bool,      # True=High, False=Low from GPIO
    # ── Optional — pass if you add a DO sensor later ─────────
    dissolved_oxygen: float = None,
    # ── Temporal context (auto-filled from system clock) ─────
    hour:        int = 12,
    month:       int = 6,
    day_of_week: int = 2,
    season:      int = 0
) -> dict:

    # ── SENSOR CONVERSIONS ────────────────────────────────────
    # Nitrogen (mg/kg) → Ammonia (g/ml)
    ammonia = nitrogen / 1000.0

    # EC (µS/cm) → Nitrate (g/ml)
    # Standard pond water conversion: EC × 0.7 = nitrate in mg/L
    # then mg/L ÷ 1000 = g/ml
    nitrate = (ec * 0.7) / 1000.0

    # Turbidity: Digital High/Low → NTU approximation
    # High signal = turbid/bloom-risk water = 100 NTU
    # Low signal  = clear water             = 20 NTU
    turbidity = 100.0 if turbidity_high else 20.0

    # Dissolved Oxygen: use sensor if available, else estimate
    if dissolved_oxygen is None:
        dissolved_oxygen = estimate_dissolved_oxygen(temperature, ph)

    # ── FEATURE CONSTRUCTION ──────────────────────────────────
    raw = {
        'Temperature(C)':         temperature,
        'Turbidity(NTU)':         turbidity,
        'PH':                     ph,
        'Ammonia(g/ml)':          ammonia,
        'Nitrate(g/ml)':          nitrate,
        'Dissolved Oxygen(g/ml)': dissolved_oxygen,
        'hour':                   hour,
        'month':                  month,
        'day_of_week':            day_of_week,
        'season':                 season,
        # Rolling features — use current reading as proxy
        # (in production, pass the last 6 readings' average here)
        'ammonia_rolling_mean':   ammonia,
        'nitrate_rolling_mean':   nitrate,
        'temp_rolling_mean':      temperature,
        'ammonia_rolling_std':    0.0,
        'nitrate_rolling_std':    0.0,
        # Log transforms
        'ammonia_log':            np.log1p(ammonia),
        'nitrate_log':            np.log1p(nitrate),
        # Spike flag — ammonia > 10 g/ml threshold
        'ammonia_spike':          1 if ammonia > 10 else 0,
        # Interaction features
        'nitrate_ph_interaction':   nitrate * ph,
        'ammonia_ph_interaction':   ammonia * ph,
        'temp_nitrate_interaction': temperature * nitrate,
    }

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
        # What the model actually received (after conversion)
        'model_inputs': {
            'ammonia':          round(ammonia, 5),
            'nitrate':          round(nitrate, 5),
            'ph':               ph,
            'temperature':      temperature,
            'turbidity':        turbidity,
            'dissolved_oxygen': round(dissolved_oxygen, 5),
        },
        # Raw hardware readings (for logging/debugging)
        'hardware_inputs': {
            'nitrogen_mg_kg':    nitrogen,
            'ec_us_cm':          ec,
            'ph':                ph,
            'temperature_c':     temperature,
            'turbidity_digital': 'HIGH' if turbidity_high else 'LOW',
        },
        'do_estimated': dissolved_oxygen is None
    }
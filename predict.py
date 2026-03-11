"""
AquaSentinel — predict.py
=========================
Hardware-to-Model conversion layer.

YOUR HARDWARE (RS485 Multi-Probe + DS18B20 + GPIO):
    nitrogen     (mg/kg)    Total Nitrogen
    phosphorus   (mg/kg)    not used by model — logged only
    potassium    (mg/kg)    not used by model — logged only
    ec           (µS/cm)    Electrical Conductivity
    ph           (pH)       direct
    temperature  (°C)       direct
    turbidity    (bool)     GPIO digital — True=High, False=Low

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
    EC        → Nitrate   : standard ionic approximation + TN cross-check
    Turbidity → NTU       : digital proxy (High=100, Low=20 NTU)
    DO        → estimated : Henry's Law solubility + pH bloom correction
                            (APHA Standard Methods, 2017)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from collections import deque

# ── Load model artifacts ──────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
MODEL_DIR    = BASE_DIR / "model"

print("Loading model artifacts...")
model        = joblib.load(MODEL_DIR / "random_forest_model.pkl")
scaler       = joblib.load(MODEL_DIR / "scaler.pkl")
feature_cols = joblib.load(MODEL_DIR / "feature_columns.pkl")
print(f"Model loaded. Expects {len(feature_cols)} features.")

# ── FEATURE_COLS — must match notebook Cell 6 exactly ────────────────────
FEATURE_COLS = [
    # Raw sensors (6)
    'Temperature(C)', 'Turbidity(NTU)', 'Dissolved Oxygen(g/ml)',
    'PH', 'Ammonia(g/ml)', 'Nitrate(g/ml)',
    # Temporal (4)
    'hour', 'month', 'day_of_week', 'season',
    # Rolling means (3)
    'ammonia_rolling_mean', 'nitrate_rolling_mean', 'temp_rolling_mean',
    # Rolling std (2)
    'ammonia_rolling_std', 'nitrate_rolling_std',
    # Log transforms (2)
    'ammonia_log', 'nitrate_log',
    # Spike flag (1)
    'ammonia_spike',
    # Interactions (3)
    'nitrate_ph_interaction', 'ammonia_ph_interaction', 'temp_nitrate_interaction',
]

# ── Rolling window buffer (last 6 readings) ───────────────────────────────
# This replaces the 0.0 std dev placeholder in earlier versions.
# As readings accumulate, rolling std becomes real rather than fixed.
WINDOW = 6
_ammonia_buffer     = deque(maxlen=WINDOW)
_nitrate_buffer     = deque(maxlen=WINDOW)
_temperature_buffer = deque(maxlen=WINDOW)

# ── Ammonia spike threshold ───────────────────────────────────────────────
# From notebook Cell 6: spike_threshold = df_feat['Ammonia(g/ml)'].quantile(0.75)
# Calculated on IoTPond6 training data = ~10.08 g/ml
# Any ammonia reading above this is flagged as a spike event.
AMMONIA_SPIKE_THRESHOLD = 10.08


# ═══════════════════════════════════════════════════════════════════════════
# CONVERSION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def nitrogen_to_ammonia(total_nitrogen: float, ph: float) -> float:
    """
    Convert Total Nitrogen (mg/kg) → Ammonia in g/ml.

    Science:
        Total Nitrogen = TAN (Total Ammonia Nitrogen) + Nitrate-N + Organic-N
        TAN fraction in fish ponds = 15–25% depending on pH.

        Un-ionised ammonia (NH3) — the toxic form — dominates at high pH.
        At pH > 8.0, free ammonia fraction rises sharply (Boyd 1998, Table 3.2).
        At pH < 7.5, ammonia is mostly ionised (NH4+) and far less toxic.

        We use a conservative upper-bound fraction to avoid under-predicting
        ammonia toxicity — it is safer to over-estimate than under-estimate.

    mg/kg ÷ 1000 = g/kg ≈ g/ml in dilute aqueous solution (water density ≈ 1)
    """
    if ph > 8.0:
        fraction = 0.25   # High pH: NH3 dominates, maximum toxic fraction
    elif ph > 7.5:
        fraction = 0.20   # Moderate pH: mixed NH3/NH4+
    else:
        fraction = 0.15   # Low pH: NH4+ dominates, lower toxic fraction

    return round((total_nitrogen * fraction) / 1000.0, 6)


def ec_and_nitrogen_to_nitrate(ec: float, total_nitrogen: float, ph: float) -> float:
    """
    Derive Nitrate (g/ml) from both EC and Total Nitrogen, then blend.

    Two independent estimates — blended for robustness:

    1. EC-based estimate:
       In fish pond water, nitrate-N is the dominant dissolved ion.
       Standard conversion: nitrate_mg/L ≈ EC_µS/cm × 0.7
       (Rhoades et al. 1989 — widely used in aquatic chemistry)
       Then mg/L ÷ 1000 = g/ml

    2. TN-based estimate:
       Nitrate-N makes up approximately 60–65% of TN in a fish pond
       after ammonia and organic nitrogen fractions are subtracted.
       (FAO Aquaculture Guidelines, 2015)

    Blend: 55% EC-based + 45% TN-based
    EC is the more direct measurement so it gets the higher weight.
    TN-based provides a cross-check that prevents EC noise from
    causing extreme nitrate estimates.
    """
    # Estimate 1: from EC
    nitrate_from_ec = (ec * 0.7) / 1000.0

    # Estimate 2: from Total Nitrogen (remaining fraction after ammonia)
    if ph > 8.0:
        nitrate_fraction = 0.60   # Less TN available as nitrate at high pH
    elif ph > 7.5:
        nitrate_fraction = 0.63
    else:
        nitrate_fraction = 0.65   # More TN as nitrate at lower pH

    nitrate_from_tn = (total_nitrogen * nitrate_fraction) / 1000.0

    # Blended estimate: EC leads, TN cross-checks
    blended = (nitrate_from_ec * 0.55) + (nitrate_from_tn * 0.45)

    return round(blended, 6)


def turbidity_digital_to_ntu(turbidity_high: bool) -> float:
    """
    Convert GPIO digital turbidity signal → NTU approximation.

    Training data context (IoTPond6.csv):
        95% of Turbidity(NTU) readings = 100 NTU
        5% of readings = lower values (~20–40 NTU)

    This sensor only reports whether turbidity has crossed a threshold
    (cloudy vs clear). We map to values consistent with training data
    distribution so the model receives input within its calibration range.

    High signal = water is turbid/cloudy = bloom-risk conditions → 100 NTU
    Low signal  = water is clear         = safe conditions        →  20 NTU
    """
    return 100.0 if turbidity_high else 20.0


def estimate_dissolved_oxygen(temperature: float, ph: float) -> float:
    """
    Estimate Dissolved Oxygen (g/ml) from temperature and pH.

    Science:
        Base DO from temperature using Henry's Law solubility approximation:
            DO_sat = 14.62 - 0.3898T + 0.006969T² - 0.0000590T³
        Source: APHA Standard Methods for the Examination of Water (2017)
        Valid range: 0–35°C

        pH correction reflects algal bloom dynamics:
            High pH = algae actively photosynthesising = CO2 consumed
            At NIGHT, same algae consume O2 through respiration.
            High daytime pH is therefore a reliable predictor of
            low overnight DO — the most dangerous period for fish.

        Result converted from mg/L to g/ml:
            1 mg/L = 0.001 g/L ≈ 0.001 g/ml (dilute solution)

    Returns conservative lower-bound estimate to avoid under-predicting
    bloom risk — safer to over-flag than to miss an oxygen crash.
    """
    # Henry's Law solubility curve
    base_do = (14.62
               - (0.3898  * temperature)
               + (0.006969 * temperature ** 2)
               - (0.00005896 * temperature ** 3))

    # pH correction: high pH = active bloom = O2 consumption at night
    if ph > 8.5:
        base_do *= 0.50    # Severe bloom: DO likely below 4 mg/L overnight
    elif ph > 8.0:
        base_do *= 0.65    # Active bloom: significant O2 depletion
    elif ph > 7.5:
        base_do *= 0.82    # Mild concern: moderate reduction

    # Convert mg/L → g/ml, floor at 0.001 to avoid zero input to model
    return round(max(base_do * 0.001, 0.001), 6)


def get_temporal_context() -> dict:
    """
    Auto-fill temporal features from system clock.
    Matches the feature engineering in notebook Cell 6.

    Uganda seasons (from notebook):
        Dry:  June, July, August, September → season = 0
        Wet:  all other months             → season = 1
    """
    now   = datetime.now()
    month = now.month
    season = 0 if month in [6, 7, 8, 9] else 1

    return {
        'hour':        now.hour,
        'month':       month,
        'day_of_week': now.weekday(),   # 0=Monday, 6=Sunday
        'season':      season,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_bloom_risk(
    # ── Your hardware parameters — exactly as sensors report them ──────
    total_nitrogen:  float,     # mg/kg  from RS485 Multi-Probe
    ec:              float,     # µS/cm  from RS485 Multi-Probe
    ph:              float,     # pH     from RS485 Multi-Probe
    temperature:     float,     # °C     from RS485 + DS18B20
    turbidity_high:  bool,      # True/False from GPIO Digital Sensor
    # ── Logged but not used by current model ───────────────────────────
    phosphorus:      float = 0.0,  # mg/kg — stored for future model version
    potassium:       float = 0.0,  # mg/kg — stored for future model version
    # ── Optional: pass real DO reading if you add a DO sensor later ────
    dissolved_oxygen: float = None,   # g/ml — bypasses estimation if provided
) -> dict:
    """
    Predict algal bloom risk from your hardware sensor readings.

    Accepts the 7 parameters your RS485 Multi-Probe and GPIO sensors
    provide. Converts them to the 21 features the trained Random Forest
    model expects, then returns a complete risk assessment.

    Parameters
    ----------
    total_nitrogen   : float   Total Nitrogen in mg/kg (RS485)
    ec               : float   Electrical Conductivity in µS/cm (RS485)
    ph               : float   pH 0–14 (RS485)
    temperature      : float   Water temperature in °C (DS18B20)
    turbidity_high   : bool    True if GPIO turbidity sensor is HIGH
    phosphorus       : float   Phosphorus mg/kg — logged, not used by model
    potassium        : float   Potassium mg/kg  — logged, not used by model
    dissolved_oxygen : float   Optional DO in g/ml — pass when sensor added

    Returns
    -------
    dict with risk_level, risk_label, confidence, probabilities,
         action, model_inputs, hardware_inputs, data_quality
    """

    # ── STEP 1: Temporal context from system clock ──────────────────────
    time_ctx = get_temporal_context()

    # ── STEP 2: Hardware → Model unit conversions ───────────────────────

    # Nitrogen (mg/kg) → Ammonia (g/ml) via pH-adjusted TN fractionation
    ammonia = nitrogen_to_ammonia(total_nitrogen, ph)

    # EC (µS/cm) + Nitrogen (mg/kg) → Nitrate (g/ml) blended estimate
    nitrate = ec_and_nitrogen_to_nitrate(ec, total_nitrogen, ph)

    # Turbidity: GPIO digital → NTU proxy
    turbidity = turbidity_digital_to_ntu(turbidity_high)

    # Dissolved Oxygen: use sensor if provided, else estimate
    do_was_estimated = dissolved_oxygen is None
    if do_was_estimated:
        dissolved_oxygen = estimate_dissolved_oxygen(temperature, ph)

    # ── STEP 3: Update rolling buffers with this reading ────────────────
    _ammonia_buffer.append(ammonia)
    _nitrate_buffer.append(nitrate)
    _temperature_buffer.append(temperature)

    ammonia_values     = list(_ammonia_buffer)
    nitrate_values     = list(_nitrate_buffer)
    temperature_values = list(_temperature_buffer)

    ammonia_rolling_mean = float(np.mean(ammonia_values))
    nitrate_rolling_mean = float(np.mean(nitrate_values))
    temp_rolling_mean    = float(np.mean(temperature_values))

    # Rolling std: 0.0 for first reading, real value after 2+ readings
    ammonia_rolling_std = float(np.std(ammonia_values))     if len(ammonia_values) > 1 else 0.0
    nitrate_rolling_std = float(np.std(nitrate_values))     if len(nitrate_values) > 1 else 0.0

    # ── STEP 4: Build all 21 features ───────────────────────────────────
    raw = {
        # Raw sensors (6)
        'Temperature(C)':           temperature,
        'Turbidity(NTU)':           turbidity,
        'Dissolved Oxygen(g/ml)':   dissolved_oxygen,
        'PH':                       ph,
        'Ammonia(g/ml)':            ammonia,
        'Nitrate(g/ml)':            nitrate,
        # Temporal (4)
        'hour':                     time_ctx['hour'],
        'month':                    time_ctx['month'],
        'day_of_week':              time_ctx['day_of_week'],
        'season':                   time_ctx['season'],
        # Rolling means (3) — real trend from last 6 readings
        'ammonia_rolling_mean':     ammonia_rolling_mean,
        'nitrate_rolling_mean':     nitrate_rolling_mean,
        'temp_rolling_mean':        temp_rolling_mean,
        # Rolling std (2) — real volatility from last 6 readings
        'ammonia_rolling_std':      ammonia_rolling_std,
        'nitrate_rolling_std':      nitrate_rolling_std,
        # Log transforms (2)
        'ammonia_log':              np.log1p(ammonia),
        'nitrate_log':              np.log1p(nitrate),
        # Spike flag (1) — threshold from training data p75
        'ammonia_spike':            int(ammonia > AMMONIA_SPIKE_THRESHOLD),
        # Interactions (3)
        'nitrate_ph_interaction':   nitrate   * ph,
        'ammonia_ph_interaction':   ammonia   * ph,
        'temp_nitrate_interaction': temperature * nitrate,
    }

    # ── STEP 5: Scale and predict ────────────────────────────────────────
    input_df     = pd.DataFrame([raw])[feature_cols]
    input_scaled = scaler.transform(input_df)

    prediction    = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    confidence    = float(probabilities[prediction] * 100)

    # ── STEP 6: Build response ───────────────────────────────────────────
    risk_map = {
        0: {
            'label':    'LOW RISK',
            'icon':     'green',
            'severity': 'low',
            'action':   'Normal conditions. Continue routine monitoring. '
                        'All parameters within safe operating range.',
        },
        1: {
            'label':    'MEDIUM RISK',
            'icon':     'yellow',
            'severity': 'medium',
            'action':   'Elevated risk detected. Increase aeration and '
                        'reduce feeding by 30%. Test water manually. '
                        'Monitor every 30 minutes.',
        },
        2: {
            'label':    'HIGH RISK',
            'icon':     'red',
            'severity': 'high',
            'action':   'CRITICAL — Immediate intervention required. '
                        'Maximum aeration, stop feeding immediately, '
                        'consider 20% water exchange. '
                        'Alert pond manager immediately.',
        },
    }

    info = risk_map[int(prediction)]

    return {
        # ── Core prediction ──────────────────────────────────────────────
        'risk_level':    int(prediction),
        'risk_label':    info['label'],
        'icon':          info['icon'],
        'severity':      info['severity'],
        'confidence':    round(confidence, 2),
        'action':        info['action'],
        'probabilities': {
            'low_risk':    round(float(probabilities[0] * 100), 2),
            'medium_risk': round(float(probabilities[1] * 100), 2),
            'high_risk':   round(float(probabilities[2] * 100), 2),
        },
        # ── What model actually received (after conversion) ──────────────
        'model_inputs': {
            'ammonia_g_ml':          ammonia,
            'nitrate_g_ml':          nitrate,
            'ph':                    ph,
            'temperature_c':         temperature,
            'turbidity_ntu':         turbidity,
            'dissolved_oxygen_g_ml': dissolved_oxygen,
        },
        # ── Raw hardware readings (for logging and debugging) ─────────────
        'hardware_inputs': {
            'total_nitrogen_mg_kg':  total_nitrogen,
            'ec_us_cm':              ec,
            'ph':                    ph,
            'temperature_c':         temperature,
            'turbidity_digital':     'HIGH' if turbidity_high else 'LOW',
            'phosphorus_mg_kg':      phosphorus,    # logged, not used by model
            'potassium_mg_kg':       potassium,     # logged, not used by model
        },
        # ── Transparency flags ───────────────────────────────────────────
        'data_quality': {
            'do_source':              'sensor'         if not do_was_estimated
                                      else 'estimated (Henry Law + pH correction)',
            'ammonia_source':         'pH-adjusted TN fractionation (Boyd 1998)',
            'nitrate_source':         'blended EC + TN estimate (55% EC / 45% TN)',
            'turbidity_source':       'digital proxy (High=100 NTU, Low=20 NTU)',
            'rolling_window_size':    len(_ammonia_buffer),
            'rolling_std_active':     len(_ammonia_buffer) > 1,
        },
    }
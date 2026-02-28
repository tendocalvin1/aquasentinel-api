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


def predict_bloom_risk(
    temperature: float,
    turbidity: float,
    ph: float,
    ammonia: float,
    nitrate: float,
    dissolved_oxygen: float,
    hour: int = 12,
    month: int = 6,
    day_of_week: int = 2,
    season: int = 0
) -> dict:

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
    }

    raw['ammonia_rolling_mean'] = ammonia
    raw['nitrate_rolling_mean'] = nitrate
    raw['temp_rolling_mean']    = temperature
    raw['ammonia_rolling_std']  = 0.0
    raw['nitrate_rolling_std']  = 0.0

    raw['ammonia_log'] = np.log1p(ammonia)
    raw['nitrate_log'] = np.log1p(nitrate)

    raw['ammonia_spike'] = 1 if ammonia > 10 else 0

    raw['nitrate_ph_interaction']   = nitrate * ph
    raw['ammonia_ph_interaction']   = ammonia * ph
    raw['temp_nitrate_interaction'] = temperature * nitrate

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
        'input_received': {
            'ammonia':          ammonia,
            'nitrate':          nitrate,
            'ph':               ph,
            'temperature':      temperature,
            'turbidity':        turbidity,
            'dissolved_oxygen': dissolved_oxygen
        }
    }
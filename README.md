# AquaSentinel API

> Real-time water quality assessment for aquaculture pond monitoring — powered by IoT sensors and Machine Learning.

AquaSentinel is an AI-powered system that assesses water quality in fish ponds using real IoT sensor data. Built on a Random Forest model trained on 3,887 samples from the Aquaculture Water Quality Dataset (AWD), it classifies pond water into three quality levels with **85.5% accuracy** and **100% Poor Precision** — meaning every POOR alert is a genuine water quality problem, with zero false alarms.

---

## Live API

| | |
|---|---|
| **Base URL** | https://aquasentinel-api-production.up.railway.app |
| **Interactive Docs** | https://aquasentinel-api-production.up.railway.app/docs |
| **Version** | 3.0.0 |
| **Status** | 🟢 Online |

---

## The Problem

Poor water quality is the leading cause of fish stock loss in East African aquaculture ponds. Elevated Nitrite and Phosphorus, combined with thermal stress and pH imbalance, create conditions that kill fish overnight. No affordable, automated water quality monitoring system existed for smallholder fish farmers in Uganda — AquaSentinel was built to solve that.

---

## How It Works

```
RS485 Multi-Probe + DS18B20 (4 sensor readings)
        ↓
AquaSentinel API (FastAPI on Railway)
        ↓  engineers 10 features from 4 raw readings
Random Forest Model (trained on 3,887 aquaculture samples)
        ↓
Pond Manager receives EXCELLENT / GOOD / POOR + recommended action
```

The four hardware parameters map **directly** to what the model was trained on — no unit conversion layer is needed.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | System status and version |
| `GET`  | `/api/health` | Model health check — feature list and accuracy |
| `POST` | `/api/predict` | **Main endpoint** — assess water quality from sensor readings |
| `GET`  | `/api/history` | Last 50 assessments with timestamps (persisted to SQLite) |
| `GET`  | `/api/stats` | Aggregated quality level counts |

---

## Example Request

`POST /api/predict`

```json
{
  "temperature": 25.0,
  "ph": 7.2,
  "nitrite": 0.1,
  "phosphorus": 0.05
}
```

### Parameter Reference

| Parameter | Unit | Valid Range | Hardware Source | Required |
|-----------|------|------------|----------------|----------|
| `temperature` | °C | 0 – 50 | DS18B20 / RS485 | ✅ |
| `ph` | pH | 0 – 14 | RS485 pH probe | ✅ |
| `nitrite` | mg/L | 0 – 20 | RS485 Nitrite channel | ✅ |
| `phosphorus` | mg/L | 0 – 20 | RS485 Phosphorus channel | ✅ |

---

## Example Response

```json
{
  "quality_level": 0,
  "quality_label": "EXCELLENT",
  "icon": "green",
  "severity": "excellent",
  "confidence": 94.5,
  "action": "Optimal water quality. All parameters within ideal range. Continue routine monitoring every 6 hours.",
  "probabilities": {
    "excellent": 94.5,
    "good": 4.2,
    "poor": 1.3
  },
  "sensor_inputs": {
    "temperature_c": 25.0,
    "ph": 7.2,
    "nitrite_mg_l": 0.1,
    "phosphorus_mg_l": 0.05
  },
  "data_quality": {
    "features_used": 10,
    "feature_source": "4 raw RS485 readings → 10 engineered features",
    "no_conversion": true,
    "hardware_match": "RS485 measures exactly what model was trained on"
  },
  "validation": {
    "inputs_valid": true,
    "warnings": [],
    "values_clamped": false
  }
}
```

---

## Water Quality Levels

| Level | Label | Meaning | Recommended Action |
|-------|-------|---------|-------------------|
| `0` | 🟢 EXCELLENT | Optimal conditions | Routine monitoring every 6 hours. All parameters ideal. |
| `1` | 🟡 GOOD | Acceptable conditions | Monitor every 2–3 hours. Light aeration increase if trend worsening. |
| `2` | 🔴 POOR | Critical deterioration | **Immediate action.** Increase aeration, reduce feeding 50%, consider 20% water exchange. Alert pond manager now. |

---

## Feature Engineering

The model uses **10 features** built from 4 raw sensor readings:

| Feature | Type | Biological Rationale |
|---------|------|---------------------|
| `Temp` | Raw | Thermal stress on fish |
| `pH` | Raw | Chemical environment |
| `Nitrite (mg/L)` | Raw | Primary toxicity driver (19.7% importance) |
| `Phosphorus (mg/L)` | Raw | Nutrient load indicator |
| `nitrite_log` | Log transform | Corrects right-skew — top feature (17.6% importance) |
| `phosphorus_log` | Log transform | Corrects right-skew for Phosphorus |
| `ph_nitrite_interaction` | Interaction | Un-ionised nitrite rises sharply at high pH |
| `phosphorus_nitrite_product` | Interaction | Dual nutrient stress index |
| `temp_ph_interaction` | Interaction | Compound thermal + chemical stress |
| `temp_squared` | Polynomial | Non-linear thermal effects (O₂ solubility) |

---

## Model Performance

### Three-Model Comparison

| Metric | Logistic Regression | Decision Tree | **Random Forest** |
|--------|--------------------|--------------|--------------------|
| Overall Accuracy | 73.1% | 84.8% | **85.5%** |
| Macro F1 Score | 0.71 | 0.84 | **0.85** |
| CV Mean F1 (5-fold) | 0.701 | 0.842 | **0.844** |
| CV Std Dev | 0.017 | 0.013 | **0.016** |
| Poor Recall | 0.39 | 0.59 | **0.59** |
| Poor Precision | 0.73 | 0.98 | **1.00** |
| Poor F1 | 0.51 | 0.73 | **0.74** |

### Top Feature Importances

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | nitrite_log | 19.71% |
| 2 | Nitrite (mg L-1) | 17.63% |
| 3 | ph_nitrite_interaction | 12.55% |
| 4 | Temp | 10.05% |
| 5 | temp_squared | 9.94% |

Top 3 features account for ~50% of predictive power. Top 5 account for ~70%.

### Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=20,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

---

## Dataset

| Property | Value |
|----------|-------|
| Name | Aquaculture Water Quality Dataset (AWD) |
| Source | Mendeley Data — DOI: 10.17632/y78ty2g293.1 |
| Citation | Veeramsetty, Arabelli, Bernatin (2024) |
| Raw samples | 4,300 |
| Clean samples | 3,887 (after removing 100 temperature outliers, 16 pH outliers, 300 duplicates) |
| Features engineered | 10 (from 4 raw sensor readings) |
| Labels | 0=Excellent (1,250) · 1=Good (1,250) · 2=Poor (1,387) |
| Training set | 3,109 samples (80%) |
| Test set | 778 samples (20%) |
| Split method | Stratified by class, random_state=42 |

---

## Run Locally

```bash
git clone https://github.com/tendocalvin1/aquasentinel-api.git
cd aquasentinel-api

# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

API available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| ML Model | Random Forest — scikit-learn |
| API Framework | FastAPI 0.110.0 |
| Server | Uvicorn |
| Database | SQLite via SQLAlchemy — persists across restarts |
| Deployment | Railway via Docker |
| Version Control | GitHub |
| Hardware | RS485 Multi-Probe (Nitrite, Phosphorus, pH), DS18B20 temperature sensor |

---

## Project Structure

```
aquasentinel-api/
├── main.py              # FastAPI app — endpoints, input validation
├── predict.py           # Feature engineering + Random Forest inference
├── database.py          # SQLAlchemy models and assessment logging
├── requirements.txt
├── Dockerfile
└── model/
    ├── random_forest_model.pkl   # Trained Random Forest (100 trees)
    ├── scaler.pkl                # StandardScaler fitted on training data only
    └── feature_columns.pkl       # Ordered list of 10 feature names
```

---

## Connecting to the Node.js Backend

```javascript
// services/aquasentinel.js
const axios = require('axios');

async function assessWaterQuality(sensorData) {
  const response = await axios.post(
    'https://aquasentinel-api-production.up.railway.app/api/predict',
    {
      temperature: sensorData.temperature,   // °C
      ph:          sensorData.ph,
      nitrite:     sensorData.nitrite,       // mg/L
      phosphorus:  sensorData.phosphorus,    // mg/L
    }
  );
  return response.data;
}
```

---

## Changelog

### v3.0.0 — Current
- **Complete pivot** — Water Quality Assessment replaces Algal Bloom Risk
- New dataset: AWD (Veeramsetty et al., 2024) — 3,887 aquaculture samples
- New parameters: Temperature, pH, Nitrite, Phosphorus (4 → direct RS485 readings)
- New labels: Excellent / Good / Poor (replaces Low / Medium / High Risk)
- 10 engineered features from 4 raw readings — no conversion layer needed
- Poor Precision = 1.00 — zero false alarms on critical water quality alerts
- Database schema updated — WaterQualityAssessment table

### v2.1.0
- database.py connected — predictions persisted across restarts
- Input validation with Field bounds
- AMMONIA_SPIKE_THRESHOLD corrected to 23.8816
- Error messages sanitised

### v2.0.0
- Hardware-aligned parameters — RS485 conversion layer introduced
- Rolling window buffers for temporal features

### v1.0.0
- Initial deployment — algal bloom risk prediction
- Random Forest on IoTPond6 dataset (89,283 samples)

---

## References

- Veeramsetty, V., Arabelli, R., Bernatin, T. (2024). *Aquaculture Water Quality Dataset.* Mendeley Data, V1. DOI: 10.17632/y78ty2g293.1
- Boyd, C.E. (1998). *Water Quality in Ponds for Aquaculture.* Birmingham Publishing.
- FAO (2015). *Aquaculture Development: Water Quality for Pond Aquaculture.*

---

## Team

| No. | Name | Registration No. | Access No. |
|-----|------|-----------------|------------|
| 1 | Tendo Calvin | S23B23/013 | B24247 |
| 2 | Ezamamti Ronald Austine | S23B23/018 | B24252 |
| 3 | Kisa Emmanuel | S23B23/028 | B24259 |

**Institution:** Uganda Christian University
**Project:** AquaSentinel — AI-Powered Fish Pond Water Quality Assessment
**Year:** 2025/2026
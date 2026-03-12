# AquaSentinel API

> Real-time algal bloom risk prediction for aquaculture pond monitoring — powered by IoT sensors and Machine Learning.

AquaSentinel is an AI-powered early warning system that predicts algal bloom risk in fish ponds using real IoT sensor data. Built on a Random Forest model trained on 89,283 real sensor readings from a monitored aquaculture pond, it classifies pond conditions into three risk levels with **97% accuracy** and a **0.97 High Risk Recall** — meaning it catches 97 out of every 100 genuine bloom emergencies.

---

## Live API

| | |
|---|---|
| **Base URL** | https://aquasentinel-api-production.up.railway.app |
| **Interactive Docs** | https://aquasentinel-api-production.up.railway.app/docs |
| **Version** | 2.0.0 |
| **Status** | 🟢 Online |

---

## The Problem

Algal blooms are the leading cause of fish stock loss in East African aquaculture ponds. Farms lose up to 40% of annual stock to overnight water quality crashes that could have been prevented with early detection. No affordable, automated early-warning system existed for smallholder fish farmers in Uganda — AquaSentinel was built to solve that.

---

## How It Works

```
IoT Hardware (RS485 Multi-Probe + DS18B20 + GPIO)
        ↓  sends 7 sensor readings every 5 minutes
AquaSentinel API (FastAPI on Railway)
        ↓  converts hardware readings to 21 model features
Random Forest Model (trained on 89,283 real pond readings)
        ↓  classifies bloom risk
Pond Manager receives LOW / MEDIUM / HIGH RISK alert + recommended action
```

The API accepts raw hardware sensor values and handles all unit conversion internally — including pH-adjusted Total Nitrogen fractionation (Boyd 1998), EC-to-Nitrate ionic conversion (Rhoades 1989), and Dissolved Oxygen estimation via Henry's Law solubility (APHA 2017).

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | System status and version |
| `GET` | `/api/health` | Model health check — confirms 21 features loaded |
| `POST` | `/api/predict` | **Main endpoint** — predict bloom risk from sensor readings |
| `GET` | `/api/history` | Last 50 predictions with timestamps |
| `GET` | `/api/stats` | Aggregated risk level counts |

---

## Example Request

`POST /api/predict`

The request body accepts your **actual hardware sensor parameters** directly. The API converts them internally to the 21 features the model expects.

```json
{
  "total_nitrogen": 45.0,
  "ec": 850.0,
  "ph": 7.4,
  "temperature": 26.5,
  "turbidity_high": false,
  "phosphorus": 12.0,
  "potassium": 8.0
}
```

### Parameter Reference

| Parameter | Unit | Source | Required |
|-----------|------|--------|----------|
| `total_nitrogen` | mg/kg | RS485 Multi-Probe | ✅ |
| `ec` | µS/cm | RS485 Multi-Probe | ✅ |
| `ph` | pH (0–14) | RS485 Multi-Probe | ✅ |
| `temperature` | °C | RS485 + DS18B20 | ✅ |
| `turbidity_high` | boolean | GPIO Digital Sensor | ✅ |
| `phosphorus` | mg/kg | RS485 Multi-Probe | Optional (logged) |
| `potassium` | mg/kg | RS485 Multi-Probe | Optional (logged) |
| `dissolved_oxygen` | g/ml | Future DO sensor | Optional (estimated if absent) |

---

## Example Response

```json
{
  "risk_level": 1,
  "risk_label": "MEDIUM RISK",
  "icon": "yellow",
  "severity": "medium",
  "confidence": 39.32,
  "action": "Elevated risk detected. Increase aeration and reduce feeding by 30%. Test water manually. Monitor every 30 minutes.",
  "probabilities": {
    "low_risk": 24.85,
    "medium_risk": 39.32,
    "high_risk": 35.83
  },
  "model_inputs": {
    "ammonia_g_ml": 0.00675,
    "nitrate_g_ml": 0.340413,
    "ph": 7.4,
    "temperature_c": 26.5,
    "turbidity_ntu": 20.0,
    "dissolved_oxygen_g_ml": 0.008087
  },
  "hardware_inputs": {
    "total_nitrogen_mg_kg": 45.0,
    "ec_us_cm": 850.0,
    "ph": 7.4,
    "temperature_c": 26.5,
    "turbidity_digital": "LOW",
    "phosphorus_mg_kg": 12.0,
    "potassium_mg_kg": 8.0
  },
  "data_quality": {
    "do_source": "estimated (Henry Law + pH correction)",
    "ammonia_source": "pH-adjusted TN fractionation (Boyd 1998)",
    "nitrate_source": "blended EC + TN estimate (55% EC / 45% TN)",
    "turbidity_source": "digital proxy (High=100 NTU, Low=20 NTU)",
    "rolling_window_size": 1,
    "rolling_std_active": false
  }
}
```

---

## Risk Levels

| Level | Label | Confidence Indicator | Recommended Action |
|-------|-------|---------------------|-------------------|
| `0` | 🟢 LOW RISK | Normal probability distribution | Routine monitoring. All parameters within safe range. |
| `1` | 🟡 MEDIUM RISK | Elevated nutrient or pH readings | Increase aeration, reduce feeding by 30%, monitor every 30 minutes. |
| `2` | 🔴 HIGH RISK | Critical parameter threshold exceeded | **Immediate intervention.** Stop feeding, maximum aeration, consider 20% water exchange. Alert pond manager now. |

> **Note on confidence:** Low confidence (e.g. 39%) does not indicate a system fault — it means pond parameters are genuinely borderline and the model is reporting honest uncertainty. A pond with mixed signals (low nitrogen but elevated EC) will correctly produce a split probability distribution.

---

## Hardware Conversion Layer

The API bridges the gap between what our RS485 Multi-Probe measures and what the Random Forest model was trained on:

| Hardware Measurement | Conversion Method | Reference |
|---------------------|------------------|-----------|
| Total Nitrogen → Ammonia | pH-adjusted TAN fractionation (15–25% of TN) | Boyd (1998) |
| EC → Nitrate | Blended: 55% ionic (EC × 0.7 / 1000) + 45% TN-derived | Rhoades et al. (1989) |
| GPIO High/Low → NTU | High = 100 NTU, Low = 20 NTU | Training data distribution |
| DO (absent) → DO (g/ml) | Henry's Law: 14.62 − 0.3898T + 0.006969T² − 0.0000590T³ + pH correction | APHA (2017) |

All conversion methods and data quality flags are returned transparently in every API response under `data_quality`.

---

## Model Performance

### Three-Model Comparison

| Metric | Logistic Regression | Decision Tree | **Random Forest** |
|--------|--------------------|--------------|--------------------|
| Overall Accuracy | ~72% | ~93% | **97%** |
| Macro F1 Score | 0.65 | 0.91 | **0.96** |
| CV Mean F1 (5-fold) | 0.653 | 0.912 | **0.952** |
| CV Standard Deviation | 0.006 | 0.003 | **0.003** |
| Train/Test Gap | < 0.03 | < 0.02 | **< 0.01** |
| High Risk Recall | ~0.70 | ~0.92 | **0.97** |
| High Risk Precision | ~0.43 | ~0.90 | **0.93** |
| High Risk F1 | ~0.54 | ~0.91 | **0.95** |

### Published Benchmark Comparison

| Study | Year | Best Accuracy | Method |
|-------|------|--------------|--------|
| Hafeez et al., *Remote Sensing* | 2019 | 89–94% | Random Forest, water quality |
| Zhu et al., *Ecological Indicators* | 2020 | 85–96% | RF + XGBoost, engineered labels |
| **AquaSentinel (this project)** | 2025 | **97%** | **Random Forest, IoT aquaculture, Uganda** |

### Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=30,
    max_features='sqrt',
    class_weight='balanced',   # handles 61/26/13% imbalance without SMOTE
    random_state=42,
    n_jobs=-1
)
```

### Top Feature Importances

| Rank | Feature | Importance | Tier |
|------|---------|-----------|------|
| 1 | Ammonia(g/ml) | ~13.0% | Critical |
| 2 | ammonia_ph_interaction | ~11.8% | Critical |
| 3 | ammonia_log | ~11.4% | Critical |
| 4 | Nitrate(g/ml) | ~10.8% | Important |
| 5 | ammonia_rolling_mean | ~8.5% | Important |

Top 3 features account for 36% of predictive power. Top 9 account for 90%.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | IoTPond6.csv — real IoT aquaculture pond sensors |
| Location | Uganda |
| Period | June – October 2021 |
| Raw rows | 91,050 |
| Clean rows | 89,283 |
| Features engineered | 21 (from 6 raw sensor columns) |
| Labels | Engineered via multi-parameter scoring function (Boyd 1998, FAO 2015) |
| Training set | ~71,426 samples (80%) |
| Test set | ~17,857 samples (20%) |
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

API will be available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| ML Model | Random Forest — scikit-learn 1.6.1 |
| API Framework | FastAPI 0.110.0 |
| Server | Uvicorn |
| Database | SQLite via SQLAlchemy |
| Deployment | Railway via Docker |
| Version Control | GitHub |
| Hardware | RS485 Multi-Probe, DS18B20 temperature sensor, GPIO digital turbidity sensor |

---

## Project Structure

```
aquasentinel-api/
├── main.py              # FastAPI app — endpoints and request/response models
├── predict.py           # Hardware conversion layer + Random Forest inference
├── database.py          # SQLAlchemy models and prediction logging
├── requirements.txt
├── Dockerfile
└── model/
    ├── random_forest_model.pkl   # Trained Random Forest (100 trees)
    ├── scaler.pkl                # StandardScaler fitted on training data only
    └── feature_columns.pkl       # Ordered list of 21 feature names
```

---

## Connecting to the Node.js Backend

```javascript
// services/aquasentinel.js
const axios = require('axios');

async function predictBloomRisk(sensorData) {
  const response = await axios.post(
    'https://aquasentinel-api-production.up.railway.app/api/predict',
    {
      total_nitrogen:  sensorData.total_nitrogen,   // mg/kg
      ec:              sensorData.ec,               // µS/cm
      ph:              sensorData.ph,
      temperature:     sensorData.temperature,      // °C
      turbidity_high:  sensorData.turbidity_high,   // boolean
      phosphorus:      sensorData.phosphorus ?? 0.0,
      potassium:       sensorData.potassium   ?? 0.0,
    }
  );
  return response.data;
}
```

---

## References

- Boyd, C.E. (1998). *Water Quality in Ponds for Aquaculture.* Birmingham Publishing.
- FAO (2015). *Aquaculture Development: Water Quality for Pond Aquaculture.*
- Zhu, M. et al. (2020). Machine Learning for the Water Quality Evaluation. *Ecological Indicators*, 115, 106426.
- Hafeez, S. et al. (2019). Comparison of Machine Learning Algorithms for Water Quality Predictions. *Remote Sensing*, 11(18), 2163.
- Rhoades, J.D. et al. (1989). *Soil Salinity Assessment.* FAO Irrigation and Drainage Paper 57.
- APHA (2017). *Standard Methods for the Examination of Water and Wastewater.* 23rd Edition.

---

## Team

| No. | Name | Registration No. | Access No. |
|-----|------|-----------------|------------|
| 1 | Tendo Calvin | S23B23/013 | B24247 |
| 2 | Ezamamti Ronald Austine | S23B23/018 | B24252 |
| 3 | Kisa Emmanuel | S23B23/028 | B24259 |

**Institution:** Uganda Christian University  
**Project:** AquaSentinel — AI-Powered Algal Bloom Risk Prediction System  
**Year:** 2025/2026

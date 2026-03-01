# AquaSentinel API

> Real-time algal bloom risk prediction for aquaculture pond monitoring.

AquaSentinel is an AI-powered early warning system that predicts algal bloom
risk in fish ponds using IoT sensor data. Built on a Random Forest model
trained on 89,283 real sensor readings, it classifies pond conditions into
three risk levels with 99.9% accuracy.

## Live API

- Base URL: https://aquasentinel-api-production.up.railway.app
- Docs: https://aquasentinel-api-production.up.railway.app/docs

## The Problem

Algal blooms are the leading cause of fish stock loss in East African
aquaculture ponds. Farms lose up to 40% of annual stock to water quality
events that could have been prevented with early detection.

## How It Works

IoT Sensors send readings every 5 minutes to the API.
The feature engineering pipeline processes 21 features.
The Random Forest model classifies bloom risk.
Pond managers receive actionable alerts immediately.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | API status |
| GET | /api/health | Model health check |
| POST | /api/predict | Predict bloom risk |
| GET | /api/history | Prediction history |
| GET | /api/stats | Summary statistics |

## Example Request

POST /api/predict
```json
{
  "temperature": 25.5,
  "turbidity": 85.0,
  "ph": 7.2,
  "ammonia": 12.0,
  "nitrate": 450.0,
  "dissolved_oxygen": 5.5
}
```

## Example Response
```json
{
  "risk_level": 1,
  "risk_label": "MEDIUM RISK",
  "confidence": 99.0,
  "action": "Elevated risk detected. Increase aeration and reduce feeding by 30%.",
  "timestamp": "2026-03-01T05:08:51.448136"
}
```

## Risk Levels

| Level | Label | Action |
|-------|-------|--------|
| 0 | LOW RISK | Routine monitoring |
| 1 | MEDIUM RISK | Increase aeration, reduce feeding |
| 2 | HIGH RISK | Immediate intervention required |

## Model Performance

| Model | Errors | High Risk Missed | False Alarms |
|-------|--------|-----------------|--------------|
| Logistic Regression | 671 | 31 | 618 |
| Decision Tree | 8 | 2 | 2 |
| Random Forest | 6 | 3 | 0 |

- Training samples: 89,283
- Test samples: 17,857
- Features: 21
- Top predictor: Ammonia at 13.03% importance

## Run Locally
```bash
git clone https://github.com/tendocalvin1/aquasentinel-api.git
cd aquasentinel-api
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| ML Model | Random Forest scikit-learn 1.6.1 |
| API Framework | FastAPI 0.110.0 |
| Database | SQLite via SQLAlchemy |
| Deployment | Railway via Docker |
| Version Control | GitHub |

## Project Structure
```
aquasentinel-api/
├── main.py
├── predict.py
├── database.py
├── requirements.txt
├── Dockerfile
└── model/
    ├── random_forest_model.pkl
    ├── scaler.pkl
    └── feature_columns.pkl
```

## Author

## Team

This project was built as a collaborative academic and engineering effort.

| No. | Name | Registration No. | Access No. |
|-----|------|-----------------|------------|
| 1 | Ezamamti Ronald Austine | S23B23/018 | B24252 |
| 2 | Kisa Emmanuel | S23B23/028 | B24259 |
| 3 | Tendo Calvin | S23B23/013 | B24247 |

**Institution:** Uganda Christian University 
**Project:** AquaSentinel — AI-Powered Algal Bloom Risk Prediction System  
**Year:** 2026
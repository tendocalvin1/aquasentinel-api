from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional

from predict import predict_bloom_risk
from database import (
    SessionLocal, create_tables,
    save_prediction, PredictionRecord
)

app = FastAPI(
    title="AquaSentinel API",
    description="Real-time algal bloom risk prediction for aquaculture pond monitoring.",
    version="1.0.0"
)

create_tables()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class SensorReading(BaseModel):
    temperature:      float = Field(..., ge=0,  le=50,   description="Temperature in Celsius")
    turbidity:        float = Field(..., ge=0,  le=1000, description="Turbidity in NTU")
    ph:               float = Field(..., ge=0,  le=14,   description="pH level")
    ammonia:          float = Field(..., ge=0,           description="Ammonia in g/ml")
    nitrate:          float = Field(..., ge=0,           description="Nitrate in g/ml")
    dissolved_oxygen: float = Field(..., ge=0,           description="Dissolved Oxygen in g/ml")
    hour:             Optional[int] = Field(default=12, ge=0, le=23)
    month:            Optional[int] = Field(default=6,  ge=1, le=12)
    day_of_week:      Optional[int] = Field(default=2,  ge=0, le=6)
    season:           Optional[int] = Field(default=0,  ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 25.5,
                "turbidity":   85.0,
                "ph":          7.2,
                "ammonia":     12.0,
                "nitrate":     450.0,
                "dissolved_oxygen": 5.5
            }
        }


@app.get("/")
def root():
    return {
        "product": "AquaSentinel",
        "version": "1.0.0",
        "status":  "online",
        "message": "Algal bloom risk prediction API"
    }


@app.get("/api/health")
def health_check():
    return {
        "status":    "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model":     "Random Forest v1.0",
        "features":  21
    }


@app.post("/api/predict")
def predict(
    reading: SensorReading,
    db: Session = Depends(get_db)
):
    try:
        result = predict_bloom_risk(
            temperature      = reading.temperature,
            turbidity        = reading.turbidity,
            ph               = reading.ph,
            ammonia          = reading.ammonia,
            nitrate          = reading.nitrate,
            dissolved_oxygen = reading.dissolved_oxygen,
            hour             = reading.hour,
            month            = reading.month,
            day_of_week      = reading.day_of_week,
            season           = reading.season
        )

        save_prediction(
            db,
            sensor_data = reading.model_dump(),
            result      = result
        )

        result['timestamp'] = datetime.utcnow().isoformat()
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/api/history")
def get_history(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    records = db.query(PredictionRecord)\
                .order_by(PredictionRecord.timestamp.desc())\
                .limit(limit)\
                .all()

    return {
        "count": len(records),
        "predictions": [
            {
                "id":         r.id,
                "timestamp":  r.timestamp.isoformat(),
                "risk_level": r.risk_level,
                "risk_label": r.risk_label,
                "confidence": r.confidence,
                "severity":   r.severity,
                "ammonia":    r.ammonia,
                "nitrate":    r.nitrate,
                "ph":         r.ph
            }
            for r in records
        ]
    }


@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(PredictionRecord).count()
    high  = db.query(PredictionRecord)\
              .filter(PredictionRecord.risk_level == 2).count()
    med   = db.query(PredictionRecord)\
              .filter(PredictionRecord.risk_level == 1).count()
    low   = db.query(PredictionRecord)\
              .filter(PredictionRecord.risk_level == 0).count()

    return {
        "total_predictions": total,
        "by_risk_level": {
            "high_risk":   high,
            "medium_risk": med,
            "low_risk":    low
        },
        "high_risk_percentage": round(
            (high / total * 100) if total > 0 else 0, 2
        )
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
"""
AquaSentinel API — main.py
Hardware-aligned FastAPI endpoint for the fish pond management system.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import uvicorn

from predict import predict_bloom_risk

app = FastAPI(
    title="AquaSentinel API",
    description="Algal bloom risk prediction for fish pond monitoring",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request model — matches your exact hardware parameters ─────────────────
class SensorReading(BaseModel):
    # RS485 Multi-Probe
    total_nitrogen:  float = Field(..., description="Total Nitrogen in mg/kg",  example=45.0)
    ec:              float = Field(..., description="Electrical Conductivity µS/cm", example=850.0)
    ph:              float = Field(..., description="pH value 0–14",             example=7.4)
    phosphorus:      float = Field(0.0,  description="Phosphorus mg/kg (logged)", example=12.0)
    potassium:       float = Field(0.0,  description="Potassium mg/kg (logged)",  example=8.0)
    # DS18B20
    temperature:     float = Field(..., description="Water temperature °C",      example=26.5)
    # GPIO Digital
    turbidity_high:  bool  = Field(..., description="True=cloudy, False=clear",  example=False)
    # Optional — pass when DO sensor is added
    dissolved_oxygen: Optional[float] = Field(None, description="DO g/ml (optional)")


# In-memory prediction log (last 50)
prediction_log = []


@app.get("/")
def root():
    return {
        "system": "AquaSentinel",
        "status": "online",
        "version": "2.0.0",
        "hardware": "RS485 Multi-Probe + DS18B20 + GPIO Turbidity"
    }


@app.get("/api/health")
def health():
    from predict import model, feature_cols
    return {
        "status":    "healthy",
        "model":     "Random Forest",
        "features":  len(feature_cols),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/predict")
def predict(reading: SensorReading):
    try:
        result = predict_bloom_risk(
            total_nitrogen   = reading.total_nitrogen,
            ec               = reading.ec,
            ph               = reading.ph,
            temperature      = reading.temperature,
            turbidity_high   = reading.turbidity_high,
            phosphorus       = reading.phosphorus,
            potassium        = reading.potassium,
            dissolved_oxygen = reading.dissolved_oxygen,
        )

        # Log it
        prediction_log.append({
            "timestamp":   datetime.now().isoformat(),
            "risk_label":  result["risk_label"],
            "confidence":  result["confidence"],
            "hardware":    reading.dict()
        })
        if len(prediction_log) > 50:
            prediction_log.pop(0)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
def history():
    return {"count": len(prediction_log), "predictions": prediction_log[-50:]}


@app.get("/api/stats")
def stats():
    if not prediction_log:
        return {"total": 0}
    levels = [p["risk_label"] for p in prediction_log]
    return {
        "total":       len(levels),
        "low_risk":    levels.count("LOW RISK"),
        "medium_risk": levels.count("MEDIUM RISK"),
        "high_risk":   levels.count("HIGH RISK"),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
"""
AquaSentinel API — main.py  (v3.0.0)
=====================================
Water Quality Assessment API — FastAPI on Railway

CHANGES IN v3.0.0:
    - Complete rewrite — water quality assessment replaces bloom risk
    - SensorReading accepts 4 hardware parameters: temperature, ph, nitrite, phosphorus
    - Response returns quality_level (0/1/2), quality_label (Excellent/Good/Poor)
    - No hardware conversion layer — RS485 measures what model expects directly
    - Database updated — stores water quality labels instead of risk levels
    - All endpoint descriptions updated to reflect water quality context
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import uvicorn

from predict import predict_water_quality, model, feature_cols
from database import (
    Base, engine, SessionLocal,
    save_prediction, get_history, get_stats,
)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("aquasentinel.main")

# ── Create database tables if they don't exist ────────────────────────────
Base.metadata.create_all(bind=engine)
logger.info("Database tables ready.")

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="AquaSentinel API",
    description=(
        "Real-time water quality assessment for fish pond monitoring. "
        "Accepts RS485 Multi-Probe + DS18B20 sensor readings and "
        "returns Excellent / Good / Poor water quality classification."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# REQUEST MODEL
# ═══════════════════════════════════════════════════════════════════════════

class SensorReading(BaseModel):
    """
    Four sensor readings from the RS485 Multi-Probe + DS18B20 hardware.
    These are the exact four parameters the model was trained on —
    no unit conversion is needed.
    """
    temperature: float = Field(
        ...,
        ge=0.0, le=50.0,
        description="Water temperature in °C (DS18B20 / RS485)",
        example=25.0,
    )
    ph: float = Field(
        ...,
        ge=0.0, le=14.0,
        description="pH value 0–14 (RS485 pH probe)",
        example=7.2,
    )
    nitrite: float = Field(
        ...,
        ge=0.0, le=20.0,
        description="Nitrite concentration in mg/L (RS485)",
        example=0.1,
    )
    phosphorus: float = Field(
        ...,
        ge=0.0, le=20.0,
        description="Phosphorus concentration in mg/L (RS485)",
        example=0.05,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "system":      "AquaSentinel",
        "description": "Fish Pond Water Quality Assessment",
        "status":      "online",
        "version":     "3.0.0",
        "model":       "Random Forest — AWD Dataset (3,887 samples)",
        "hardware":    "RS485 Multi-Probe + DS18B20",
        "labels":      {
            "0": "Excellent",
            "1": "Good",
            "2": "Poor",
        },
    }


@app.get("/api/health")
def health():
    """
    Model health check.
    Confirms the Random Forest is loaded and reports feature count.
    """
    return {
        "status":           "healthy",
        "model":            "Random Forest — Water Quality",
        "features":         len(feature_cols),
        "feature_list":     feature_cols,
        "labels":           ["Excellent", "Good", "Poor"],
        "model_accuracy":   "85.5%",
        "timestamp":        datetime.now().isoformat(),
    }


@app.post("/api/predict")
def predict(reading: SensorReading):
    """
    Main prediction endpoint — Water Quality Assessment.

    Accepts four RS485 sensor readings and returns a complete
    water quality assessment: Excellent, Good, or Poor.

    No unit conversion required — the RS485 Multi-Probe measures
    exactly the parameters the model was trained on.
    """
    try:
        result = predict_water_quality(
            temperature=reading.temperature,
            ph=reading.ph,
            nitrite=reading.nitrite,
            phosphorus=reading.phosphorus,
        )

        # ── Persist to database ──────────────────────────────────────────
        db = SessionLocal()
        try:
            save_prediction(db, reading=reading.dict(), result=result)
        except Exception as db_err:
            logger.error("DB write failed: %s", db_err)
        finally:
            db.close()

        return result

    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please check your sensor input values.",
        )


@app.get("/api/history")
def history(limit: int = 50):
    """
    Returns the last N water quality assessments from the database.
    Persists across container restarts.
    """
    db = SessionLocal()
    try:
        records = get_history(db, limit=limit)
        return {"count": len(records), "assessments": records}
    except Exception as e:
        logger.error("History query failed: %s", e)
        raise HTTPException(status_code=500, detail="Could not retrieve history.")
    finally:
        db.close()


@app.get("/api/stats")
def stats():
    """
    Returns aggregated water quality counts from the database.
    Persists across container restarts.
    """
    db = SessionLocal()
    try:
        return get_stats(db)
    except Exception as e:
        logger.error("Stats query failed: %s", e)
        raise HTTPException(status_code=500, detail="Could not retrieve stats.")
    finally:
        db.close()


# ── Local development entry point ────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
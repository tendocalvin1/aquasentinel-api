"""
AquaSentinel API — main.py  (v2.1.0)
======================================
Hardware-aligned FastAPI endpoint for the fish pond management system.

CHANGES IN v2.1.0:
    - database.py connected — predictions now persist to SQLite across restarts
    - Field bounds added to SensorReading — rejects physically impossible inputs
    - model and feature_cols imported at top level — health endpoint no longer
      re-imports on every request
    - Error messages sanitised — internal paths no longer exposed to callers
    - Startup event seeds rolling buffers from last 6 DB records after restart
    - Structured logging added throughout
    - /api/history and /api/stats now read from database, not in-memory list
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
import uvicorn

from predict import predict_bloom_risk, seed_rolling_buffers, model, feature_cols
from database import (
    Base, engine, SessionLocal,
    save_prediction, get_history, get_stats, get_last_n_model_inputs,
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
        "Real-time algal bloom risk prediction for fish pond monitoring. "
        "Accepts RS485 Multi-Probe + DS18B20 + GPIO sensor readings and "
        "returns LOW / MEDIUM / HIGH risk classification."
    ),
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: seed rolling buffers from DB ────────────────────────────────
@app.on_event("startup")
def on_startup():
    """
    On every container start / Railway redeploy, restore the last 6
    ammonia / nitrate / temperature readings from the database so that
    rolling_std_active is True immediately rather than after 6 new readings.
    """
    try:
        last_six = get_last_n_model_inputs(6)
        if last_six:
            seed_rolling_buffers(last_six)
            logger.info("Rolling buffers seeded from DB (%d records).", len(last_six))
        else:
            logger.info("No DB history found — rolling buffers start empty.")
    except Exception as e:
        # Non-fatal — app still runs, buffers just start empty
        logger.warning("Could not seed rolling buffers: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
# REQUEST MODEL — matches your exact hardware parameters with bounds
# ═══════════════════════════════════════════════════════════════════════════

class SensorReading(BaseModel):
    # ── RS485 Multi-Probe ─────────────────────────────────────────────
    total_nitrogen: float = Field(
        ...,
        ge=0.0, le=2000.0,
        description="Total Nitrogen in mg/kg",
        example=45.0,
    )
    ec: float = Field(
        ...,
        ge=0.0, le=8000.0,
        description="Electrical Conductivity in µS/cm",
        example=850.0,
    )
    ph: float = Field(
        ...,
        ge=0.0, le=14.0,
        description="pH value 0–14",
        example=7.4,
    )
    phosphorus: float = Field(
        0.0,
        ge=0.0, le=1000.0,
        description="Phosphorus in mg/kg (logged for future model)",
        example=12.0,
    )
    potassium: float = Field(
        0.0,
        ge=0.0, le=1000.0,
        description="Potassium in mg/kg (logged for future model)",
        example=8.0,
    )
    # ── DS18B20 temperature sensor ────────────────────────────────────
    temperature: float = Field(
        ...,
        ge=10.0, le=40.0,
        description="Water temperature in °C",
        example=26.5,
    )
    # ── GPIO digital turbidity sensor ────────────────────────────────
    turbidity_high: bool = Field(
        ...,
        description="True = turbidity HIGH (cloudy), False = LOW (clear)",
        example=False,
    )
    # ── Optional DO sensor (add when hardware is available) ───────────
    dissolved_oxygen: Optional[float] = Field(
        None,
        ge=0.0, le=50.0,
        description="Dissolved Oxygen in g/ml — estimated if not provided",
        example=None,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "system":   "AquaSentinel",
        "status":   "online",
        "version":  "2.1.0",
        "hardware": "RS485 Multi-Probe + DS18B20 + GPIO Turbidity",
    }


@app.get("/api/health")
def health():
    """
    Model health check. Confirms the Random Forest is loaded and
    reports how many features it expects.
    model and feature_cols are imported at the top of this file —
    no re-import on every request.
    """
    return {
        "status":    "healthy",
        "model":     "Random Forest",
        "features":  len(feature_cols),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/predict")
def predict(reading: SensorReading):
    """
    Main prediction endpoint.

    Accepts your hardware sensor readings, converts them to the 21 model
    features, runs the Random Forest, and returns a complete risk assessment.

    Error messages are sanitised — internal file paths and stack traces
    are logged server-side but never returned to the caller.
    """
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

        # ── Persist to database ──────────────────────────────────────────
        db = SessionLocal()
        try:
            save_prediction(db, reading=reading.dict(), result=result)
        except Exception as db_err:
            # Non-fatal — return the prediction even if DB write fails
            logger.error("DB write failed: %s", db_err)
        finally:
            db.close()

        return result

    except Exception as e:
        # Log full detail server-side for debugging
        logger.error("Prediction error: %s", e, exc_info=True)
        # Return a clean message — never expose internals to the caller
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please check your sensor input values.",
        )


@app.get("/api/history")
def history(limit: int = 50):
    """
    Returns the last N predictions from the database.
    Data persists across restarts — reads from SQLite, not memory.
    """
    db = SessionLocal()
    try:
        records = get_history(db, limit=limit)
        return {"count": len(records), "predictions": records}
    except Exception as e:
        logger.error("History query failed: %s", e)
        raise HTTPException(status_code=500, detail="Could not retrieve history.")
    finally:
        db.close()


@app.get("/api/stats")
def stats():
    """
    Returns aggregated risk level counts from the database.
    Data persists across restarts — reads from SQLite, not memory.
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
"""
AquaSentinel — database.py  (v2.1.0)
======================================
SQLAlchemy models and database helpers.

Every prediction is now persisted to SQLite so that:
    - /api/history and /api/stats survive container restarts
    - Rolling buffers can be re-seeded from real history on startup
    - Phosphorus and Potassium readings accumulate for future model retrain
    - All hardware and model inputs are stored for audit and debugging

DATABASE FILE:
    aquasentinel.db  — created automatically in the project root on first run.
    Railway persists this file across deploys as long as the volume is mounted.
    To enable a persistent volume on Railway:
        Railway dashboard → your service → Settings → Volumes
        Mount path: /app  (or wherever your project root is)
"""

import logging
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, Boolean,
    String, DateTime, Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger("aquasentinel.database")

# ── Database setup ────────────────────────────────────────────────────────
DATABASE_URL = "sqlite:///./aquasentinel.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE MODEL
# ═══════════════════════════════════════════════════════════════════════════

class Prediction(Base):
    """
    One row per API call to /api/predict.

    Stores:
        - Timestamp
        - Raw hardware inputs exactly as received from the sensor
        - Converted model inputs (after conversion layer)
        - Prediction output (risk level, label, confidence)
        - Data quality flags
    """
    __tablename__ = "predictions"

    id          = Column(Integer, primary_key=True, index=True)
    timestamp   = Column(DateTime, default=datetime.utcnow, index=True)

    # ── Hardware inputs (RS485 + DS18B20 + GPIO) ─────────────────────
    total_nitrogen  = Column(Float,   nullable=False)
    ec              = Column(Float,   nullable=False)
    ph              = Column(Float,   nullable=False)
    temperature     = Column(Float,   nullable=False)
    turbidity_high  = Column(Boolean, nullable=False)
    phosphorus      = Column(Float,   default=0.0)    # logged for future retrain
    potassium       = Column(Float,   default=0.0)    # logged for future retrain

    # ── Converted model inputs (after conversion layer) ───────────────
    ammonia_g_ml          = Column(Float, nullable=False)
    nitrate_g_ml          = Column(Float, nullable=False)
    turbidity_ntu         = Column(Float, nullable=False)
    dissolved_oxygen_g_ml = Column(Float, nullable=False)
    do_was_estimated      = Column(Boolean, default=True)

    # ── Prediction output ─────────────────────────────────────────────
    risk_level    = Column(Integer, nullable=False)
    risk_label    = Column(String,  nullable=False)
    confidence    = Column(Float,   nullable=False)
    prob_low      = Column(Float,   nullable=False)
    prob_medium   = Column(Float,   nullable=False)
    prob_high     = Column(Float,   nullable=False)

    # ── Validation ────────────────────────────────────────────────────
    inputs_valid  = Column(Boolean, default=True)
    warnings      = Column(Text,    default="")     # JSON string if any warnings


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS  (called by main.py)
# ═══════════════════════════════════════════════════════════════════════════

def save_prediction(db, reading: dict, result: dict) -> Prediction:
    """
    Persist one prediction to the database.

    Parameters
    ----------
    db      : SQLAlchemy Session
    reading : dict — the raw SensorReading fields from the API request
    result  : dict — the full response from predict_bloom_risk()

    Returns
    -------
    The saved Prediction ORM object.
    """
    import json

    mi = result.get("model_inputs", {})
    dq = result.get("data_quality", {})
    vl = result.get("validation",   {})
    pb = result.get("probabilities", {})

    record = Prediction(
        timestamp   = datetime.utcnow(),

        # Hardware inputs
        total_nitrogen = reading.get("total_nitrogen", 0.0),
        ec             = reading.get("ec",             0.0),
        ph             = reading.get("ph",             0.0),
        temperature    = reading.get("temperature",    0.0),
        turbidity_high = reading.get("turbidity_high", False),
        phosphorus     = reading.get("phosphorus",     0.0),
        potassium      = reading.get("potassium",      0.0),

        # Converted model inputs
        ammonia_g_ml          = mi.get("ammonia_g_ml",          0.0),
        nitrate_g_ml          = mi.get("nitrate_g_ml",          0.0),
        turbidity_ntu         = mi.get("turbidity_ntu",         0.0),
        dissolved_oxygen_g_ml = mi.get("dissolved_oxygen_g_ml", 0.0),
        do_was_estimated      = dq.get("do_source", "") != "sensor",

        # Prediction
        risk_level  = result.get("risk_level",  0),
        risk_label  = result.get("risk_label",  ""),
        confidence  = result.get("confidence",  0.0),
        prob_low    = pb.get("low_risk",    0.0),
        prob_medium = pb.get("medium_risk", 0.0),
        prob_high   = pb.get("high_risk",   0.0),

        # Validation
        inputs_valid = vl.get("inputs_valid", True),
        warnings     = json.dumps(vl.get("warnings", [])),
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    logger.debug("Saved prediction id=%d  label=%s", record.id, record.risk_label)
    return record


def get_history(db, limit: int = 50) -> list:
    """
    Return the last N predictions, newest first.
    Called by /api/history.
    """
    rows = (
        db.query(Prediction)
        .order_by(Prediction.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":           r.id,
            "timestamp":    r.timestamp.isoformat(),
            "risk_label":   r.risk_label,
            "risk_level":   r.risk_level,
            "confidence":   r.confidence,
            "hardware": {
                "total_nitrogen": r.total_nitrogen,
                "ec":             r.ec,
                "ph":             r.ph,
                "temperature":    r.temperature,
                "turbidity_high": r.turbidity_high,
                "phosphorus":     r.phosphorus,
                "potassium":      r.potassium,
            },
            "model_inputs": {
                "ammonia_g_ml":          r.ammonia_g_ml,
                "nitrate_g_ml":          r.nitrate_g_ml,
                "turbidity_ntu":         r.turbidity_ntu,
                "dissolved_oxygen_g_ml": r.dissolved_oxygen_g_ml,
                "do_was_estimated":      r.do_was_estimated,
            },
        }
        for r in rows
    ]


def get_stats(db) -> dict:
    """
    Return aggregated counts per risk level.
    Called by /api/stats.
    """
    total  = db.query(Prediction).count()
    low    = db.query(Prediction).filter(Prediction.risk_level == 0).count()
    medium = db.query(Prediction).filter(Prediction.risk_level == 1).count()
    high   = db.query(Prediction).filter(Prediction.risk_level == 2).count()

    return {
        "total":       total,
        "low_risk":    low,
        "medium_risk": medium,
        "high_risk":   high,
        "high_risk_pct": round(high / total * 100, 1) if total > 0 else 0.0,
    }


def get_last_n_model_inputs(n: int = 6) -> list:
    """
    Return the last N converted model inputs for rolling buffer seeding.
    Called by main.py on_startup() via seed_rolling_buffers().

    Returns a list of dicts with keys:
        ammonia_g_ml, nitrate_g_ml, temperature_c
    ordered oldest → newest so buffers are filled in time order.
    """
    db   = SessionLocal()
    rows = (
        db.query(Prediction)
        .order_by(Prediction.timestamp.desc())
        .limit(n)
        .all()
    )
    db.close()

    # Reverse to oldest-first for correct deque insertion order
    return [
        {
            "ammonia_g_ml":  r.ammonia_g_ml,
            "nitrate_g_ml":  r.nitrate_g_ml,
            "temperature_c": r.temperature,
        }
        for r in reversed(rows)
    ]
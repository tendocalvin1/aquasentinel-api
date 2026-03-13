"""
AquaSentinel — database.py  (v3.0.0)
=====================================
SQLAlchemy models and helpers for Water Quality Assessment.

Every prediction is persisted to SQLite so that:
    - /api/history and /api/stats survive container restarts
    - All sensor readings accumulate as a field deployment dataset
    - Water quality trends are trackable over time

DATABASE FILE:
    aquasentinel.db — created automatically in the project root on first run.
    For Railway persistence, mount a volume at /app in Railway settings.
"""

import logging
import json
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float,
    String, DateTime, Boolean, Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger("aquasentinel.database")

# ── Database setup ────────────────────────────────────────────────────────
DATABASE_URL = "sqlite:///./aquasentinel.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE MODEL
# ═══════════════════════════════════════════════════════════════════════════

class WaterQualityAssessment(Base):
    """
    One row per call to POST /api/predict.

    Stores raw sensor inputs, prediction result, and confidence.
    Each row is one water quality snapshot from the pond.
    """
    __tablename__ = "water_quality_assessments"

    id           = Column(Integer, primary_key=True, index=True)
    timestamp    = Column(DateTime, default=datetime.utcnow, index=True)

    # ── Sensor inputs (RS485 + DS18B20) ──────────────────────────────
    temperature  = Column(Float, nullable=False)   # °C
    ph           = Column(Float, nullable=False)   # pH
    nitrite      = Column(Float, nullable=False)   # mg/L
    phosphorus   = Column(Float, nullable=False)   # mg/L

    # ── Prediction output ─────────────────────────────────────────────
    quality_level  = Column(Integer, nullable=False)   # 0=Excellent 1=Good 2=Poor
    quality_label  = Column(String,  nullable=False)   # "EXCELLENT" / "GOOD" / "POOR"
    confidence     = Column(Float,   nullable=False)   # %
    prob_excellent = Column(Float,   nullable=False)
    prob_good      = Column(Float,   nullable=False)
    prob_poor      = Column(Float,   nullable=False)

    # ── Validation flags ──────────────────────────────────────────────
    inputs_valid   = Column(Boolean, default=True)
    warnings       = Column(Text,    default="")    # JSON string


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def save_prediction(db, reading: dict, result: dict) -> WaterQualityAssessment:
    """
    Persist one water quality assessment to the database.

    Parameters
    ----------
    db      : SQLAlchemy Session
    reading : dict — raw SensorReading fields from the API request
    result  : dict — full response from predict_water_quality()
    """
    vl = result.get("validation",   {})
    pb = result.get("probabilities", {})

    record = WaterQualityAssessment(
        timestamp      = datetime.utcnow(),
        temperature    = reading.get("temperature", 0.0),
        ph             = reading.get("ph",          0.0),
        nitrite        = reading.get("nitrite",      0.0),
        phosphorus     = reading.get("phosphorus",   0.0),
        quality_level  = result.get("quality_level", 0),
        quality_label  = result.get("quality_label", ""),
        confidence     = result.get("confidence",    0.0),
        prob_excellent = pb.get("excellent", 0.0),
        prob_good      = pb.get("good",      0.0),
        prob_poor      = pb.get("poor",      0.0),
        inputs_valid   = vl.get("inputs_valid", True),
        warnings       = json.dumps(vl.get("warnings", [])),
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    logger.debug(
        "Saved assessment id=%d  label=%s  confidence=%.1f%%",
        record.id, record.quality_label, record.confidence,
    )
    return record


def get_history(db, limit: int = 50) -> list:
    """
    Return the last N assessments, newest first.
    Called by GET /api/history.
    """
    rows = (
        db.query(WaterQualityAssessment)
        .order_by(WaterQualityAssessment.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":            r.id,
            "timestamp":     r.timestamp.isoformat(),
            "quality_label": r.quality_label,
            "quality_level": r.quality_level,
            "confidence":    r.confidence,
            "sensor_inputs": {
                "temperature_c":   r.temperature,
                "ph":              r.ph,
                "nitrite_mg_l":    r.nitrite,
                "phosphorus_mg_l": r.phosphorus,
            },
            "probabilities": {
                "excellent": r.prob_excellent,
                "good":      r.prob_good,
                "poor":      r.prob_poor,
            },
        }
        for r in rows
    ]


def get_stats(db) -> dict:
    """
    Return aggregated counts per water quality level.
    Called by GET /api/stats.
    """
    total     = db.query(WaterQualityAssessment).count()
    excellent = db.query(WaterQualityAssessment).filter(
        WaterQualityAssessment.quality_level == 0).count()
    good      = db.query(WaterQualityAssessment).filter(
        WaterQualityAssessment.quality_level == 1).count()
    poor      = db.query(WaterQualityAssessment).filter(
        WaterQualityAssessment.quality_level == 2).count()

    return {
        "total":          total,
        "excellent":      excellent,
        "good":           good,
        "poor":           poor,
        "poor_pct":       round(poor / total * 100, 1) if total > 0 else 0.0,
        "excellent_pct":  round(excellent / total * 100, 1) if total > 0 else 0.0,
    }

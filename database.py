from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./aquasentinel.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id               = Column(Integer, primary_key=True, index=True)
    timestamp        = Column(DateTime, default=datetime.utcnow)
    temperature      = Column(Float)
    turbidity        = Column(Float)
    ph               = Column(Float)
    ammonia          = Column(Float)
    nitrate          = Column(Float)
    dissolved_oxygen = Column(Float)
    risk_level       = Column(Integer)
    risk_label       = Column(String)
    confidence       = Column(Float)
    severity         = Column(String)


def create_tables():
    Base.metadata.create_all(bind=engine)


def save_prediction(db_session, sensor_data: dict, result: dict):
    record = PredictionRecord(
        temperature      = sensor_data['temperature'],
        turbidity        = sensor_data['turbidity'],
        ph               = sensor_data['ph'],
        ammonia          = sensor_data['ammonia'],
        nitrate          = sensor_data['nitrate'],
        dissolved_oxygen = sensor_data['dissolved_oxygen'],
        risk_level       = result['risk_level'],
        risk_label       = result['risk_label'],
        confidence       = result['confidence'],
        severity         = result['severity']
    )
    db_session.add(record)
    db_session.commit()
    return record
import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config.settings import DB_URL

Base = declarative_base()

class RecognitionEvent(Base):
    """
    Model storing a single face recognition event.
    """
    __tablename__ = "recognition_events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    source = Column(String(50), nullable=False) # 'camera' or 'upload'
    
    person_name = Column(String(100), nullable=False) # Identity matched, or 'Unknown'
    confidence = Column(Float, nullable=False)        # Matching confidence (e.g., 0.85) => 1.0 - distance
    
    report_text = Column(Text, nullable=True)         # Human readable summary
    report_json = Column(Text, nullable=True)         # JSON format info

    image_path = Column(String(255), nullable=True)   # Optional: Path where the frame/upload is stored

# Initialize the database engine
engine = create_engine(
    DB_URL, 
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def log_recognition_event(source: str, person_name: str, confidence: float, report_text: str, report_json: str, image_path: str = ""):
    """Helper method to easily log an event into the DB."""
    db = SessionLocal()
    try:
        event = RecognitionEvent(
            source=source,
            person_name=person_name,
            confidence=confidence,
            report_text=report_text,
            report_json=report_json,
            image_path=image_path
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return event
    finally:
        db.close()

def get_recent_events(limit=50):
    """Retrieve recent events from the database."""
    db = SessionLocal()
    try:
        return db.query(RecognitionEvent).order_by(RecognitionEvent.timestamp.desc()).limit(limit).all()
    finally:
        db.close()

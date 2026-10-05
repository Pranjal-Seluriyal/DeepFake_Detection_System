import os
os.environ.setdefault("DISABLE_SQLALCHEMY_CEXT", "1")
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    prediction_type = db.Column(db.String(20), nullable=False, default="image")  # 'image' or 'audio'
    original_filename = db.Column(db.String(255), nullable=False)
    media_url = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(50), nullable=False)  # 'Deepfake' or 'Real'
    confidence = db.Column(db.Float, nullable=False)    # e.g., 94.25
    raw_score = db.Column(db.Float, nullable=True)      # e.g., 0.9425
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "prediction_type": self.prediction_type,
            "original_filename": self.original_filename,
            "media_url": self.media_url,
            "status": self.status,
            "confidence": f"{self.confidence:.2f}%",
            "confidence_value": self.confidence,
            "raw_score": self.raw_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_at_display": self.created_at.strftime("%b %d, %Y - %H:%M UTC") if self.created_at else ""
        }

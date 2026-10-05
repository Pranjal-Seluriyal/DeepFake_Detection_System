import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", "deeptrust-ai-secret-key-change-in-production-2026")
    
    # Upload Settings
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", str(BASE_DIR / "uploads"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 32 * 1024 * 1024))  # 32 MB
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
    ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}
    
    # Database Settings
    # Production uses PostgreSQL via DATABASE_URL; fallback to local SQLite for local dev
    db_url = os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgres://"):
        # Fix for Heroku/Render postgres:// -> postgresql://
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    SQLALCHEMY_DATABASE_URI = db_url or f"sqlite:///{BASE_DIR / 'predictions.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Cloud Storage (Cloudinary)
    CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")
    CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
    
    # Hugging Face Model Repository Settings
    HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "PranjalSeluriyal/Deepfake_Detection")
    HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "model_best.pth")
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Model Settings (UCF-Net for Image, CNN+BiLSTM for Audio)
    IMAGE_MODEL_PATH = os.getenv("IMAGE_MODEL_PATH")
    UCFNET_CLIP_PATH = os.getenv("UCFNET_CLIP_PATH", str(BASE_DIR / "models" / "pretrained" / "clip-vit-large-patch14-local"))
    OLD_IMAGE_MODEL_PATH = str(BASE_DIR / "models" / "old_image_model_backup" / "image_model.pt")
    AUDIO_MODEL_PATH = os.getenv("AUDIO_MODEL_PATH", str(BASE_DIR / "models" / "audio_model.pth"))

    # Remote Backend Proxy (Optional - for Vercel Serverless hybrid deployment)
    BACKEND_API_URL = os.getenv("BACKEND_API_URL")

    # Startup Model Preloading
    PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "true").lower() in ("true", "1", "yes")
    
    # Decision Thresholds
    BEST_THRESH_IMAGE = float(os.getenv("BEST_THRESH_IMAGE", 0.50))
    BEST_THRESH_AUDIO = float(os.getenv("BEST_THRESH_AUDIO", 0.95))
    
    # Image Input Specs
    IMG_SIZE = 224

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig if os.getenv("FLASK_ENV") != "production" else ProductionConfig
}

import os
import logging
from flask import Flask
from flask_cors import CORS

from config import config
from database import db
from routes.api import api_bp
from routes.views import views_bp
from ml.inference import get_image_model, get_audio_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DeepTrustApp")

def create_app(config_name=None):
    """Application factory for DeepTrust AI."""
    if config_name is None:
        config_name = os.getenv("FLASK_ENV", "default")

    app = Flask(__name__)
    app_config = config.get(config_name, config["default"])
    app.config.from_object(app_config)

    # Enable Cross-Origin Resource Sharing
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize Database
    db.init_app(app)

    # Register Blueprints
    app.register_blueprint(views_bp)
    app.register_blueprint(api_bp)

    # Initialize tables and preload models once at startup
    with app.app_context():
        try:
            db.create_all()
            logger.info("Database tables initialized.")
        except Exception as e:
            logger.error("Database initialization failed: %s", str(e))

        # Preload models into memory if configured
        if app.config.get("PRELOAD_MODELS", True):
            try:
                logger.info("Pre-loading deep learning models...")
                get_image_model()
                get_audio_model()
                logger.info("Models pre-loaded successfully.")
            except Exception as e:
                logger.error("Model pre-load error (will retry lazily on request): %s", str(e))
        else:
            logger.info("Model pre-loading skipped at startup (lazy loading enabled).")

    return app

# Expose app instance for WSGI / Gunicorn / Flask CLI
app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1")
    logger.info("Starting DeepTrust AI server on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=debug)

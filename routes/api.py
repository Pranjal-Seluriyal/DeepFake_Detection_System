import os
import io
import base64
import logging
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from database import db, Prediction
from ml.inference import predict_image, predict_audio, get_image_model, get_audio_model
from services.storage import upload_file, is_cloud_storage_enabled
from config import Config

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")

@api_bp.before_request
def check_proxy_backend():
    """If BACKEND_API_URL is configured (e.g. Vercel frontend calling dedicated ML backend), forward requests."""
    backend_url = getattr(Config, "BACKEND_API_URL", None)
    if not backend_url or request.path == "/api/health":
        return None

    try:
        import requests
        from flask import Response
        backend_url = backend_url.rstrip("/")
        query = f"?{request.query_string.decode('utf-8')}" if request.query_string else ""
        target = f"{backend_url}{request.path}{query}"
        headers = {k: v for k, v in request.headers if k.lower() not in ("host", "content-length")}

        if request.is_json:
            resp = requests.request(request.method, target, json=request.get_json(), headers=headers, timeout=180)
        elif request.files:
            files = {k: (f.filename, f.stream, f.mimetype) for k, f in request.files.items()}
            resp = requests.request(request.method, target, data=request.form.to_dict(), files=files, headers=headers, timeout=180)
        else:
            resp = requests.request(request.method, target, data=request.get_data(), headers=headers, timeout=180)

        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        response_headers = [(k, v) for (k, v) in resp.raw.headers.items() if k.lower() not in excluded_headers]
        return Response(resp.content, resp.status_code, response_headers)
    except Exception as e:
        logger.error("Failed to proxy API request to %s: %s", getattr(Config, "BACKEND_API_URL", ""), e)
        return jsonify({"success": False, "error": f"Inference backend unavailable: {e}"}), 502

def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health status endpoint for monitoring and container health checks."""
    db_status = "healthy"
    try:
        db.session.execute(db.text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    image_model_loaded = get_image_model() is not None
    audio_model_loaded = get_audio_model() is not None

    return jsonify({
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "models": {
            "image_model": "loaded" if image_model_loaded else "unavailable",
            "audio_model": "loaded" if audio_model_loaded else "unavailable"
        },
        "storage": "cloudinary" if is_cloud_storage_enabled() else "local_fallback"
    })


@api_bp.route("/predict/image", methods=["POST"])
def api_predict_image():
    """
    Main image deepfake detection endpoint.
    Accepts multipart/form-data with 'file', or JSON with 'image' (base64 data URL).
    """
    try:
        file_bytes = None
        filename = "captured_frame.jpg"

        # Check for standard multipart file upload
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"success": False, "error": "No file selected."}), 400

            filename = secure_filename(file.filename) or "upload.jpg"
            if not allowed_file(filename, Config.ALLOWED_IMAGE_EXTENSIONS):
                return jsonify({
                    "success": False,
                    "error": f"Invalid image format. Allowed: {', '.join(Config.ALLOWED_IMAGE_EXTENSIONS)}"
                }), 400

            file_bytes = file.read()

        # Check for base64 data URL upload (e.g., from camera capture)
        elif request.is_json and "image" in request.json:
            data_url = request.json["image"]
            if "," in data_url:
                header, encoded = data_url.split(",", 1)
            else:
                encoded = data_url
            try:
                file_bytes = base64.b64decode(encoded)
                filename = request.json.get("filename", "camera_capture.jpg")
            except Exception:
                return jsonify({"success": False, "error": "Invalid base64 image data."}), 400

        else:
            return jsonify({"success": False, "error": "No image provided. Upload a file or pass base64 image."}), 400

        if not file_bytes or len(file_bytes) == 0:
            return jsonify({"success": False, "error": "Uploaded image file is empty."}), 400

        if len(file_bytes) > Config.MAX_CONTENT_LENGTH:
            return jsonify({
                "success": False,
                "error": f"Image exceeds maximum size limit of {Config.MAX_CONTENT_LENGTH // (1024 * 1024)}MB."
            }), 413

        # 1. Run ML inference on image bytes
        result = predict_image(file_bytes)

        # 2. Upload image to Cloud Storage (Cloudinary or local fallback)
        media_url, _ = upload_file(io.BytesIO(file_bytes), filename, resource_type="image", folder="deeptrust_images")

        # 3. Store record in PostgreSQL via SQLAlchemy
        prediction_record = Prediction(
            prediction_type="image",
            original_filename=filename,
            media_url=media_url,
            status=result["status"],
            confidence=result["confidence"],
            raw_score=result["raw_score"]
        )

        try:
            db.session.add(prediction_record)
            db.session.commit()
            record_id = prediction_record.id
            created_at = prediction_record.created_at.isoformat() if prediction_record.created_at else None
        except Exception as db_err:
            logger.error("Failed to commit prediction to database: %s", str(db_err))
            db.session.rollback()
            record_id = None
            created_at = None

        return jsonify({
            "success": True,
            "prediction": result["status"],
            "status": result["status"],
            "confidence": f"{result['confidence']}%",
            "confidence_value": result["confidence"],
            "raw_score": result["raw_score"],
            "metrics": result.get("metrics", {}),
            "image_url": media_url,
            "prediction_id": record_id,
            "created_at": created_at,
            "storage_provider": "cloudinary" if is_cloud_storage_enabled() else "local"
        }), 200

    except ValueError as ve:
        logger.warning("Validation error during image prediction: %s", str(ve))
        return jsonify({"success": False, "error": str(ve)}), 400
    except RuntimeError as re:
        logger.error("Model unavailable during image prediction: %s", str(re))
        return jsonify({
            "success": False,
            "error": "The AI detection model is currently unavailable or downloading weights from Hugging Face. Please retry in a few moments."
        }), 503
    except Exception as e:
        logger.error("Internal error during image prediction: %s", str(e), exc_info=True)
        return jsonify({"success": False, "error": "An error occurred during AI analysis. Please try again."}), 500


@api_bp.route("/predict/audio", methods=["POST"])
def api_predict_audio():
    """Audio deepfake detection endpoint."""
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No audio file provided."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No audio file selected."}), 400

        filename = secure_filename(file.filename) or "sample.wav"
        if not allowed_file(filename, Config.ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({
                "success": False,
                "error": f"Invalid audio format. Allowed: {', '.join(Config.ALLOWED_AUDIO_EXTENSIONS)}"
            }), 400

        # Temporarily save for librosa / pydub processing
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        temp_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(temp_path)

        # 1. Run ML inference
        result = predict_audio(temp_path)

        # 2. Upload to Cloud Storage
        media_url, _ = upload_file(temp_path, filename, resource_type="auto", folder="deeptrust_audio")

        # 3. Store record in PostgreSQL
        prediction_record = Prediction(
            prediction_type="audio",
            original_filename=filename,
            media_url=media_url,
            status=result["status"],
            confidence=result["confidence"],
            raw_score=result["raw_score"]
        )

        try:
            db.session.add(prediction_record)
            db.session.commit()
            record_id = prediction_record.id
            created_at = prediction_record.created_at.isoformat() if prediction_record.created_at else None
        except Exception as db_err:
            logger.error("Database save failed: %s", str(db_err))
            db.session.rollback()
            record_id = None
            created_at = None

        # Clean up local temp file if Cloudinary was used
        if is_cloud_storage_enabled() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

        return jsonify({
            "success": True,
            "prediction": result["status"],
            "status": result["status"],
            "confidence": f"{result['confidence']}%",
            "confidence_value": result["confidence"],
            "raw_score": result["raw_score"],
            "audio_url": media_url,
            "prediction_id": record_id,
            "created_at": created_at
        }), 200

    except Exception as e:
        logger.error("Audio prediction error: %s", str(e), exc_info=True)
        return jsonify({"success": False, "error": "An error occurred during audio analysis."}), 500


@api_bp.route("/predictions", methods=["GET"])
def get_predictions():
    """Retrieve historical prediction records with pagination and filtering."""
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
        pred_type = request.args.get("type")

        query = Prediction.query.order_by(Prediction.created_at.desc())
        if pred_type in ["image", "audio"]:
            query = query.filter_by(prediction_type=pred_type)

        records = query.limit(limit).all()
        return jsonify({
            "success": True,
            "count": len(records),
            "predictions": [r.to_dict() for r in records]
        }), 200
    except Exception as e:
        logger.error("Failed to fetch predictions: %s", str(e))
        return jsonify({"success": False, "error": "Failed to retrieve history."}), 500


@api_bp.route("/predictions/<int:prediction_id>", methods=["GET"])
def get_prediction_detail(prediction_id):
    """Retrieve single prediction record by ID."""
    record = Prediction.query.get(prediction_id)
    if not record:
        return jsonify({"success": False, "error": "Prediction record not found."}), 404
    return jsonify({"success": True, "prediction": record.to_dict()}), 200

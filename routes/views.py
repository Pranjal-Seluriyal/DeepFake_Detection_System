import os
from flask import Blueprint, render_template, request, jsonify, send_from_directory, current_app
from werkzeug.utils import secure_filename

from database import db, Prediction
from ml.inference import predict_image, predict_audio
from services.storage import upload_file
from config import Config

views_bp = Blueprint("views", __name__)

@views_bp.route("/")
def home():
    """Main landing page & forensic verification portal."""
    return render_template("index.html")


@views_bp.route("/documentation")
def documentation():
    """Technical architecture and documentation page."""
    return render_template("documentation.html")


@views_bp.route("/history")
def history():
    """Prediction history gallery view."""
    return render_template("history.html")


@views_bp.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Safely serves uploaded files for local development fallback."""
    return send_from_directory(Config.UPLOAD_FOLDER, filename)


@views_bp.route("/analyze", methods=["POST"])
def analyze():
    """
    Backwards-compatible legacy route for existing tests or scripts.
    Maintains exact legacy contract: { status: '...', confidence: '...%' }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    mode = request.form.get("mode", "image")

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        if mode == "image":
            file_bytes = file.read()
            filename = secure_filename(file.filename) or "image.jpg"
            result = predict_image(file_bytes)

            # Store in Cloud Storage & DB
            import io
            media_url, _ = upload_file(io.BytesIO(file_bytes), filename, resource_type="image", folder="deeptrust_images")

            pred = Prediction(
                prediction_type="image",
                original_filename=filename,
                media_url=media_url,
                status=result["status"],
                confidence=result["confidence"],
                raw_score=result["raw_score"]
            )
            try:
                db.session.add(pred)
                db.session.commit()
            except Exception:
                db.session.rollback()

            return jsonify({
                "status": result["status"],
                "confidence": f"{result['confidence']}%",
                "image_url": media_url,
                "prediction_id": pred.id if pred.id else None
            })

        elif mode == "audio":
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            filename = secure_filename(file.filename) or "audio.wav"
            temp_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(temp_path)

            result = predict_audio(temp_path)
            media_url, _ = upload_file(temp_path, filename, resource_type="auto", folder="deeptrust_audio")

            pred = Prediction(
                prediction_type="audio",
                original_filename=filename,
                media_url=media_url,
                status=result["status"],
                confidence=result["confidence"],
                raw_score=result["raw_score"]
            )
            try:
                db.session.add(pred)
                db.session.commit()
            except Exception:
                db.session.rollback()

            return jsonify({
                "status": result["status"],
                "confidence": f"{result['confidence']}%",
                "audio_url": media_url,
                "prediction_id": pred.id if pred.id else None
            })

        else:
            return jsonify({"error": "Invalid mode"}), 400

    except RuntimeError as re:
        return jsonify({"error": str(re)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

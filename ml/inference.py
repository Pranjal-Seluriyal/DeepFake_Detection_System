import os
import logging
from pathlib import Path
import numpy as np
import torch

from .architectures import AudioModel
from .ucfnet_detector import UCFNetDetector
from config import Config

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_THRESH_AUDIO = Config.BEST_THRESH_AUDIO

# Global model singletons (loaded ONCE at startup)
_ucfnet_detector = None
_audio_model = None

# ---------------------------------------------------------
# IMAGE MODEL (UCF-NET ViT-L/14 + DINOv2-L + MoE)
# ---------------------------------------------------------

def get_image_model():
    """Returns singleton instance of UCFNetDetector, loading weights once at startup."""
    global _ucfnet_detector
    if _ucfnet_detector is not None:
        return _ucfnet_detector.model

    logger.info("Loading UCFNet model...")
    try:
        from services.model_loader import get_ucfnet_model_path
        model_path = Config.IMAGE_MODEL_PATH or get_ucfnet_model_path(
            repo_id=Config.HF_MODEL_REPO,
            filename=Config.HF_MODEL_FILENAME,
            token=Config.HF_TOKEN
        )
    except Exception as e:
        logger.error("Failed to obtain UCFNet model weights: %s", str(e))
        return None

    try:
        logger.info("Initializing UCF-Net Deepfake Image Detector on %s...", DEVICE)
        _ucfnet_detector = UCFNetDetector.get_instance(
            checkpoint_path=model_path,
            clip_path=Config.UCFNET_CLIP_PATH,
            device=DEVICE
        )
        logger.info("UCFNet model loaded successfully.")
        return _ucfnet_detector.model
    except Exception as e:
        logger.error("Failed to initialize UCF-Net detector: %s", str(e))
        return None


def predict_image(image_input):
    """
    Performs deepfake inference on an image using official UCF-Net.
    Accepts: file path (str/Path), file bytes, PIL Image, or numpy array.
    Returns:
        dict with status, confidence, raw_score, label, fake_probability, real_probability, metrics
    """
    global _ucfnet_detector
    if _ucfnet_detector is None:
        model = get_image_model()
        if model is None or _ucfnet_detector is None:
            raise RuntimeError(
                "UCFNet model is currently unavailable. Failed to download or load model weights from Hugging Face."
            )

    return _ucfnet_detector.predict(image_input)


# ---------------------------------------------------------
# AUDIO MODEL (PRESERVED: CNN + BiLSTM + Attention)
# ---------------------------------------------------------

def get_audio_model():
    """Returns singleton instance of AudioModel, loading weights if needed."""
    global _audio_model
    if _audio_model is not None:
        return _audio_model

    logger.info("Initializing AudioModel on %s...", DEVICE)
    model = AudioModel().to(DEVICE)
    model_path = Config.AUDIO_MODEL_PATH

    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=DEVICE)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
            else:
                model.load_state_dict(ckpt)
            model.eval()
            logger.info("Audio model loaded successfully from %s", model_path)
        except Exception as e:
            logger.error("Failed to load audio model weights: %s", str(e))
    else:
        logger.warning("Audio model weights not found at %s", model_path)

    _audio_model = model
    return _audio_model


def convert_to_wav(input_path):
    ext = os.path.splitext(input_path)[1].lower().replace(".", "")
    if ext == "wav":
        return input_path

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        output_path = os.path.splitext(input_path)[0] + "_converted.wav"
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        logger.warning("Audio conversion fallback: %s", str(e))
        return input_path


def preprocess_audio(path):
    import librosa

    wav_path = convert_to_wav(path)
    y, sr = librosa.load(wav_path, sr=16000)

    L = sr * 3
    if len(y) < L:
        y = np.pad(y, (0, L - len(y)))
    else:
        y = y[:L]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel)

    pitch = librosa.yin(y, fmin=50, fmax=300)
    energy = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    T = mel_db.shape[1]

    def fix(x):
        return np.pad(x, (0, T - len(x)))[:T]

    features = np.vstack([mel_db, fix(pitch), fix(energy), fix(zcr)])
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return tensor, wav_path


def predict_audio(audio_path):
    """
    Performs deepfake inference on an audio file using existing CNN+BiLSTM model.
    Preserves exact formula:
        score = torch.sigmoid(model(x)).item()
        status = 'Deepfake' if score > BEST_THRESH_AUDIO else 'Real'
        confidence = round((score if status == 'Deepfake' else 1 - score) * 100, 2)
    """
    model = get_audio_model()
    x, _ = preprocess_audio(audio_path)

    with torch.no_grad():
        logits = model(x)
        score = torch.sigmoid(logits).item()

    status = "Deepfake" if score > BEST_THRESH_AUDIO else "Real"
    confidence = round((score if status == "Deepfake" else 1 - score) * 100, 2)

    return {
        "status": status,
        "confidence": confidence,
        "raw_score": round(score, 4),
        "threshold": BEST_THRESH_AUDIO
    }

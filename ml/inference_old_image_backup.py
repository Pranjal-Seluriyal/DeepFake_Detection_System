import os
import logging
from pathlib import Path
import numpy as np
import cv2
import torch

from .architectures import ImageDeepfakeDetector, AudioModel
from config import Config

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = Config.IMG_SIZE
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BEST_THRESH_IMAGE = Config.BEST_THRESH_IMAGE
BEST_THRESH_AUDIO = Config.BEST_THRESH_AUDIO

# Global model singletons (loaded ONCE at startup)
_image_model = None
_audio_model = None

def get_image_model():
    """Returns singleton instance of ImageDeepfakeDetector, loading weights if needed."""
    global _image_model
    if _image_model is not None:
        return _image_model

    logger.info("Initializing ImageDeepfakeDetector on %s...", DEVICE)
    model = ImageDeepfakeDetector().to(DEVICE)
    model_path = Config.IMAGE_MODEL_PATH

    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=DEVICE)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
            else:
                model.load_state_dict(ckpt)
            model.eval()
            logger.info("Image model loaded successfully from %s", model_path)
        except Exception as e:
            logger.error("Failed to load image model weights: %s", str(e))
    else:
        logger.warning("Image model weights not found at %s", model_path)

    _image_model = model
    return _image_model


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


# ---------------------------------------------------------
# IMAGE PREPROCESSING & INFERENCE
# ---------------------------------------------------------

def normalize_01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def preprocess_image(image_input):
    """
    Takes a file path (str/Path), bytes, or numpy array.
    Returns 4 tensors: (d, e, s, a) matching the 4-branch architecture.
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Could not read image from path: {image_input}")
    elif isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from provided byte stream.")
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    u8 = (gray * 255).astype(np.uint8)

    # 1. 2D FFT Frequency magnitude (Diffusion branch)
    fft = normalize_01(np.abs(np.fft.fftshift(np.fft.fft2(gray))))[None]

    # 2. Canny Edge Map (Structural consistency)
    edge = (cv2.Canny(u8, 100, 200).astype(np.float32) / 255.0)[None]

    # 3. ImageNet normalized RGB (Semantic branch)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)

    # 4. Laplacian Filter (Sensor / Acquisition noise)
    lap = normalize_01(cv2.Laplacian(u8, cv2.CV_64F).astype(np.float32))[None]

    # Convert to PyTorch tensors on DEVICE
    d = torch.tensor(fft, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    e = torch.tensor(edge, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    s = torch.tensor(rgb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    a = torch.tensor(lap, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    return d, e, s, a


def predict_image(image_input):
    """
    Performs deepfake inference on an image.
    Preserves exact formula:
        score = torch.sigmoid(logits).item()
        status = 'Deepfake' if score > BEST_THRESH_IMAGE else 'Real'
        confidence = round((score if status == 'Deepfake' else 1 - score) * 100, 2)
    """
    model = get_image_model()
    d, e, s, a = preprocess_image(image_input)

    with torch.no_grad():
        logits, feats = model(d, e, s, a)
        score = torch.sigmoid(logits).item()

    status = "Deepfake" if score > BEST_THRESH_IMAGE else "Real"
    confidence = round((score if status == "Deepfake" else 1 - score) * 100, 2)

    # Sub-metric indicators for forensic visualization
    fft_variance = float(torch.var(d).cpu().item())
    edge_density = float(torch.mean(e).cpu().item())
    lap_energy = float(torch.mean(a).cpu().item())

    return {
        "status": status,
        "confidence": confidence,
        "raw_score": round(score, 4),
        "threshold": BEST_THRESH_IMAGE,
        "metrics": {
            "frequency_dispersion": round(fft_variance * 100, 2),
            "edge_density": round(edge_density * 100, 2),
            "acquisition_noise": round(lap_energy * 100, 2),
        }
    }


# ---------------------------------------------------------
# AUDIO PREPROCESSING & INFERENCE
# ---------------------------------------------------------

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
    Performs deepfake inference on an audio file.
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

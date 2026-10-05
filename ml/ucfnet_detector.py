"""
UCFNetDetector: Official UCF-Net Deepfake Image Detector Wrapper
Harnessing CLIP ViT-L/14 and DINOv2 ViT-L with Uncertainty-Aware Cascaded Fusion (MoE).
"""

import os
import sys
import io
import time
import types
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

# Configure logging
logger = logging.getLogger("UCFNetDetector")

# Mock training metrics class to avoid importing unnecessary training packages (scipy/sklearn metrics)
if "metrics.base_metrics_class" not in sys.modules:
    mock_metrics = types.ModuleType("metrics.base_metrics_class")
    mock_metrics.calculate_metrics_for_train = lambda *args, **kwargs: (0.0, 0.0, 0.0, 0.0)
    sys.modules["metrics.base_metrics_class"] = mock_metrics

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
UCFNET_CORE = BASE_DIR / "ml" / "ucfnet_core"

# Setup sys.path for UCF-Net modules
for path_entry in [
    str(UCFNET_CORE),
    str(UCFNET_CORE / "hf_space"),
    str(UCFNET_CORE / "DeepfakeBench" / "training"),
]:
    if os.path.exists(path_entry) and path_entry not in sys.path:
        sys.path.insert(0, path_entry)

# Import fast loading from official UCF-Net hf_space
from fast_load import load_detector_fast


class UCFNetDetector:
    """
    Production wrapper for UCF-Net Image Deepfake Detection.
    Provides singleton loading and simple detector.predict(image) interface.
    """

    _instance = None

    def __init__(self, checkpoint_path=None, config_path=None, clip_path=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Resolve model weights checkpoint path via model loader if not specified
        if checkpoint_path is None:
            from services.model_loader import get_ucfnet_model_path
            checkpoint_path = get_ucfnet_model_path()

        self.checkpoint_path = Path(checkpoint_path)

        self.config_path = self._resolve_path(
            config_path,
            [
                UCFNET_CORE / "DeepfakeBench" / "training" / "config" / "detector" / "ucfnet.yaml",
            ]
        )

        self.test_config_path = self._resolve_path(
            None,
            [
                UCFNET_CORE / "DeepfakeBench" / "training" / "config" / "test_config.yaml",
            ]
        )

        self.clip_path = self._resolve_path(
            clip_path,
            [
                BASE_DIR / "models" / "pretrained" / "clip-vit-large-patch14-local",
            ]
        )

        logger.info("Initializing UCF-Net on device: %s", self.device)
        logger.info("UCF-Net checkpoint: %s", self.checkpoint_path)
        logger.info("UCF-Net config: %s", self.config_path)

        self.config = self._build_config()
        self.resolution = int(self.config.get("resolution", 224))
        self.mean = np.asarray(self.config.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.asarray(self.config.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

        start_time = time.time()
        self.model = self._load_model()
        logger.info("UCF-Net loaded successfully in %.2f seconds.", time.time() - start_time)

        # CPU-based face detector initialization
        self.face_detector = self._init_face_detector()

    def _resolve_path(self, user_path, fallback_candidates):
        if user_path and os.path.exists(user_path):
            return Path(user_path)
        for candidate in fallback_candidates:
            if candidate and os.path.exists(candidate):
                return Path(candidate)
        return Path(fallback_candidates[0])

    def _build_config(self):
        with open(self.config_path, "r") as f:
            detector_cfg = yaml.safe_load(f) or {}
        with open(self.test_config_path, "r") as f:
            test_cfg = yaml.safe_load(f) or {}

        cfg = dict(test_cfg)
        cfg.update({k: v for k, v in detector_cfg.items() if v is not None})
        cfg["cuda"] = (self.device.type == "cuda")
        cfg["lmdb"] = False
        cfg["workers"] = 0

        clip_cfg = dict(cfg.get("clip") or {})
        clip_cfg["model_path"] = str(self.clip_path)
        cfg["clip"] = clip_cfg

        dino_cfg = dict(cfg.get("dino") or {})
        dino_cfg["weights"] = ""  # fast_load skips reading backbone weights since checkpoint contains them
        cfg["dino"] = dino_cfg

        return cfg

    def _load_model(self):
        training_dir = str(self.config_path.parent.parent.parent)
        repo_root = str(self.config_path.parent.parent.parent.parent)

        model = load_detector_fast(
            config=self.config,
            weights_path=str(self.checkpoint_path),
            device=self.device,
            training_dir=training_dir,
            repo_root=repo_root,
        )
        model.eval()
        return model

    def _init_face_detector(self):
        """Initializes a local CPU-based face detector (OpenCV YuNet)."""
        yunet_path = self._resolve_path(
            None,
            [
                BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx",
            ]
        )
        if yunet_path.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                detector = cv2.FaceDetectorYN.create(
                    model=str(yunet_path),
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.5,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                logger.info("Initialized YuNet face detector from %s", yunet_path)
                return detector
            except Exception as e:
                logger.warning("Failed to initialize YuNet face detector: %s", e)
        return None

    def _detect_faces(self, img_rgb):
        """
        Detects faces in an RGB image.
        Returns list of bounding boxes [(x, y, w, h)].
        """
        if self.face_detector is None or img_rgb is None:
            return []

        h, w = img_rgb.shape[:2]
        if h <= 0 or w <= 0:
            return []

        bboxes = []
        try:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            self.face_detector.setInputSize((w, h))
            _, faces = self.face_detector.detect(img_bgr)
            if faces is not None:
                for face in faces:
                    box = face[:4].astype(int)
                    if box[2] > 10 and box[3] > 10:
                        bboxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        except Exception as e:
            logger.warning("Face detection failed during inference: %s", e)

        return bboxes

    def _crop_face(self, img_rgb, bbox, margin=0.3):
        """
        Crops face from img_rgb given bbox (x, y, w, h) with margin,
        preserving a square 1:1 aspect ratio.
        """
        img_h, img_w = img_rgb.shape[:2]
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        size = max(w, h) * (1.0 + margin)
        half = size / 2.0

        x1 = int(round(cx - half))
        y1 = int(round(cy - half))
        x2 = int(round(cx + half))
        y2 = int(round(cy + half))

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - img_w)
        pad_bottom = max(0, y2 - img_h)

        x1_clamped = max(0, x1)
        y1_clamped = max(0, y1)
        x2_clamped = min(img_w, x2)
        y2_clamped = min(img_h, y2)

        cropped = img_rgb[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            cropped = cv2.copyMakeBorder(
                cropped,
                pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_REFLECT_101,
            )

        return cropped

    def _load_image_rgb(self, image_input):
        """Converts any supported image input format into an RGB uint8 numpy array."""
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
            if img is None:
                pil_img = Image.open(str(image_input)).convert("RGB")
                return np.asarray(pil_img)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, (bytes, bytearray)):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
                return np.asarray(pil_img)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            return np.asarray(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            img = image_input
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img.shape[2] == 3:
                return img
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _img_to_tensor(self, img_rgb):
        """Preprocesses an RGB numpy uint8 image into a normalized CHW tensor."""
        resized = cv2.resize(img_rgb, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)
        norm = resized.astype(np.float32) / 255.0
        norm = (norm - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)
        return torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    def _forward_tensor(self, tensor):
        """Runs the UCF-Net forward pass on a preprocessed tensor, returning fake probability."""
        data_dict = {
            "image": tensor,
            "label": torch.zeros(1, dtype=torch.long, device=self.device),
            "mask": None,
            "landmark": None,
        }
        with torch.no_grad():
            output = self.model(data_dict, inference=True)
            return float(output["prob"].detach().cpu().item())

    def preprocess(self, image_input):
        """
        Preprocesses full image for UCF-Net (224x224, RGB, ImageNet mean/std).
        Maintained for backwards compatibility.
        """
        img_rgb = self._load_image_rgb(image_input)
        return self._img_to_tensor(img_rgb)

    def predict(self, image_input):
        """
        Runs UCF-Net inference on the input image.
        If face(s) are detected:
            - Crops each face preserving square aspect ratio with small margin
            - Resizes face crop to 224x224
            - Combines results using highest fake probability across detected faces
        If no face is detected:
            - Falls back to existing full-image UCF-Net inference unchanged
        """
        if isinstance(image_input, torch.Tensor):
            fake_prob = self._forward_tensor(image_input)
            face_detected = False
            num_faces = 0
        else:
            img_rgb = self._load_image_rgb(image_input)
            faces = self._detect_faces(img_rgb)
            num_faces = len(faces)

            if num_faces > 0:
                face_detected = True
                face_probs = []
                for bbox in faces:
                    crop_rgb = self._crop_face(img_rgb, bbox, margin=0.3)
                    crop_tensor = self._img_to_tensor(crop_rgb)
                    face_probs.append(self._forward_tensor(crop_tensor))
                fake_prob = max(face_probs)
            else:
                face_detected = False
                full_tensor = self._img_to_tensor(img_rgb)
                fake_prob = self._forward_tensor(full_tensor)

        real_prob = 1.0 - fake_prob
        is_fake = fake_prob >= 0.5
        status = "Deepfake" if is_fake else "Real"
        label = "DEEPFAKE" if is_fake else "REAL"
        confidence = round((fake_prob if is_fake else real_prob) * 100, 2)

        return {
            "label": label,
            "status": status,
            "prediction": status,
            "confidence": confidence,
            "raw_score": round(fake_prob, 4),
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "threshold": 0.5,
            "model": "UCF-Net (CLIP-ViT-L/14 + DINOv2-L + MoE)",
            "device": str(self.device),
            "face_detected": face_detected,
            "faces_count": num_faces,
            "metrics": {
                "fake_probability": round(fake_prob * 100, 2),
                "real_probability": round(real_prob * 100, 2),
                "uncertainty_margin": round(abs(fake_prob - 0.5) * 200, 2),
            },
        }

    @classmethod
    def get_instance(cls, **kwargs):
        """Singleton accessor to ensure UCF-Net is loaded ONCE."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

"""
services/model_loader.py
Model loading and caching utility for DeepTrust AI.
Handles automatic fetching and caching of the UCFNet PyTorch model from Hugging Face
using huggingface_hub.hf_hub_download.
"""

import os
import sys
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

# Suppress symlink warnings on Windows platforms for huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

logger = logging.getLogger("ModelLoader")

BASE_DIR = Path(__file__).resolve().parent.parent

# In-memory cached model path and thread-safety lock
_download_lock = threading.Lock()
_cached_model_path: Optional[str] = None


def get_hf_config() -> Tuple[str, str, Optional[str]]:
    """
    Returns the configured Hugging Face repo ID, model filename, and optional authentication token.
    Configurable via environment variables:
      - HF_MODEL_REPO (default: PranjalSeluriyal/Deepfake_Detection)
      - HF_MODEL_FILENAME (default: model_best.pth)
      - HF_TOKEN (default: None, for public repositories)
    """
    repo_id = os.getenv("HF_MODEL_REPO", "PranjalSeluriyal/Deepfake_Detection").strip()
    filename = os.getenv("HF_MODEL_FILENAME", "model_best.pth").strip()
    token = os.getenv("HF_TOKEN", "").strip() or None
    return repo_id, filename, token


def is_model_cached_locally(repo_id: Optional[str] = None, filename: Optional[str] = None, token: Optional[str] = None) -> Optional[str]:
    """
    Checks if the model file is already present on local disk without triggering network calls:
    1. In-memory cached path
    2. Explicit IMAGE_MODEL_PATH environment variable
    3. Hugging Face hub local cache (local_files_only=True)
    4. Local candidate directories in the repository (e.g. models/)
    
    Returns the absolute path str if found, else None.
    """
    global _cached_model_path
    if _cached_model_path and os.path.exists(_cached_model_path):
        return _cached_model_path

    default_repo, default_filename, default_token = get_hf_config()
    repo_id = repo_id or default_repo
    filename = filename or default_filename
    token = token or default_token

    # 1. Explicit environment variable override
    env_override = os.getenv("IMAGE_MODEL_PATH")
    if env_override and os.path.isfile(env_override):
        _cached_model_path = str(Path(env_override).resolve())
        return _cached_model_path

    # 2. Check Hugging Face local cache without network
    try:
        from huggingface_hub import hf_hub_download
        cached_hf = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            local_files_only=True
        )
        if os.path.isfile(cached_hf):
            _cached_model_path = str(Path(cached_hf).resolve())
            return _cached_model_path
    except Exception:
        pass

    # 3. Local candidate files in project directory
    local_candidates = [
        BASE_DIR / "models" / "ucfnet" / filename,
        BASE_DIR / "models" / filename,
    ]
    for candidate in local_candidates:
        if candidate.is_file():
            _cached_model_path = str(candidate.resolve())
            return _cached_model_path

    return None


def get_ucfnet_model_path(
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False
) -> str:
    """
    Obtains the local path to the UCFNet model weights file.
    
    1. Checks in-memory cache variable for instant resolution.
    2. Checks if an explicit IMAGE_MODEL_PATH env var points to an existing file.
    3. Checks Hugging Face local cache (local_files_only=True) to avoid network calls.
    4. Checks if the model file is already present in local candidate directories.
    5. If not found locally, downloads the model from Hugging Face using hf_hub_download().
    6. Returns the downloaded local model path and caches it in memory.
    """
    global _cached_model_path

    if not force_download and _cached_model_path and os.path.exists(_cached_model_path):
        return _cached_model_path

    default_repo, default_filename, default_token = get_hf_config()
    repo_id = repo_id or default_repo
    filename = filename or default_filename
    token = token or default_token

    with _download_lock:
        # Re-check inside lock
        if not force_download:
            local_path = is_model_cached_locally(repo_id=repo_id, filename=filename, token=token)
            if local_path:
                logger.info("UCFNet model located locally: %s", local_path)
                _cached_model_path = local_path
                return _cached_model_path

        # If not available locally, download from Hugging Face
        logger.info("Downloading UCFNet model from Hugging Face: %s/%s...", repo_id, filename)
        try:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                cache_dir=cache_dir
            )
            resolved_path = str(Path(downloaded_path).resolve())
            logger.info("UCFNet model downloaded successfully to: %s", resolved_path)
            _cached_model_path = resolved_path
            return _cached_model_path
        except Exception as e:
            logger.error(
                "Failed to download UCFNet model from Hugging Face (%s/%s): %s",
                repo_id, filename, str(e)
            )
            raise RuntimeError(
                f"Failed to obtain UCFNet model from Hugging Face repository '{repo_id}' ({filename}): {e}. "
                "Please verify your network connection, repository visibility, or provide a HF_TOKEN."
            ) from e

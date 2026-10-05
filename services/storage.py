import os
import uuid
import logging
from pathlib import Path
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

# Check if cloudinary is installed
try:
    import cloudinary
    import cloudinary.uploader
    CLOUDINARY_AVAILABLE = True
except ImportError:
    CLOUDINARY_AVAILABLE = False
    logger.warning("Cloudinary package not available. Falling back to local storage.")

def init_cloudinary():
    """Initializes Cloudinary from environment variables if available."""
    if not CLOUDINARY_AVAILABLE:
        return False
    
    cloudinary_url = os.getenv("CLOUDINARY_URL")
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    api_key = os.getenv("CLOUDINARY_API_KEY")
    api_secret = os.getenv("CLOUDINARY_API_SECRET")

    if cloudinary_url:
        cloudinary.config(cloudinary_url=cloudinary_url)
        return True
    elif cloud_name and api_key and api_secret:
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True
        )
        return True
    return False

CLOUDINARY_ACTIVE = init_cloudinary()

def is_cloud_storage_enabled():
    """Returns True if Cloudinary is active and ready."""
    return CLOUDINARY_ACTIVE

def upload_file(file_path_or_stream, original_filename, resource_type="image", folder="deeptrust_scans"):
    """
    Uploads a file to Cloudinary if configured; otherwise saves locally.
    
    Returns:
        tuple: (url: str, public_id_or_path: str)
    """
    # Generate clean, unique safe filename
    base_name = secure_filename(original_filename) if original_filename else "scan"
    unique_suffix = uuid.uuid4().hex[:8]
    ext = os.path.splitext(base_name)[1].lower()
    unique_filename = f"{os.path.splitext(base_name)[0]}_{unique_suffix}{ext}"

    if is_cloud_storage_enabled():
        try:
            logger.info("Uploading %s to Cloudinary in folder '%s'...", base_name, folder)
            upload_result = cloudinary.uploader.upload(
                file_path_or_stream,
                folder=folder,
                resource_type=resource_type,
                use_filename=True,
                unique_filename=True
            )
            secure_url = upload_result.get("secure_url")
            public_id = upload_result.get("public_id")
            logger.info("Cloudinary upload successful: %s", secure_url)
            return secure_url, public_id
        except Exception as e:
            logger.error("Cloudinary upload failed: %s. Falling back to local storage.", str(e))
            # Fall back to local below

    # Local fallback
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    local_path = upload_dir / unique_filename

    if isinstance(file_path_or_stream, (str, Path)):
        # It's an existing file path, copy or read
        import shutil
        shutil.copy2(file_path_or_stream, local_path)
    else:
        # File-like object (FileStorage or BytesIO)
        file_path_or_stream.seek(0)
        with open(local_path, "wb") as f:
            f.write(file_path_or_stream.read())

    # Return web-accessible path
    web_url = f"/uploads/{unique_filename}"
    return web_url, str(local_path)

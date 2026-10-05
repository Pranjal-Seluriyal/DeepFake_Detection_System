import os
import sys
from pathlib import Path

# Add project root to sys.path so modules (app, config, routes, services, ml) are importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# On serverless environments, disable synchronous startup preloading to prevent cold-start timeout
os.environ.setdefault("PRELOAD_MODELS", "false")

# Expose the WSGI Flask application entrypoint for Vercel Python runtime
from app import app

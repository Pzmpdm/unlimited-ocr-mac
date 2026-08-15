"""Configuration for Unlimited-OCR Web."""
import os
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).parent.resolve()
OCR_REPO_DIR = PROJECT_DIR.parent
UPLOAD_DIR = PROJECT_DIR / "uploads"
PUBLIC_DIR = PROJECT_DIR / "public"

# Ensure dirs
UPLOAD_DIR.mkdir(exist_ok=True)

# Server
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8800"))

# OCR defaults
DEFAULT_MAX_LENGTH = 8192
DEFAULT_BASE_SIZE = 1024
DEFAULT_IMAGE_SIZE = 640
PDF_DPI = 200

# Hard cap for generated tokens per page. A request can ask for more
# (e.g. 12000 for a dense page) but generation is clamped here.
OCR_MAX_TOKENS = int(os.getenv("OCR_MAX_TOKENS", "8192"))

# Translation (OpenAI-compatible API)
TRANSLATE_API_BASE = os.environ.get("TRANSLATE_API_BASE", "")
TRANSLATE_API_KEY = os.environ.get("TRANSLATE_API_KEY", "")
TRANSLATE_MODEL = os.environ.get("TRANSLATE_MODEL", "gpt-4o")

# Session expiry (seconds)
SESSION_TTL = 7200

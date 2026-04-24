"""Load Gemini API key from environment or .env file (never commit .env)."""
import os
from pathlib import Path


def _load_dotenv(root: Path | None = None):
    try:
        from dotenv import load_dotenv
        env_path = (root or Path(__file__).parent.parent) / ".env"
        load_dotenv(env_path)
    except ImportError:
        pass


def get_api_key(root: Path | None = None):
    _load_dotenv(root)
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY not set. Either export it or add it to .env"
        )
    return key

"""Loads settings from the .env file so the rest of the app reads them in one place."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PLACEHOLDER_KEYS = {"", "your-openai-api-key-here", "your-gemini-api-key-here"}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Both providers are used through the OpenAI Python SDK. Gemini offers an
# OpenAI-compatible endpoint, so only the key, model and base URL change.
PROVIDERS = {
    "openai": {
        "label": "OpenAI", "api_key": OPENAI_API_KEY, "model": OPENAI_MODEL, "base_url": None,
        "key_name": "OPENAI_API_KEY", "billing_url": "platform.openai.com → Settings → Billing",
        "default_rpm": 0,
    },
    "gemini": {
        "label": "Google Gemini", "api_key": GEMINI_API_KEY, "model": GEMINI_MODEL,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_name": "GEMINI_API_KEY", "billing_url": "aistudio.google.com",
        "default_rpm": 8,  # stay under the free tier's per-minute limit
    },
}

AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").strip().lower()
if AI_PROVIDER not in PROVIDERS:
    raise ValueError(f"AI_PROVIDER must be one of {list(PROVIDERS)}, got '{AI_PROVIDER}'.")

_active = PROVIDERS[AI_PROVIDER]
AI_PROVIDER_LABEL = _active["label"]
AI_API_KEY = _active["api_key"]
AI_MODEL = _active["model"]
AI_BASE_URL = _active["base_url"]
AI_KEY_NAME = _active["key_name"]
AI_BILLING_URL = _active["billing_url"]
# 0 = no pacing. Free tiers limit requests per minute, so we space requests out.
REQUESTS_PER_MINUTE = int(os.getenv("AI_REQUESTS_PER_MINUTE", _active["default_rpm"]))

DATABASE_PATH = PROJECT_ROOT / os.getenv("DATABASE_PATH", "data/db/notes.db")
EXPORTS_DIR = PROJECT_ROOT / "exports"


def has_api_key() -> bool:
    return AI_API_KEY.strip() not in PLACEHOLDER_KEYS

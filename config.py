"""Configuration helpers for API keys and environment overrides.

Reads tokens from environment variables or .env file. Defaults to None when not set.
"""
import os
from pathlib import Path


def _load_dotenv(dotenv_path=None):
    """Lightweight .env loader that sets values if not present in env already."""
    path = Path(dotenv_path) if dotenv_path else Path(__file__).resolve().parent / '.env'
    if not path.exists():
        return {}

    values = {}
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v is not None:
                values[k] = v
                # Do not overwrite existing environment variables
                if os.environ.get(k) is None:
                    os.environ[k] = v
    except Exception:
        return {}
    return values


# Attempt to load project .env (non-destructive)
_load_dotenv()

# WAQI / AQICN token (kept for backward compatibility)
WAQI_TOKEN = os.environ.get('WAQI_TOKEN') or os.environ.get('AQICN_TOKEN')

# Placeholder if an OpenAQ token is ever needed in the future
OPENAQ_TOKEN = os.environ.get('OPENAQ_TOKEN')

def get_waqi_token():
    return WAQI_TOKEN

def get_openaq_token():
    return OPENAQ_TOKEN

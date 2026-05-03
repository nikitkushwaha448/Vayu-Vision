"""
Configuration loader for Air-Pulse
Loads environment variables from .env file
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

# Get WAQI token from environment
WAQI_TOKEN = os.getenv("WAQI_TOKEN", "")

# API endpoints
OPENAQ_BASE_URL = "https://api.openaq.org/v2"
WAQI_BASE_URL = "https://api.waqi.info"

def get_waqi_token():
    """Get WAQI API token"""
    return WAQI_TOKEN

def get_openaq_url():
    """Get OpenAQ base URL"""
    return OPENAQ_BASE_URL

def get_waqi_url():
    """Get WAQI base URL"""
    return WAQI_BASE_URL

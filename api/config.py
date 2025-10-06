import os
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).parent.parent

# Model configuration
MODEL_CONFIG = {
    "tts-1": {
        "backbone": "neuphonic/neutts-air-q4-gguf",  # Use quantized model for better speed
        "codec": "neuphonic/neucodec"  # Use full codec for encoding and decoding
    },
    "tts-1-hd": {
        "backbone": "neuphonic/neutts-air",  # Use full model for better quality
        "codec": "neuphonic/neucodec"
    }
}

# Voice configuration - Map OpenAI voice names to local reference audio
VOICE_CONFIG = {
    # Male voices
    "alloy": {
        "ref_audio": str(BASE_DIR / "samples" / "dave.wav"),
        "ref_text": str(BASE_DIR / "samples" / "dave.txt"),
        "description": "Neutral male voice",
        "gender": "male"
    },
    "echo": {
        "ref_audio": str(BASE_DIR / "samples" / "dave.wav"),
        "ref_text": str(BASE_DIR / "samples" / "dave.txt"),
        "description": "Warm male voice",
        "gender": "male"
    },
    # Female voices
    "onyx": {
        "ref_audio": str(BASE_DIR / "samples" / "jo.wav"),
        "ref_text": str(BASE_DIR / "samples" / "jo.txt"),
        "description": "Deep female voice",
        "gender": "female"
    },

    "fable": {
        "ref_audio": str(BASE_DIR / "samples" / "jo.wav"),
        "ref_text": str(BASE_DIR / "samples" / "jo.txt"),
        "description": "Expressive female voice",
        "gender": "female"
    },
    "nova": {
        "ref_audio": str(BASE_DIR / "samples" / "jo.wav"),
        "ref_text": str(BASE_DIR / "samples" / "jo.txt"),
        "description": "Energetic female voice",
        "gender": "female"
    },
    "shimmer": {
        "ref_audio": str(BASE_DIR / "samples" / "jo.wav"),
        "ref_text": str(BASE_DIR / "samples" / "jo.txt"),
        "description": "Gentle female voice",
        "gender": "female"
    }
}

# Server configuration
SERVER_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "1")),
}

# Device configuration
DEVICE_CONFIG = {
    "backbone_device": os.getenv("BACKBONE_DEVICE", "cpu"),
    "codec_device": os.getenv("CODEC_DEVICE", "cpu"),
}

# API Key configuration
# Read API Keys from environment variable (comma-separated for multiple keys)
API_KEYS_ENV = os.getenv("API_KEYS", "")
if API_KEYS_ENV:
    API_KEYS = set(key.strip() for key in API_KEYS_ENV.split(",") if key.strip())
else:
    # If environment variable not set, use default config (for development)
    # IMPORTANT: Set environment variable in production!
    API_KEYS = {
        "sk-neutts-demo-key-123456",  # Demo key 1
        "sk-neutts-demo-key-789012",  # Demo key 2
    }

# Enable API Key authentication
# Set to False to disable authentication (for development/testing only)
ENABLE_API_KEY_AUTH = os.getenv("ENABLE_API_KEY_AUTH", "true").lower() in ("true", "1", "yes")

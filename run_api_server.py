#!/usr/bin/env python3
"""
Start OpenAI-compatible NeuTTS Air API Server

Usage:
    python run_api_server.py
    
Environment Variables:
    API_HOST: Server host address (default: 0.0.0.0)
    API_PORT: Server port (default: 8000)
    BACKBONE_DEVICE: Backbone model device (default: cpu)
    CODEC_DEVICE: Codec device (default: cpu)
    API_KEYS: API keys (comma-separated for multiple keys)
    ENABLE_API_KEY_AUTH: Enable API Key authentication (default: true)
    PRELOAD_ON_STARTUP: Preload reference audio on startup (default: true)
"""

import os
from pathlib import Path

# Try to load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Config file loaded: {env_path}")
except ImportError:
    # If python-dotenv not installed, only read from system environment variables
    pass

import uvicorn
from api.config import SERVER_CONFIG, ENABLE_API_KEY_AUTH

if __name__ == "__main__":
    print("=" * 60)
    print("NeuTTS Air - OpenAI Compatible API Server")
    print("=" * 60)
    print(f"Server: http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
    print(f"API Docs: http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}/docs")
    print(f"Authentication: {'Enabled ✓' if ENABLE_API_KEY_AUTH else 'Disabled (Dev Mode)'}")
    if ENABLE_API_KEY_AUTH:
        from api.config import API_KEYS
        print(f"Configured {len(API_KEYS)} API Key(s)")
    print("=" * 60)
    
    uvicorn.run(
        "api.server:app",
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        reload=False,
        log_level="info"
    )

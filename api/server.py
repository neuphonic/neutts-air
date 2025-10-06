import os
from fastapi import FastAPI, HTTPException, Response, Depends
from fastapi.responses import StreamingResponse
from neuttsair.neutts import NeuTTSAir
import numpy as np
from pathlib import Path
import io

from .models import SpeechRequest, ErrorResponse
from .config import MODEL_CONFIG, VOICE_CONFIG, DEVICE_CONFIG, ENABLE_API_KEY_AUTH
from .utils import adjust_speed, convert_audio_format, get_media_type
from .auth import verify_api_key


app = FastAPI(
    title="NeuTTS Air - OpenAI Compatible API",
    description="OpenAI-compatible Text-to-Speech API powered by NeuTTS Air",
    version="1.0.0"
)

# Global TTS instance cache
tts_instances = {}

# Global reference audio encoding cache
reference_codes_cache = {}

# Enable preload on startup
PRELOAD_ON_STARTUP = os.getenv("PRELOAD_ON_STARTUP", "true").lower() in ("true", "1", "yes")


def get_tts_instance(model: str) -> NeuTTSAir:
    """
    Get or create TTS instance (with cache)
    
    Args:
        model: Model name
    
    Returns:
        NeuTTSAir instance
    """
    if model not in tts_instances:
        config = MODEL_CONFIG.get(model)
        if not config:
            raise ValueError(f"Unsupported model: {model}")
        
        print(f"Loading model: {model}")
        tts_instances[model] = NeuTTSAir(
            backbone_repo=config["backbone"],
            backbone_device=DEVICE_CONFIG["backbone_device"],
            codec_repo=config["codec"],
            codec_device=DEVICE_CONFIG["codec_device"]
        )
        print(f"Model loaded: {model}")
    
    return tts_instances[model]


def preload_reference_audios():
    """
    Preload all reference audio encodings
    
    Called at server startup to avoid encoding delay on first request
    """
    print("\n" + "=" * 60)
    print("Preloading reference audio encodings...")
    print("=" * 60)
    
    # Get all unique reference audio files
    unique_audio_files = {}
    for voice_id, config in VOICE_CONFIG.items():
        ref_audio = config["ref_audio"]
        if ref_audio not in unique_audio_files:
            unique_audio_files[ref_audio] = []
        unique_audio_files[ref_audio].append(voice_id)
    
    print(f"Found {len(unique_audio_files)} unique reference audio files")
    print()
    
    # Use default model for encoding
    default_model = "tts-1"
    
    try:
        # Get TTS instance
        tts = get_tts_instance(default_model)
        
        # Encode each unique reference audio
        for ref_audio, voice_ids in unique_audio_files.items():
            print(f"Encoding: {Path(ref_audio).name}")
            print(f"  For voices: {', '.join(voice_ids)}")
            
            try:
                # Encode reference audio
                ref_codes = tts.encode_reference(ref_audio)
                
                # Fill cache for all voices using this audio
                for voice_id in voice_ids:
                    cache_key = f"{default_model}_{voice_id}"
                    reference_codes_cache[cache_key] = ref_codes
                    print(f"  ✓ Cached: {cache_key}")
                
                print()
                
            except Exception as e:
                print(f"  ✗ Encoding failed: {e}")
                print()
        
        print("=" * 60)
        print(f"✓ Preload complete! Cached {len(reference_codes_cache)} voice encodings")
        print("=" * 60)
        print()
        
    except Exception as e:
        print(f"✗ Preload failed: {e}")
        print("Will encode on first request")
        print()


@app.on_event("startup")
async def startup_event():
    """Server startup event"""
    if PRELOAD_ON_STARTUP:
        preload_reference_audios()
    else:
        print("\nPreload disabled (PRELOAD_ON_STARTUP=false)")
        print("Reference audio will be encoded on first request\n")


@app.get("/")
async def root():
    """Root path"""
    return {
        "message": "NeuTTS Air - OpenAI 兼容 API",
        "version": "1.0.0",
        "authentication": "enabled" if ENABLE_API_KEY_AUTH else "disabled",
        "endpoints": {
            "speech": "/v1/audio/speech",
            "models": "/v1/models",
            "voices": "/v1/voices"
        }
    }


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models (OpenAI API compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1677610602,
                "owned_by": "neuphonic",
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1677610602,
                "owned_by": "neuphonic",
            }
        ]
    }


@app.get("/v1/voices")
async def list_voices(api_key: str = Depends(verify_api_key)):
    """List available voices"""
    return {
        "voices": [
            {
                "voice_id": voice_id,
                "name": voice_id.capitalize(),
                "description": config["description"],
                "gender": config.get("gender", "unknown")
            }
            for voice_id, config in VOICE_CONFIG.items()
        ]
    }


@app.post("/v1/audio/speech")
async def create_speech(
    request: SpeechRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Create speech (OpenAI API compatible)
    
    POST /v1/audio/speech
    {
        "model": "tts-1",
        "input": "今天天气不错",
        "voice": "alloy",
        "response_format": "mp3",
        "speed": 1.0
    }
    """
    try:
        # Validate input text length
        input_length = len(request.input)
        if input_length > 10000:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"Input text too long ({input_length} characters). Recommended length is up to 10000 characters for best results. For longer texts, please consider splitting into segments.",
                        "type": "invalid_request_error",
                        "code": "text_too_long"
                    }
                }
            )
        
        # Validate voice
        if request.voice not in VOICE_CONFIG:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported voice: {request.voice}. Supported voices: {', '.join(VOICE_CONFIG.keys())}"
            )
        
        # Get voice configuration
        voice_config = VOICE_CONFIG[request.voice]
        ref_audio_path = voice_config["ref_audio"]
        ref_text_path = voice_config["ref_text"]
        
        # Check if reference file exists
        if not os.path.exists(ref_audio_path):
            raise HTTPException(
                status_code=500,
                detail=f"Reference audio file not found: {ref_audio_path}"
            )
        
        # Read reference text
        if os.path.exists(ref_text_path):
            with open(ref_text_path, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
        else:
            # Use default text if reference text file not found
            ref_text = "This is a reference audio sample."
        
        # Get TTS instance
        tts = get_tts_instance(request.model)
        
        # Use cached reference audio encoding (avoid re-encoding on every request)
        cache_key = f"{request.model}_{request.voice}"
        if cache_key not in reference_codes_cache:
            print(f"Encoding reference audio (first time): {ref_audio_path}")
            ref_codes = tts.encode_reference(ref_audio_path)
            reference_codes_cache[cache_key] = ref_codes
            print(f"✓ Reference audio cached: {cache_key}")
        else:
            ref_codes = reference_codes_cache[cache_key]
            print(f"✓ Using cached reference audio: {cache_key}")
        
        # Generate speech
        print(f"Generating speech: {request.input[:50]}...")
        wav = tts.infer(request.input, ref_codes, ref_text)
        
        # Adjust speed
        if request.speed != 1.0:
            print(f"Adjusting speed: {request.speed}x")
            wav, sample_rate = adjust_speed(wav, 24000, request.speed)
        else:
            sample_rate = 24000
        
        # Convert audio format
        print(f"Converting format: {request.response_format}")
        audio_bytes = convert_audio_format(wav, sample_rate, request.response_format)
        
        # Return audio
        media_type = get_media_type(request.response_format)
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    from .config import SERVER_CONFIG
    
    uvicorn.run(
        "api.server:app",
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        reload=False
    )

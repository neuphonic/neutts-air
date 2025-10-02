import os
import io
import uuid
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional, List, Union
import soundfile as sf
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import logging

from neuttsair.neutts import NeuTTSAir

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and voice storage
tts_model = None
voice_storage = {}  # Store voice data and transcripts
VOICES_DIR = Path("voices")  # Directory to store voice files

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize models on startup
    load_models()
    # Load existing voices from disk
    load_voices_from_disk()
    yield
    # Cleanup on shutdown (if needed)
    # Clean up temporary files
    for voice_id, voice_data in voice_storage.items():
        temp_path = voice_data.get("temp_path")
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

# Initialize FastAPI app
app = FastAPI(
    title="NeuTTSAir OpenAI-Compatible API",
    description="OpenAI-compatible API for NeuTTSAir text-to-speech",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for OpenAI compatibility
class SpeechRequest(BaseModel):
    model: str = "neutts-air"
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(..., description="Voice name to use")
    response_format: str = Field("mp3", description="Audio format (mp3, wav, etc.)")
    speed: float = Field(1.0, description="Speed of synthesis (0.25-4.0)")

class VoiceResponse(BaseModel):
    voice_id: str
    name: str
    transcript: Optional[str] = None

class VoicesListResponse(BaseModel):
    voices: List[VoiceResponse]

class ErrorDetail(BaseModel):
    message: str
    type: str
    code: Optional[str] = None

class ErrorResponse(BaseModel):
    error: ErrorDetail

def load_models():
    """Load TTS models on GPU and keep them in memory"""
    global tts_model
    
    if tts_model is None:
        logger.info("Loading NeuTTSAir models on GPU...")
        try:
            tts_model = NeuTTSAir(
                backbone_repo="neuphonic/neutts-air",
                backbone_device="cuda" if torch.cuda.is_available() else "cpu",
                codec_repo="neuphonic/neucodec",
                codec_device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info("Models loaded successfully on GPU")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            try:
                tts_model = NeuTTSAir(
                    backbone_repo="neuphonic/neutts-air",
                    backbone_device="cpu",
                    codec_repo="neuphonic/neucodec",
                    codec_device="cpu"
                )
                logger.info("Models loaded on CPU as fallback")
            except Exception as cpu_error:
                logger.error(f"Failed to load models on CPU: {str(cpu_error)}")
                raise HTTPException(status_code=500, detail="Failed to load TTS models")

def load_voices_from_disk():
    """Load existing voices from the voices directory"""
    global voice_storage
    
    # Create voices directory if it doesn't exist
    VOICES_DIR.mkdir(exist_ok=True)
    
    # Load voice metadata from JSON files
    for voice_file in VOICES_DIR.glob("*.json"):
        try:
            with open(voice_file, 'r') as f:
                voice_data = json.load(f)
            
            voice_id = voice_data["voice_id"]
            voice_storage[voice_id] = voice_data
            
            # Load reference codes from file
            ref_codes_path = VOICES_DIR / f"{voice_id}_ref_codes.npy"
            if ref_codes_path.exists():
                ref_codes = np.load(ref_codes_path)
                # Convert to tensor if model is on GPU
                if tts_model and hasattr(tts_model.codec, 'device') and tts_model.codec.device.type == 'cuda':
                    ref_codes = torch.tensor(ref_codes).to(tts_model.codec.device)
                voice_storage[voice_id]["ref_codes"] = ref_codes
            
            logger.info(f"Loaded voice from disk: {voice_data['name']} (ID: {voice_id})")
        except Exception as e:
            logger.error(f"Failed to load voice from {voice_file}: {str(e)}")
    
    logger.info(f"Loaded {len(voice_storage)} voices from disk")

def save_voice_to_disk(voice_id: str, voice_data: dict):
    """Save voice data and reference codes to disk"""
    try:
        # Create voices directory if it doesn't exist
        VOICES_DIR.mkdir(exist_ok=True)
        
        # Save voice metadata (without ref_codes which is a numpy array)
        metadata = voice_data.copy()
        if "ref_codes" in metadata:
            del metadata["ref_codes"]
        
        metadata_path = VOICES_DIR / f"{voice_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save reference codes as numpy array
        if "ref_codes" in voice_data:
            ref_codes_path = VOICES_DIR / f"{voice_id}_ref_codes.npy"
            ref_codes = voice_data["ref_codes"]
            # Move tensor to CPU if it's on GPU
            if isinstance(ref_codes, torch.Tensor):
                ref_codes = ref_codes.cpu()
            np.save(ref_codes_path, ref_codes)
        
        logger.info(f"Saved voice to disk: {voice_data['name']} (ID: {voice_id})")
    except Exception as e:
        logger.error(f"Failed to save voice to disk: {str(e)}")
        raise

def delete_voice_from_disk(voice_id: str):
    """Delete voice files from disk"""
    try:
        # Delete metadata file
        metadata_path = VOICES_DIR / f"{voice_id}.json"
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Delete reference codes file
        ref_codes_path = VOICES_DIR / f"{voice_id}_ref_codes.npy"
        if ref_codes_path.exists():
            ref_codes_path.unlink()
        
        logger.info(f"Deleted voice from disk: {voice_id}")
    except Exception as e:
        logger.error(f"Failed to delete voice from disk: {str(e)}")
        raise

def preprocess_text(text: str) -> str:
    """
    Preprocess text to reduce phonemizer warnings and improve TTS quality
    """
    # Remove or replace problematic characters
    text = text.replace('...', '.')
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Ensure text ends with proper punctuation
    if text and text[-1] not in '.!?;:':
        text += '.'
    
    return text

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "NeuTTSAir OpenAI-Compatible API", "version": "1.0.0"}

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "neutts-air",
                "object": "model",
                "created": 1677610602,
                "owned_by": "neuphonic"
            }
        ]
    }

@app.post("/v1/audio/voice", response_model=VoiceResponse)
async def upload_voice(
    voice: UploadFile = File(..., description="Audio file for voice cloning"),
    name: str = Form(..., description="Name for the voice"),
    transcript: Optional[str] = Form(None, description="Transcript of the audio")
):
    """
    Upload a voice sample and transcript for voice cloning
    OpenAI-compatible endpoint for voice management
    """
    global tts_model, voice_storage
    
    if tts_model is None:
        load_models()
    
    try:
        # Generate unique voice ID
        voice_id = str(uuid.uuid4())
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await voice.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            logger.info(f"Encoding reference audio for voice: {name}")
            ref_codes = tts_model.encode_reference(temp_path)
            
            voice_storage[voice_id] = {
                "name": name,
                "voice_id": voice_id,
                "ref_codes": ref_codes,
                "transcript": transcript,
                "temp_path": temp_path
            }
            
            # Save voice to disk for persistence
            save_voice_to_disk(voice_id, voice_storage[voice_id])
            
            logger.info(f"Voice uploaded successfully: {name} (ID: {voice_id})")
            
            return VoiceResponse(
                voice_id=voice_id,
                name=name,
                transcript=transcript
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Error uploading voice: {str(e)}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to upload voice: {str(e)}")

@app.get("/v1/voices", response_model=VoicesListResponse)
async def list_voices():
    """
    List all available voices
    OpenAI-compatible endpoint
    """
    voices = []
    for voice_id, voice_data in voice_storage.items():
        voices.append(VoiceResponse(
            voice_id=voice_id,
            name=voice_data["name"],
            transcript=voice_data.get("transcript")
        ))
    
    return VoicesListResponse(voices=voices)

@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """
    Generate speech from text using a previously uploaded voice
    OpenAI-compatible endpoint
    """
    global tts_model, voice_storage
    
    if tts_model is None:
        load_models()
    
    voice_data = None
    for v_id, v_data in voice_storage.items():
        if v_data["name"] == request.voice:
            voice_data = v_data
            break
    
    if not voice_data:
        raise HTTPException(
            status_code=404, 
            detail=f"Voice '{request.voice}' not found. Upload it first using /v1/audio/voice"
        )
    
    try:
        ref_codes = voice_data["ref_codes"]
        ref_text = voice_data.get("transcript", "")
        
        # Preprocess text to reduce phonemizer warnings
        input_text = preprocess_text(request.input)
        
        logger.info(f"Generating speech for text: {input_text[:50]}...")
        wav = tts_model.infer(input_text, ref_codes, ref_text)
        
        sample_rate = 24000
        audio_buffer = io.BytesIO()
        
        if request.response_format.lower() == "wav":
            sf.write(audio_buffer, wav, sample_rate, format="WAV")
            media_type = "audio/wav"
        else:  
            sf.write(audio_buffer, wav, sample_rate, format="WAV")
            media_type = "audio/wav"
        
        audio_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(audio_buffer.read()),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")

@app.delete("/v1/audio/voice/{voice_id}")
async def delete_voice(voice_id: str):
    """
    Delete a voice from storage
    """
    global voice_storage
    
    if voice_id not in voice_storage:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    try:
        temp_path = voice_storage[voice_id].get("temp_path")
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        # Delete voice from disk
        delete_voice_from_disk(voice_id)
        
        del voice_storage[voice_id]
        
        return {"message": "Voice deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting voice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete voice: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with OpenAI-compatible error format"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                message=exc.detail,
                type="invalid_request_error"
            )
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with OpenAI-compatible error format"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(
                message="Internal server error",
                type="server_error"
            )
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    uvicorn.run(
        "oai:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )
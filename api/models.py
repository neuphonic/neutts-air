from pydantic import BaseModel, Field
from typing import Literal


class SpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request model"""
    model: Literal["tts-1", "tts-1-hd"] = Field(
        default="tts-1",
        description="TTS model selection"
    )
    input: str = Field(
        ...,
        description="Text to convert to speech",
        min_length=1,
        max_length=32000
    )
    voice: str = Field(
        default="alloy",
        description="Voice selection"
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="Audio output format"
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speech speed (0.25-4.0)"
    )


class ErrorResponse(BaseModel):
    """Error response model"""
    error: dict

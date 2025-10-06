from pydantic import BaseModel, Field
from typing import Literal


class SpeechRequest(BaseModel):
    """OpenAI 兼容的语音合成请求模型"""
    model: Literal["tts-1", "tts-1-hd"] = Field(
        default="tts-1",
        description="TTS 模型选择"
    )
    input: str = Field(
        ...,
        description="要转换为语音的文本",
        min_length=1,
        max_length=32000
    )
    voice: str = Field(
        default="alloy",
        description="语音选择"
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="音频输出格式"
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="语音速度 (0.25-4.0)"
    )


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: dict

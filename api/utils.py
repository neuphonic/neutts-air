import io
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from typing import Tuple


def adjust_speed(audio: np.ndarray, sample_rate: int, speed: float) -> Tuple[np.ndarray, int]:
    """
    调整音频速度
    
    Args:
        audio: 音频数据
        sample_rate: 采样率
        speed: 速度倍数 (0.25-4.0)
    
    Returns:
        调整后的音频数据和采样率
    """
    if speed == 1.0:
        return audio, sample_rate
    
    # 使用 librosa 进行时间拉伸
    import librosa
    audio_stretched = librosa.effects.time_stretch(audio, rate=speed)
    return audio_stretched, sample_rate


def convert_audio_format(
    audio: np.ndarray, 
    sample_rate: int, 
    output_format: str
) -> bytes:
    """
    转换音频格式
    
    Args:
        audio: 音频数据 (numpy array)
        sample_rate: 采样率
        output_format: 输出格式 (mp3, opus, aac, flac, wav, pcm)
    
    Returns:
        转换后的音频字节数据
    """
    # 先转换为 WAV 格式的字节流
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio, sample_rate, format='WAV')
    wav_buffer.seek(0)
    
    if output_format == "wav":
        return wav_buffer.read()
    
    if output_format == "pcm":
        # 返回原始 PCM 数据
        return (audio * 32767).astype(np.int16).tobytes()
    
    # 使用 pydub 转换其他格式
    audio_segment = AudioSegment.from_wav(wav_buffer)
    
    output_buffer = io.BytesIO()
    
    format_mapping = {
        "mp3": "mp3",
        "opus": "opus",
        "aac": "adts",  # AAC with ADTS headers
        "flac": "flac"
    }
    
    export_format = format_mapping.get(output_format, "mp3")
    
    # 设置导出参数
    export_params = {
        "format": export_format,
        "bitrate": "128k" if output_format == "mp3" else None,
    }
    
    # 移除 None 值
    export_params = {k: v for k, v in export_params.items() if v is not None}
    
    audio_segment.export(output_buffer, **export_params)
    output_buffer.seek(0)
    
    return output_buffer.read()


def get_media_type(format: str) -> str:
    """
    获取 HTTP Content-Type
    
    Args:
        format: 音频格式
    
    Returns:
        MIME type
    """
    media_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm"
    }
    return media_types.get(format, "audio/mpeg")

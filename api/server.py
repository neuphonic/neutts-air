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
    title="NeuTTS Air - OpenAI 兼容 API",
    description="兼容 OpenAI 语音 API 的 NeuTTS Air 服务",
    version="1.0.0"
)

# 全局 TTS 实例缓存
tts_instances = {}

# 全局参考音频编码缓存
reference_codes_cache = {}

# 是否启用启动时预加载
PRELOAD_ON_STARTUP = os.getenv("PRELOAD_ON_STARTUP", "true").lower() in ("true", "1", "yes")


def get_tts_instance(model: str) -> NeuTTSAir:
    """
    获取或创建 TTS 实例（带缓存）
    
    Args:
        model: 模型名称
    
    Returns:
        NeuTTSAir 实例
    """
    if model not in tts_instances:
        config = MODEL_CONFIG.get(model)
        if not config:
            raise ValueError(f"不支持的模型: {model}")
        
        print(f"正在加载模型: {model}")
        tts_instances[model] = NeuTTSAir(
            backbone_repo=config["backbone"],
            backbone_device=DEVICE_CONFIG["backbone_device"],
            codec_repo=config["codec"],
            codec_device=DEVICE_CONFIG["codec_device"]
        )
        print(f"模型加载完成: {model}")
    
    return tts_instances[model]


def preload_reference_audios():
    """
    预加载所有参考音频编码
    
    在服务器启动时调用，避免首次请求时的编码延迟
    """
    print("\n" + "=" * 60)
    print("预加载参考音频编码...")
    print("=" * 60)
    
    # 获取所有唯一的参考音频文件
    unique_audio_files = {}
    for voice_id, config in VOICE_CONFIG.items():
        ref_audio = config["ref_audio"]
        if ref_audio not in unique_audio_files:
            unique_audio_files[ref_audio] = []
        unique_audio_files[ref_audio].append(voice_id)
    
    print(f"发现 {len(unique_audio_files)} 个唯一的参考音频文件")
    print()
    
    # 使用默认模型进行编码
    default_model = "tts-1"
    
    try:
        # 获取 TTS 实例
        tts = get_tts_instance(default_model)
        
        # 编码每个唯一的参考音频
        for ref_audio, voice_ids in unique_audio_files.items():
            print(f"编码: {Path(ref_audio).name}")
            print(f"  用于语音: {', '.join(voice_ids)}")
            
            try:
                # 编码参考音频
                ref_codes = tts.encode_reference(ref_audio)
                
                # 为所有使用此音频的语音填充缓存
                for voice_id in voice_ids:
                    cache_key = f"{default_model}_{voice_id}"
                    reference_codes_cache[cache_key] = ref_codes
                    print(f"  ✓ 已缓存: {cache_key}")
                
                print()
                
            except Exception as e:
                print(f"  ✗ 编码失败: {e}")
                print()
        
        print("=" * 60)
        print(f"✓ 预加载完成！已缓存 {len(reference_codes_cache)} 个语音编码")
        print("=" * 60)
        print()
        
    except Exception as e:
        print(f"✗ 预加载失败: {e}")
        print("将在首次请求时进行编码")
        print()


@app.on_event("startup")
async def startup_event():
    """服务器启动事件"""
    if PRELOAD_ON_STARTUP:
        preload_reference_audios()
    else:
        print("\n预加载已禁用（PRELOAD_ON_STARTUP=false）")
        print("参考音频将在首次请求时编码\n")


@app.get("/")
async def root():
    """根路径"""
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
    """列出可用的模型（兼容 OpenAI API）"""
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
    """列出可用的语音"""
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
    创建语音（兼容 OpenAI API）
    
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
        # 验证输入文本长度
        input_length = len(request.input)
        if input_length > 10000:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"输入文本过长（{input_length} 字符）。建议输入长度不超过 10000 字符以获得最佳效果。对于长文本，请考虑分段处理。",
                        "type": "invalid_request_error",
                        "code": "text_too_long"
                    }
                }
            )
        
        # 验证语音
        if request.voice not in VOICE_CONFIG:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的语音: {request.voice}. 支持的语音: {', '.join(VOICE_CONFIG.keys())}"
            )
        
        # 获取语音配置
        voice_config = VOICE_CONFIG[request.voice]
        ref_audio_path = voice_config["ref_audio"]
        ref_text_path = voice_config["ref_text"]
        
        # 检查参考文件是否存在
        if not os.path.exists(ref_audio_path):
            raise HTTPException(
                status_code=500,
                detail=f"参考音频文件不存在: {ref_audio_path}"
            )
        
        # 读取参考文本
        if os.path.exists(ref_text_path):
            with open(ref_text_path, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
        else:
            # 如果没有参考文本文件，使用默认文本
            ref_text = "This is a reference audio sample."
        
        # 获取 TTS 实例
        tts = get_tts_instance(request.model)
        
        # 使用缓存的参考音频编码（避免每次请求都重新编码）
        cache_key = f"{request.model}_{request.voice}"
        if cache_key not in reference_codes_cache:
            print(f"编码参考音频（首次）: {ref_audio_path}")
            ref_codes = tts.encode_reference(ref_audio_path)
            reference_codes_cache[cache_key] = ref_codes
            print(f"✓ 参考音频已缓存: {cache_key}")
        else:
            ref_codes = reference_codes_cache[cache_key]
            print(f"✓ 使用缓存的参考音频: {cache_key}")
        
        # 生成语音
        print(f"生成语音: {request.input[:50]}...")
        wav = tts.infer(request.input, ref_codes, ref_text)
        
        # 调整速度
        if request.speed != 1.0:
            print(f"调整速度: {request.speed}x")
            wav, sample_rate = adjust_speed(wav, 24000, request.speed)
        else:
            sample_rate = 24000
        
        # 转换音频格式
        print(f"转换格式: {request.response_format}")
        audio_bytes = convert_audio_format(wav, sample_rate, request.response_format)
        
        # 返回音频
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
        print(f"错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
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

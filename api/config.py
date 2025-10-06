import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 模型配置
MODEL_CONFIG = {
    "tts-1": {
        "backbone": "neuphonic/neutts-air-q4-gguf",  # 使用量化模型提高速度
        "codec": "neuphonic/neucodec"  # 使用完整 codec 来编码和解码
    },
    "tts-1-hd": {
        "backbone": "neuphonic/neutts-air",  # 使用完整模型获得更高质量
        "codec": "neuphonic/neucodec"
    }
}

# 语音配置 - 映射 OpenAI 语音名称到本地参考音频
VOICE_CONFIG = {
    # 男声组
    "alloy": {
        "ref_audio": str(BASE_DIR / "samples" / "dave.wav"),
        "ref_text": str(BASE_DIR / "samples" / "dave.txt"),
        "description": "中性男声",
        "gender": "male"
    },
    "echo": {
        "ref_audio": str(BASE_DIR / "samples" / "dave.wav"),
        "ref_text": str(BASE_DIR / "samples" / "dave.txt"),
        "description": "温暖男声",
        "gender": "male"
    },
    # 女声组
    "onyx": {
        "ref_audio": str(BASE_DIR / "samples" / "jo.wav"),
        "ref_text": str(BASE_DIR / "samples" / "jo.txt"),
        "description": "深沉女声",
        "gender": "female"
    },

    "fable": {
        "ref_audio": str(BASE_DIR / "samples" / "jo.wav"),
        "ref_text": str(BASE_DIR / "samples" / "jo.txt"),
        "description": "富有表现力的女声",
        "gender": "female"
    },
    "nova": {
        "ref_audio": str(BASE_DIR / "samples" / "jo.wav"),
        "ref_text": str(BASE_DIR / "samples" / "jo.txt"),
        "description": "活力女声",
        "gender": "female"
    },
    "shimmer": {
        "ref_audio": str(BASE_DIR / "samples" / "jo.wav"),
        "ref_text": str(BASE_DIR / "samples" / "jo.txt"),
        "description": "柔和女声",
        "gender": "female"
    }
}

# 服务器配置
SERVER_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "1")),
}

# 设备配置
DEVICE_CONFIG = {
    "backbone_device": os.getenv("BACKBONE_DEVICE", "cpu"),
    "codec_device": os.getenv("CODEC_DEVICE", "cpu"),
}

# API Key 配置
# 从环境变量读取 API Keys（逗号分隔多个 key）
API_KEYS_ENV = os.getenv("API_KEYS", "")
if API_KEYS_ENV:
    API_KEYS = set(key.strip() for key in API_KEYS_ENV.split(",") if key.strip())
else:
    # 如果环境变量未设置，使用默认配置（用于开发环境）
    # 生产环境请务必设置环境变量！
    API_KEYS = {
        "sk-neutts-demo-key-123456",  # 示例 key 1
        "sk-neutts-demo-key-789012",  # 示例 key 2
    }

# 是否启用 API Key 验证
# 设置为 False 可以禁用验证（仅用于开发/测试）
ENABLE_API_KEY_AUTH = os.getenv("ENABLE_API_KEY_AUTH", "true").lower() in ("true", "1", "yes")

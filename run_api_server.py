#!/usr/bin/env python3
"""
启动 OpenAI 兼容的 NeuTTS Air API 服务器

使用方法:
    python run_api_server.py
    
环境变量:
    API_HOST: 服务器主机地址 (默认: 0.0.0.0)
    API_PORT: 服务器端口 (默认: 8000)
    BACKBONE_DEVICE: 主干模型设备 (默认: cpu)
    CODEC_DEVICE: 编解码器设备 (默认: cpu)
    API_KEYS: API 密钥（逗号分隔多个）
    ENABLE_API_KEY_AUTH: 是否启用 API Key 验证 (默认: true)
    PRELOAD_ON_STARTUP: 启动时预加载参考音频 (默认: true)
"""

import os
from pathlib import Path

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ 已加载配置文件: {env_path}")
except ImportError:
    # 如果没有安装 python-dotenv，只从系统环境变量读取
    pass

import uvicorn
from api.config import SERVER_CONFIG, ENABLE_API_KEY_AUTH

if __name__ == "__main__":
    print("=" * 60)
    print("NeuTTS Air - OpenAI 兼容 API 服务器")
    print("=" * 60)
    print(f"服务器地址: http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
    print(f"API 文档: http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}/docs")
    print(f"API 认证: {'已启用 ✓' if ENABLE_API_KEY_AUTH else '已禁用 (开发模式)'}")
    if ENABLE_API_KEY_AUTH:
        from api.config import API_KEYS
        print(f"已配置 {len(API_KEYS)} 个 API Key")
    print("=" * 60)
    
    uvicorn.run(
        "api.server:app",
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        reload=False,
        log_level="info"
    )

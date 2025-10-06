#!/usr/bin/env python3
"""
预编码参考音频脚本

该脚本会预先编码所有参考音频，生成 .pt 文件，
这样 API 服务器启动时可以直接加载，无需每次请求时编码。

使用方法:
    python scripts/preencode_references.py
"""

import os
import sys
from pathlib import Path
import torch

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neuttsair.neutts import NeuTTSAir
from api.config import VOICE_CONFIG


def main():
    print("=" * 60)
    print("预编码参考音频")
    print("=" * 60)
    print()
    
    # 创建输出目录
    output_dir = project_root / "voices_encoded"
    output_dir.mkdir(exist_ok=True)
    print(f"输出目录: {output_dir}")
    print()
    
    # 初始化编码器（只需要 codec）
    print("正在加载编码器...")
    tts = NeuTTSAir(
        backbone_repo="neuphonic/neutts-air-q4-gguf",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu"
    )
    print("✓ 编码器加载完成")
    print()
    
    # 处理每个语音
    results = []
    for voice_id, config in VOICE_CONFIG.items():
        ref_audio_path = config["ref_audio"]
        
        if not os.path.exists(ref_audio_path):
            print(f"⚠️  跳过 {voice_id}: 文件不存在 - {ref_audio_path}")
            continue
        
        print(f"处理: {voice_id}")
        print(f"  输入: {ref_audio_path}")
        
        try:
            # 编码
            ref_codes = tts.encode_reference(ref_audio_path)
            
            # 保存
            output_path = output_dir / f"{voice_id}.pt"
            torch.save(ref_codes, output_path)
            
            print(f"  ✓ 已保存: {output_path}")
            print(f"  编码形状: {ref_codes.shape}")
            results.append((voice_id, output_path, True))
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            results.append((voice_id, None, False))
        
        print()
    
    # 打印摘要
    print("=" * 60)
    print("编码完成摘要")
    print("=" * 60)
    success_count = sum(1 for _, _, success in results if success)
    print(f"总计: {len(results)} 个语音")
    print(f"成功: {success_count}")
    print(f"失败: {len(results) - success_count}")
    print()
    
    if success_count > 0:
        print("成功编码的语音:")
        for voice_id, output_path, success in results:
            if success:
                print(f"  ✓ {voice_id}: {output_path}")
        print()
        print("提示: 现在可以在 api/config.py 中配置使用预编码文件")


if __name__ == "__main__":
    main()

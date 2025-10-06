"""
长文本分段处理模块
"""

import re
from typing import List


def split_text_by_sentences(text: str, max_length: int = 500) -> List[str]:
    """
    按句子分割文本，确保每段不超过最大长度
    
    Args:
        text: 输入文本
        max_length: 每段的最大字符数
    
    Returns:
        分割后的文本段落列表
    """
    # 定义句子分隔符（中英文）
    sentence_endings = r'[。！？!?.;；\n]+'
    
    # 按句子分割
    sentences = re.split(f'({sentence_endings})', text)
    
    # 重新组合句子和标点
    combined = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            combined.append(sentences[i] + sentences[i + 1])
        elif i < len(sentences):
            combined.append(sentences[i])
    
    # 合并短句，确保不超过 max_length
    chunks = []
    current_chunk = ""
    
    for sentence in combined:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # 如果当前块加上新句子不超过限制
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            # 保存当前块（如果不为空）
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # 如果单个句子就超过限制，需要强制分割
            if len(sentence) > max_length:
                # 按标点或空格分割长句
                sub_chunks = split_long_sentence(sentence, max_length)
                chunks.extend(sub_chunks[:-1])
                current_chunk = sub_chunks[-1] if sub_chunks else ""
            else:
                current_chunk = sentence
    
    # 添加最后一块
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def split_long_sentence(sentence: str, max_length: int) -> List[str]:
    """
    分割单个长句子
    
    Args:
        sentence: 长句子
        max_length: 最大长度
    
    Returns:
        分割后的片段列表
    """
    # 先尝试按逗号、顿号等分割
    parts = re.split(r'([,，、])', sentence)
    
    chunks = []
    current_chunk = ""
    
    for part in parts:
        if len(current_chunk) + len(part) <= max_length:
            current_chunk += part
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = part
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # 如果还有超长的，就强制按字符数分割
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            # 强制按 max_length 分割
            for i in range(0, len(chunk), max_length):
                final_chunks.append(chunk[i:i + max_length])
    
    return final_chunks


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量
    
    中文字符通常 1 字符 ≈ 1 token
    英文单词通常 1 单词 ≈ 1-2 tokens
    
    Args:
        text: 输入文本
    
    Returns:
        估算的 token 数
    """
    # 简单估算：中文字符数 + 英文单词数 * 1.5
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    
    return int(chinese_chars + english_words * 1.5)

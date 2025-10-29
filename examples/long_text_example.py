from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from neuttsair.neutts import NeuTTSAir


def split_text_into_chunks(text, max_chars_per_chunk=200):
    """
    Split text into smaller chunks for generation.
    This implementation splits by sentences and ensures proper punctuation.
    """
    # Split by sentence endings (. ! ?) while preserving the punctuation
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Filter out empty sentences
    sentences = [s for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Add space between sentences if needed
        separator = " " if current_chunk and not current_chunk.endswith((" ", "\n")) else ""
        sentence_with_sep = separator + sentence if current_chunk else sentence
        
        # Check if adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence_with_sep) > max_chars_per_chunk and current_chunk:
            # Only add chunk if it's not empty
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence_with_sep

    # Add the last chunk if it exists
    if current_chunk and current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _load_text_source(value: str, description: str) -> str:
    if not value:
        raise ValueError(f"{description} must be provided.")

    candidate_path = Path(value).expanduser()
    if candidate_path.exists():
        if candidate_path.is_dir():
            raise ValueError(f"{description} path points to a directory: {candidate_path}")
        text = candidate_path.read_text(encoding="utf-8").strip()
    else:
        text = value.strip()

    if not text:
        raise ValueError(f"{description} must not be empty.")

    return text


def main(input_text, ref_audio_path, ref_text, backbone, output_path="output.wav"):
    if not ref_audio_path or not ref_text:
        print("No reference audio or text provided.")
        return None

    ref_audio_file = Path(ref_audio_path).expanduser()
    if not ref_audio_file.is_file():
        print(f"Reference audio file not found: {ref_audio_file}")
        return None

    try:
        input_text = _load_text_source(input_text, "Input text")
        ref_text = _load_text_source(ref_text, "Reference text")
    except ValueError as exc:
        print(exc)
        return None

    # Initialize NeuTTSAir with the desired model and codec
    tts = NeuTTSAir(
        backbone_repo=backbone,
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu"
    )

    print("Encoding reference audio")
    ref_codes = tts.encode_reference(str(ref_audio_file))
    if isinstance(ref_codes, torch.Tensor):
        ref_code_list: list[int] = (
            ref_codes.detach()
            .to(dtype=torch.long)
            .view(-1)
            .cpu()
            .tolist()
        )
    else:
        ref_code_array = np.asarray(ref_codes)
        ref_code_list = ref_code_array.reshape(-1).astype(int).tolist()

    def ensure_chunk_fits(chunk: str, chunk_ref_text: str, max_prompt_tokens: int) -> list[str]:
        chunk = chunk.strip()
        if not chunk:
            return []

        ref_for_prompt = chunk_ref_text if chunk_ref_text.strip() else ref_text
        prompt_ids = tts._apply_chat_template(ref_code_list, ref_for_prompt, chunk)
        if len(prompt_ids) <= max_prompt_tokens:
            return [chunk]

        words = chunk.split()
        if len(words) <= 1:
            raise ValueError(
                "Unable to fit text chunk within the model context window. Consider reducing input length."
            )

        midpoint = len(words) // 2
        left = " ".join(words[:midpoint]).strip()
        right = " ".join(words[midpoint:]).strip()

        segments: list[str] = []
        if left:
            segments.extend(ensure_chunk_fits(left, chunk_ref_text, max_prompt_tokens))
        if right:
            segments.extend(ensure_chunk_fits(right, chunk_ref_text, max_prompt_tokens))
        return segments

    # Split long text into chunks
    initial_chunks = split_text_into_chunks(input_text, max_chars_per_chunk=100)
    chunk_plan: list[tuple[str, str]] = [
        (chunk, ref_text if idx == 0 else "")
        for idx, chunk in enumerate(initial_chunks)
        if chunk.strip()
    ]

    if getattr(tts, "tokenizer", None) is not None:
        max_prompt_tokens = max(1, getattr(tts, "max_context", 2048) - 256)
        adjusted_plan: list[tuple[str, str]] = []
        for chunk_text, chunk_ref_text in chunk_plan:
            adjusted_plan.extend(
                (segment, chunk_ref_text)
                for segment in ensure_chunk_fits(chunk_text, chunk_ref_text, max_prompt_tokens)
            )
        chunk_plan = [
            (segment_text, segment_ref_text)
            for segment_text, segment_ref_text in adjusted_plan
            if segment_text.strip()
        ]

    if not chunk_plan:
        print("No text to synthesize after chunking.")
        return None

    print(f"Split input text into {len(chunk_plan)} chunks")
    
    # Debug: Print first few chunks to verify splitting
    for i, (chunk, _) in enumerate(chunk_plan[:3]):
        print(f"Chunk {i}: {chunk}")

    # Generate audio for each chunk
    audio_chunks = []
    for i, (chunk, chunk_ref_text) in enumerate(chunk_plan):
        print(f"Generating audio for chunk {i+1}/{len(chunk_plan)}: {chunk[:50]}...")
        try:
            effective_ref_text = chunk_ref_text if chunk_ref_text.strip() else ref_text
            wav = tts.infer(chunk, ref_codes, effective_ref_text)
            audio_chunks.append(wav)
            
            if i < len(chunk_plan) - 1:
                # Add a small pause between chunks (0.5 seconds of silence)
                silence_duration = int(0.5 * tts.sample_rate)
                silence = np.zeros(silence_duration, dtype=wav.dtype)
                audio_chunks.append(silence)
        except Exception as e:
            print(f"Error generating audio for chunk {i+1}: {e}")
            continue

    # Concatenate all audio chunks
    if audio_chunks:
        final_wav = np.concatenate(audio_chunks)
        print(f"Final audio duration: {len(final_wav) / tts.sample_rate:.2f} seconds")
    else:
        print("No audio generated")
        return None

    output_file = Path(output_path).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving concatenated output to {output_file}")
    sf.write(str(output_file), final_wav, tts.sample_rate)


if __name__ == "__main__":
    # get arguments from command line
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTSAir Long Text Example")
    parser.add_argument(
        "--input_text", 
        type=str, 
        required=True, 
        help="Input text to be converted to speech"
    )
    parser.add_argument(
        "--ref_audio", 
        type=str, 
        default="./samples/dave.wav", 
        help="Path to reference audio file"
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default="./samples/dave.txt", 
        help="Reference text corresponding to the reference audio",
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="output.wav", 
        help="Path to save the output audio"
    )
    parser.add_argument(
        "--backbone", 
        type=str, 
        default="neuphonic/neutts-air", 
        help="Huggingface repo containing the backbone checkpoint"
    )
    args = parser.parse_args()
    main(
        input_text=args.input_text,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        backbone=args.backbone,
        output_path=args.output_path,
    )
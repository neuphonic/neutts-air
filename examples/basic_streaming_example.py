import os
import torch
import numpy as np
from neuttsair.neutts import NeuTTSAir
import pyaudio


def main(
    input_text,
    ref_codes_path,
    ref_text,
    backbone,
    *,
    frames_per_chunk: int = 25,
    buffer_seconds: float = 1.5,
):
    assert backbone in ["neuphonic/neutts-air-q4-gguf", "neuphonic/neutts-air-q8-gguf"], "Must be a GGUF ckpt as streaming is only currently supported by llama-cpp."
    
    # Initialize NeuTTSAir with the desired model and codec
    tts = NeuTTSAir(
        backbone_repo=backbone,
        codec_repo="neuphonic/neucodec-onnx-decoder",
        codec_device="auto"
    )

    # Optional tuning for chunk cadence
    frames_per_chunk = max(1, frames_per_chunk)
    tts.streaming_frames_per_chunk = frames_per_chunk
    tts.streaming_stride_samples = frames_per_chunk * tts.hop_length

    # Check if ref_text is a path if it is read it if not just return string
    if ref_text and os.path.exists(ref_text):
        with open(ref_text, "r") as f:
            ref_text = f.read().strip()

    if ref_codes_path and os.path.exists(ref_codes_path):
        ref_codes = torch.load(ref_codes_path)

    print(f"Generating audio for input text: {input_text}")
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24_000,
        output=True
    )
    print("Streaming...")
    buffer_samples = int(max(buffer_seconds, 0.1) * tts.sample_rate)
    pending = np.empty(0, dtype=np.float32)

    for chunk in tts.infer_stream(input_text, ref_codes, ref_text):
        pending = np.concatenate((pending, chunk.astype(np.float32)))

        while pending.size >= buffer_samples:
            to_play = pending[:buffer_samples]
            audio = (to_play * 32767).astype(np.int16)
            stream.write(audio.tobytes())
            pending = pending[buffer_samples:]

    if pending.size:
        audio = (pending * 32767).astype(np.int16)
        stream.write(audio.tobytes())
    
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTSAir Example")
    parser.add_argument(
        "--input_text", 
        type=str, 
        required=True, 
        help="Input text to be converted to speech"
    )
    parser.add_argument(
        "--ref_codes", 
        type=str, 
        default="./samples/dave.pt", 
        help="Path to pre-encoded reference audio"
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
        default="neuphonic/neutts-air-q8-gguf", 
        help="Huggingface repo containing the backbone checkpoint. Must be GGUF."
    )
    parser.add_argument(
        "--frames_per_chunk",
        type=int,
        default=25,
        help="Number of spectrogram frames per streamed chunk (higher = longer segments).",
    )
    parser.add_argument(
        "--buffer_seconds",
        type=float,
        default=1.5,
        help="Seconds of audio to accumulate before pushing to the speaker.",
    )
    args = parser.parse_args()
    main(
        input_text=args.input_text,
        ref_codes_path=args.ref_codes,
        ref_text=args.ref_text,
        backbone=args.backbone,
        frames_per_chunk=args.frames_per_chunk,
        buffer_seconds=args.buffer_seconds,
    )

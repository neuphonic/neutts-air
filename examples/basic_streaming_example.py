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
    backbone_device: str = "auto",
    codec_repo: str = "neuphonic/neucodec-onnx-decoder",
    codec_device: str = "auto",
    *,
    frames_per_chunk: int = 25,
    buffer_seconds: float = 1.5,
):
    assert backbone in ["neuphonic/neutts-air-q4-gguf", "neuphonic/neutts-air-q8-gguf"], "Must be a GGUF ckpt as streaming is only currently supported by llama-cpp."
    
    # Initialize NeuTTSAir with the desired model and codec
    tts = NeuTTSAir(
        backbone_repo=backbone,
        backbone_device=backbone_device,
        codec_repo=codec_repo,
        codec_device=codec_device,
    )

    # Optional tuning for chunk cadence
    frames_per_chunk = max(1, frames_per_chunk)
    tts.streaming_frames_per_chunk = frames_per_chunk
    tts.streaming_stride_samples = frames_per_chunk * tts.hop_length

    # Check if ref_text is a path; if so, remember the path and read the text
    ref_text_path = None
    if ref_text and os.path.exists(ref_text):
        ref_text_path = ref_text
        with open(ref_text, "r") as f:
            ref_text = f.read().strip()

    # Load reference codes if provided. Always define `ref_codes` so
    # callers don't get an UnboundLocalError when the path is missing.
    ref_codes = None
    if ref_codes_path:
        if os.path.exists(ref_codes_path):
            ref_codes = torch.load(ref_codes_path)
        else:
            # If a path was supplied but not found, warn and continue with None.
            print(f"Warning: ref_codes path '{ref_codes_path}' not found — continuing without precomputed codes.")

    # If no precomputed ref_codes were provided, try some sensible fallbacks:
    # 1. If ref_text points to a text file like 'samples/dave.txt', look for
    #    a matching `samples/dave.pt` precomputed tensor.
    # 2. If that fails, look for an audio file with the same stem and call
    #    `tts.encode_reference` to produce ref_codes on-the-fly.
    if ref_codes is None:
        # Attempt to derive a basename from the original ref_text path when present.
        candidate_loaded = False
        if ref_text_path and os.path.exists(ref_text_path):
            base = os.path.splitext(os.path.basename(ref_text_path))[0]
            # search a few sensible directories for a matching precomputed .pt or audio
            search_dirs = [os.path.dirname(ref_text), ".", "samples", "cache", "artifacts", "scripts"]
            for d in search_dirs:
                if not d:
                    continue
                candidate_pt = os.path.join(d, f"{base}.pt")
                if os.path.exists(candidate_pt):
                    print(f"Found fallback precomputed codes at {candidate_pt}; loading.")
                    ref_codes = torch.load(candidate_pt)
                    candidate_loaded = True
                    break

            # if no .pt was found, look for audio files in the same search dirs and encode
            if not candidate_loaded:
                for d in search_dirs:
                    for ext in (".wav", ".flac", ".mp3"):
                        candidate_audio = os.path.join(d, f"{base}{ext}")
                        if os.path.exists(candidate_audio):
                            print(f"Found fallback audio at {candidate_audio}; encoding reference codes on-the-fly.")
                            ref_codes = tts.encode_reference(candidate_audio)
                            candidate_loaded = True
                            break
                    if candidate_loaded:
                        break

        if ref_codes is None:
            print("No reference codes available; proceeding without precomputed reference. This may affect voice similarity.")

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
        "--backbone-device",
        type=str,
        default="auto",
        help="Preferred device for the backbone (auto/cpu/gpu/mps).",
    )
    parser.add_argument(
        "--codec-repo",
        type=str,
        default="neuphonic/neucodec-onnx-decoder",
        help="Codec repo to use (neuphonic/neucodec or neuphonic/neucodec-onnx-decoder)",
    )
    parser.add_argument(
        "--codec-device",
        type=str,
        default="auto",
        help="Codec device/provider hint (auto/cpu/cuda/directml/metal/coreml).",
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
        backbone_device=args.backbone_device,
        codec_repo=args.codec_repo,
        codec_device=args.codec_device,
        frames_per_chunk=args.frames_per_chunk,
        buffer_seconds=args.buffer_seconds,
    )

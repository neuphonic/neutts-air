import os

import numpy as np
import pyaudio

from neuttsair.neutts import NeuTTSAir

try:
    import torch  # type: ignore[assignment]
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def _load_ref_codes(ref_codes_path: str | None):
    if not ref_codes_path:
        return None

    if not os.path.exists(ref_codes_path):
        raise FileNotFoundError(f"Reference codes not found: {ref_codes_path}")

    suffix = os.path.splitext(ref_codes_path)[1].lower()
    if suffix in {".pt", ".pth"}:
        if torch is None:
            raise ImportError(
                "Loading .pt reference codes requires the optional dependency `torch`.\n"
                "Install it separately with `pip install torch` or convert the codes to .npy."
            )
        return torch.load(ref_codes_path, map_location="cpu")

    if suffix == ".npy":
        return np.load(ref_codes_path)

    raise ValueError(f"Unsupported reference codes format '{suffix}'. Expected .pt, .pth, or .npy.")


def main(input_text, ref_codes_path, ref_text, backbone):
    if not ref_codes_path or not ref_text:
        print("No reference codes or text provided.")
        return

    if backbone not in {
        "neuphonic/neutts-air-q4-gguf",
        "neuphonic/neutts-air-q8-gguf",
    }:
        raise ValueError("Streaming requires a GGUF checkpoint compatible with llama-cpp.")

    # Initialize NeuTTSAir with the desired model and codec
    tts = NeuTTSAir(
        backbone_repo=backbone,
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec-onnx-decoder",
        codec_device="cpu",
    )

    # Check if ref_text is a path if it is read it if not just return string
    if ref_text and os.path.exists(ref_text):
        with open(ref_text, "r") as f:
            ref_text = f.read().strip()

    ref_codes = _load_ref_codes(ref_codes_path)

    print(f"Generating audio for input text: {input_text}")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24_000, output=True)
    print("Streaming...")
    for chunk in tts.infer_stream(input_text, ref_codes, ref_text):
        audio = (chunk * 32767).astype(np.int16)
        print(audio.shape)
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
        help="Input text to be converted to speech",
    )
    parser.add_argument(
        "--ref_codes",
        type=str,
        default="./samples/dave.pt",
        help="Path to pre-encoded reference audio",
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
        help="Path to save the output audio",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="neuphonic/neutts-air-q8-gguf",
        help="Huggingface repo containing the backbone checkpoint. Must be GGUF.",
    )
    args = parser.parse_args()
    main(
        input_text=args.input_text,
        ref_codes_path=args.ref_codes,
        ref_text=args.ref_text,
        backbone=args.backbone,
    )

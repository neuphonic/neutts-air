import os
from pathlib import Path

import soundfile as sf
import torch
import numpy as np
from neuttsair.neutts import NeuTTSAir


def main(input_text, ref_codes_path, ref_text, backbone, output_path="output.wav"):
    if not ref_codes_path or not ref_text:
        print("No reference audio or text provided.")
        return None

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

    if ref_codes_path and os.path.exists(ref_codes_path):
        ref_codes = _load_ref_codes(ref_codes_path)
    else:
        raise FileNotFoundError(f"Reference codes not found: {ref_codes_path}")

    print(f"Generating audio for input text: {input_text}")
    wav = tts.infer(input_text, ref_codes, ref_text)

    print(f"Saving output to {output_path}")
    sf.write(output_path, wav, 24000)


if __name__ == "__main__":
    # get arguments from command line
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
        default="neuphonic/neutts-air",
        help="Huggingface repo containing the backbone checkpoint",
    )
    args = parser.parse_args()
    main(
        input_text=args.input_text,
        ref_codes_path=args.ref_codes,
        ref_text=args.ref_text,
        backbone=args.backbone,
        output_path=args.output_path,
    )


def _load_ref_codes(ref_codes_path: str):
    path = Path(ref_codes_path)
    suffix = path.suffix.lower()

    if suffix in {".pt", ".pth"}:
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "Loading '.pt' reference codes requires torch >= 2.0 with weights_only support "
                "to avoid executing arbitrary pickle payloads."
            ) from exc
    elif suffix == ".npy":
        data = np.load(path, allow_pickle=False)
    else:
        raise ValueError(
            f"Unsupported reference code format '{path.suffix}'. Expected .pt, .pth, or .npy"
        )

    if isinstance(data, torch.Tensor):
        return data.detach().cpu().view(-1).tolist()
    if isinstance(data, np.ndarray):
        return data.reshape(-1).tolist()
    if isinstance(data, (list, tuple)):
        return [int(val) for val in data]

    raise ValueError("Reference code file must contain tensor-like data.")

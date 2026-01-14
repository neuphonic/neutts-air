import torch
import numpy as np
import pytest
from neuttsair import NeuTTSAir


BACKBONES = [
    "neuphonic/neutts-air",
    "neuphonic/neutts-air-q8-gguf",
    "neuphonic/neutts-air-q4-gguf",
    "neuphonic/neutts-nano",
    "neuphonic/neutts-nano-q8-gguf",
    "neuphonic/neutts-nano-q4-gguf",
]
GGUF_BACKBONES = [
    i for i in BACKBONES 
    if i.endswith(".gguf")
]
CODECS = [
    "neuphonic/neucodec",
    "neuphonic/distill-neucodec",
    "neuphonic/neucodec-onnx-decoder",
]


@pytest.fixture()
def reference_data() -> tuple[torch.Tensor, str]:
    ref_codes = torch.load("./samples/dave.pt")
    with open("./samples/dave.txt", "r") as f:
        ref_text = f.read()
    return ref_codes, ref_text


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("codec", CODECS)
def test_model_loading_and_inference(backbone, codec, reference_data, tmp_path):
    """
    Tests every combination of backbone and codec.
    Ensures a valid numpy audio array is returned.
    """
    ref_codes, ref_text = reference_data
    input_text = "Testing."

    try:
        model = NeuTTSAir(
            backbone_repo=backbone,
            backbone_device="cpu",
            codec_repo=codec,
            codec_device="cpu"
        )
    except Exception as e:
        pytest.fail(f"Failed to load combination {backbone} + {codec}: {e}")

    audio = model.infer(text=input_text, ref_codes=ref_codes, ref_text=ref_text)
    assert isinstance(audio, np.ndarray), "Output should be a numpy array"
    assert len(audio) > 0, "Generated audio should not be empty"
    assert not np.isnan(audio).any(), "Audio contains NaN values"
    assert audio.dtype in [np.float32, np.float64]
    
    print(f"Successfully generated {len(audio)/24000:.2f}s of audio for {codec}")


@pytest.mark.parametrize("backbone", GGUF_BACKBONES)
@pytest.mark.parametrize("codec", CODECS)
def test_streaming_ggml(backbone, codec, reference_data):
    ref_codes, ref_text = reference_data

    try:
        model = NeuTTSAir(
            backbone_repo=backbone,
            backbone_device="cpu",
            codec_repo=codec,
            codec_device="cpu"
        )
    except Exception as e:
        pytest.fail(f"Failed to load combination {backbone} + {codec}: {e}")
    
    gen = model.infer_stream("This is a streaming test that should be comprised of multple chunks.", ref_codes, ref_text)
    
    chunks = []
    for chunk in gen:
        assert isinstance(chunk, np.ndarray)
        chunks.append(chunk)
        
    assert len(chunks) > 0, "Stream yielded no audio chunks"
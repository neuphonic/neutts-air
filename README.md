# NeuTTS

HuggingFace 🤗:
- NeuTTS-Air* [Model](https://huggingface.co/neuphonic/neutts-air), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-air-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-air-q4-gguf) [Spaces](https://huggingface.co/spaces/neuphonic/neutts-air)
- NeuTTS-Nano [Model](https://huggingface.co/neuphonic/neutts-nano), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-nano-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-nano-q4-gguf)

[NeuTTS-Air Demo Video](https://github.com/user-attachments/assets/020547bc-9e3e-440f-b016-ae61ca645184)

_Created by [Neuphonic](http://neuphonic.com/) - building faster, smaller, on-device voice AI_

State-of-the-art Voice AI has been locked behind web APIs for too long. NeuTTS is a collection of open source, on-device, TTS speech language models with instant voice cloning. Built off of LLM backbones, NeuTTS brings natural-sounding speech, real-time performance, built-in security and speaker cloning to your local device - unlocking a new category of embedded voice agents, assistants, toys, and compliance-safe apps.

## Key Features

- 🗣Best-in-class realism for its size - produces natural, ultra-realistic voices that sound human
- 📱Optimised for on-device deployment - provided in GGML format, ready to run on phones, laptops, or even Raspberry Pis
- 👫Instant voice cloning - create your own speaker with as little as 3 seconds of audio
- 🚄Simple LM + codec architecture built - the sweet spot between speed, size, and quality for real-world applications

> [!CAUTION]
> Websites like neutts.com are popping up and they're not affliated with Neuphonic, our github or this repo.
>
> We are on neuphonic.com only. Please be careful out there! 🙏

## Model Details

NeuTTS models are built from small LLM backbones - lightweight yet capable language models optimised for text understanding and generation - as well as a powerful combination of technologies designed for efficiency and quality:

- **Supported Languages**: English
- **Audio Codec**: [NeuCodec](https://huggingface.co/neuphonic/neucodec) - our 50hz neural audio codec that achieves exceptional audio quality at low bitrates using a single codebook
- **Context Window**: 2048 tokens, enough for processing ~30 seconds of audio (including prompt duration)
- **Format**: Available in GGML format for efficient on-device inference
- **Responsibility**: Watermarked outputs
- **Inference Speed**: Real-time generation on mid-range devices
- **Power Consumption**: Optimised for mobile and embedded devices

## Benchmarking Models

The two models were benhcmarked using the Q4 quantisation [neutts-air-Q4-0](https://huggingface.co/neuphonic/neutts-air-q4-gguf) and [neutts-nano-Q4-0](https://huggingface.co/neuphonic/neutts-nano-q4-gguf).
Benchmarks on CPU were run through llama-bench (llama.cpp) to measure prefill and decode throughput at multiple context sizes. We report CPU throughput number below using 500 prefill tokens and generating 250 tokens.

For GPU's (specifically RTX 4090), we leverage vLLM to maximise throughpu. We run benchmarks using the [vLLM benchmark](https://docs.vllm.ai/en/stable/cli/bench/throughput/). Similarily we included benchmarks when using 500 prefill tokens and generating 250 tokens per request.

We include benchmarks on four devices: Galaxy A25 5G, AMD Ryzen 9HX 370, iMac M4 16GB, NVIDIA GeForce RTX 4090.

Please note that these benchmarks only benchmarks the Speech Language Model and does not include the Codec which is needed for a full audio generation pipeline.

|  | Air | Nano |
|---|---:|---:|
| **Model Name** | Air | Nano |
| **# Params (Active)** | ~360m | ~120m |
| **# Params (Emb + Active)** | ~552m | ~229m |
| **Cloning** | Yes | Yes |
| **Galaxy A25 5G (CPU only)** | 20 t/s | 45 t/s|
| **AMD Ryzen 9 HX 370 (CPU only)** | 119 t/s | 221 t/s |
| **iMAc M4 16 GB (CPU only)** | 111 t/s | 195 t/s |
| **RTX 4090** | 16194 t/s | 19268 t/s |
| **License** | Apache 2.0 | NeuTTS Open License 1.0 |

> [A]: CPU throughput measured on an AMD Ryzen 9 HX 370 using llama-bench, reporting prefill throughput at 1K context (tokens/s). Bench used 14 threads for prefill and 16 threads for decode (as configured in the benchmark run)

## Get Started with NeuTTS

> [!NOTE]
> We have added a [streaming example](examples/basic_streaming_example.py) using the `llama-cpp-python` library as well as a [finetuning script](examples/finetune.py). For finetuning, please refer to the [finetune guide](TRAINING.md) for more details.

1. **Clone Git Repo**

   ```bash
   git clone https://github.com/neuphonic/neutts.git
   cd neutts
   ```

2. **Install `espeak` (required dependency)**

   Please refer to the following link for instructions on how to install `espeak`:

   https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md

   ```bash
   # Mac OS
   brew install espeak

   # Ubuntu/Debian
   sudo apt install espeak

   # Windows install
   # via chocolatey (https://community.chocolatey.org/packages?page=1&prerelease=False&moderatorQueue=False&tags=espeak)
   choco install espeak-ng
   # via wingit
   winget install -e --id eSpeak-NG.eSpeak-NG
   # via msi (need to add to path or folow the "Windows users who installed via msi" below)
   # find the msi at https://github.com/espeak-ng/espeak-ng/releases
   ```

   Mac users may need to put the following lines at the top of the neutts.py file.

   ```python
   from phonemizer.backend.espeak.wrapper import EspeakWrapper
   _ESPEAK_LIBRARY = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib'  #use the Path to the library.
   EspeakWrapper.set_library(_ESPEAK_LIBRARY)
   ```

   Windows users who installed via msi / do not have their install on path need to run the following (see https://github.com/bootphon/phonemizer/issues/163)
   ```pwsh
   $env:PHONEMIZER_ESPEAK_LIBRARY = "c:\Program Files\eSpeak NG\libespeak-ng.dll"
   $env:PHONEMIZER_ESPEAK_PATH = "c:\Program Files\eSpeak NG"
   setx PHONEMIZER_ESPEAK_LIBRARY "c:\Program Files\eSpeak NG\libespeak-ng.dll"
   setx PHONEMIZER_ESPEAK_PATH "c:\Program Files\eSpeak NG"
   ```

3. **Install Python dependencies**

   The requirements file includes the dependencies needed to run the model with PyTorch.
   When using an ONNX decoder or a GGML model, some dependencies (such as PyTorch) are no longer required.

   The inference is compatible and tested on `python>=3.11`.

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install Llama-cpp-python to use the `GGUF` models.**

   ```bash
   pip install llama-cpp-python
   ```

   To run llama-cpp with GPU suport (CUDA, MPS) support please refer to:
   https://pypi.org/project/llama-cpp-python/

5. **(Optional) Install onnxruntime to use the `.onnx` decoder.**
   If you want to run the onnxdecoder
   ```bash
   pip install onnxruntime
   ```

## Running the Model

Run the basic example script to synthesize speech:

```bash
python -m examples.basic_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_audio samples/dave.wav \
  --ref_text samples/dave.txt
```

To specify a particular model repo for the backbone or codec, add the `--backbone` argument. Available backbones are listed in [NeuTTS-Air huggingface collection](https://huggingface.co/collections/neuphonic/neutts-air-68cc14b7033b4c56197ef350).

Several examples are available, including a Jupyter notebook in the `examples` folder.

### One-Code Block Usage

```python
from neutts import NeuTTS
import soundfile as sf

tts = NeuTTS(
   backbone_repo="neuphonic/neutts-air", # or 'neutts-air-q4-gguf' with llama-cpp-python installed
   backbone_device="cpu",
   codec_repo="neuphonic/neucodec",
   codec_device="cpu"
)
input_text = "My name is Dave, and um, I'm from London."

ref_text = "samples/dave.txt"
ref_audio_path = "samples/dave.wav"

ref_text = open(ref_text, "r").read().strip()
ref_codes = tts.encode_reference(ref_audio_path)

wav = tts.infer(input_text, ref_codes, ref_text)
sf.write("test.wav", wav, 24000)
```

### Streaming

Speech can also be synthesised in _streaming mode_, where audio is generated in chunks and plays as generated. Note that this requires pyaudio to be installed. To do this, run:

```bash
python -m examples.basic_streaming_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_codes samples/dave.pt \
  --ref_text samples/dave.txt
```

Again, a particular model repo can be specified with the `--backbone` argument - note that for streaming the model must be in GGUF format.

## Preparing References for Cloning

NeuTTS requires two inputs:

1. A reference audio sample (`.wav` file)
2. A text string

The model then synthesises the text as speech in the style of the reference audio. This is what enables NeuTTS models instant voice cloning capability.

### Example Reference Files

You can find some ready-to-use samples in the `examples` folder:

- `samples/dave.wav`
- `samples/jo.wav`

### Guidelines for Best Results

For optimal performance, reference audio samples should be:

1. **Mono channel**
2. **16-44 kHz sample rate**
3. **3–15 seconds in length**
4. **Saved as a `.wav` file**
5. **Clean** — minimal to no background noise
6. **Natural, continuous speech** — like a monologue or conversation, with few pauses, so the model can capture tone effectively

## Guidelines for minimizing Latency

For optimal performance on-device:

1. Use the GGUF model backbones
2. Pre-encode references
3. Use the [onnx codec decoder](https://huggingface.co/neuphonic/neucodec-onnx-decoder)

Take a look at this example [examples README](examples/README.md###minimal-latency-example) to get started.

## Responsibility

Every audio file generated by NeuTTS includes [Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth).

## Disclaimer

Don't use this model to do bad things… please.

## Developer Requirements

To run the pre commit hooks to contribute to this project run:

```bash
pip install pre-commit
```

Then:

```bash
pre-commit install
```

## Running Tests

First, install the dev requirements:

```
pip install -r requirements-dev.txt
```

To run the tests:

```
pytest tests/
```

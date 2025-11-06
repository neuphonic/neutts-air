# NeuTTS Air ☁️

HuggingFace 🤗: [Model](https://huggingface.co/neuphonic/neutts-air), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-air-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-air-q4-gguf) [Spaces](https://huggingface.co/spaces/neuphonic/neutts-air)

[Demo Video](https://github.com/user-attachments/assets/020547bc-9e3e-440f-b016-ae61ca645184)

_Created by [Neuphonic](http://neuphonic.com/) - building faster, smaller, on-device voice AI_

State-of-the-art Voice AI has been locked behind web APIs for too long. NeuTTS Air is the world’s first super-realistic, on-device, TTS speech language model with instant voice cloning. Built off a 0.5B LLM backbone, NeuTTS Air brings natural-sounding speech, real-time performance, built-in security and speaker cloning to your local device - unlocking a new category of embedded voice agents, assistants, toys, and compliance-safe apps.

## Key Features

- 🗣Best-in-class realism for its size - produces natural, ultra-realistic voices that sound human
- 📱Optimised for on-device deployment - provided in GGML format, ready to run on phones, laptops, or even Raspberry Pis
- 👫Instant voice cloning - create your own speaker with as little as 3 seconds of audio
- 🚄Simple LM + codec architecture built off a 0.5B backbone - the sweet spot between speed, size, and quality for real-world applications

> [!CAUTION]
> Websites like neutts.com are popping up and they're not affiliated with Neuphonic, our github or this repo.
>
> We are on neuphonic.com only. Please be careful out there! 🙏

## Model Details

NeuTTS Air is built off Qwen 0.5B - a lightweight yet capable language model optimised for text understanding and generation - as well as a powerful combination of technologies designed for efficiency and quality:

- **Supported Languages**: English
- **Audio Codec**: [NeuCodec](https://huggingface.co/neuphonic/neucodec) - our 50hz neural audio codec that achieves exceptional audio quality at low bitrates using a single codebook
- **Context Window**: 2048 tokens, enough for processing ~30 seconds of audio (including prompt duration)
- **Format**: Available in GGML format for efficient on-device inference
- **Responsibility**: Watermarked outputs
- **Inference Speed**: Real-time generation on mid-range devices
- **Power Consumption**: Optimised for mobile and embedded devices

## Get Started

> [!NOTE]
> We have added a [streaming example](examples/basic_streaming_example.py) using the `llama-cpp-python` library as well as a [finetuning script](examples/finetune.py). For finetuning, please refer to the [finetune guide](TRAINING.md) for more details.

1. **Clone Git Repo**

   ```bash
   git clone https://github.com/neuphonic/neutts-air.git
   cd neutts-air
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

5. **(Optional) Install ONNX Runtime to use the `.onnx` decoder.**

   Install the variant that matches your hardware:

   ```bash
   # CPU-only runtime
   pip install onnxruntime

   # CUDA-enabled runtime (NVIDIA GPUs)
   pip install onnxruntime-gpu

   # DirectML runtime (Windows GPUs)
   pip install onnxruntime-directml
   ```

   ONNX Runtime selects execution providers automatically. Override the provider order with the `--codec_device` argument when you need a specific GPU or CPU placement.

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

`backbone_device` now defaults to `"auto"`, which prefers CUDA (or Apple MPS) when available and falls back to CPU otherwise. Override it manually if you need to pin the backbone to a specific device (e.g. `"cpu"` or `"cuda:1"`).

`codec_device` follows similar rules for the ONNX decoder:

- omit the argument or set `"auto"` to choose the first available GPU execution provider and transparently fall back to CPU if none are detected;
- set `"cuda"`, `"cuda:<index>"`, `"directml"`, or `"rocm"` to prefer that GPU provider while still falling back to CPU when the provider is missing;
- set `"cpu"` to keep the decoder on the CPU exclusively.

### Device / Execution Provider Logic
---------------------------------

The same ONNX Runtime provider–selection logic used by the benchmarking tools is applied during normal inference and streaming via `neuttsair/neutts.py`.

- The system detects available ONNX providers (examples: `CUDAExecutionProvider`, `ROCMExecutionProvider`, `DmlExecutionProvider`, `MetalExecutionProvider`, `CoreMLExecutionProvider`).
- If a GPU/back-end provider is available it will be preferred; otherwise the runtime falls back to `CPUExecutionProvider`.
- On Apple Silicon (M1/M2/M3) the default PyPI `onnxruntime` wheel may not expose Metal/CoreML providers. If no native macOS provider is present the codec will run on CPU — install a native macOS ONNX package (see `examples/README.md`) to enable `MetalExecutionProvider`/`CoreMLExecutionProvider`.
- You can override automatic selection by passing `backbone_device` (for the backbone) or `codec_device` (for the decoder) to force `cpu` or a specific provider identifier (for example `--codec_device metal` or `--codec_device cuda:0`).



### One-Code Block Usage

```python
from neuttsair.neutts import NeuTTSAir
import soundfile as sf

tts = NeuTTSAir(
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

NeuTTS Air requires two inputs:

1. A reference audio sample (`.wav` file)
2. A text string

The model then synthesises the text as speech in the style of the reference audio. This is what enables NeuTTS Air's instant voice cloning capability.

### Example Reference Files

You can find some ready-to-use samples in the `examples` folder:

- `samples/dave.wav`
- `samples/jo.wav`

### Guidelines for Best Results

For optimal performance, reference audio samples should be:

1. **Mono channel**
2. **16-44 kHz sample rate**
3. **3-15 seconds in length**
4. **Saved as a `.wav` file**
5. **Clean** — minimal to no background noise
6. **Natural, continuous speech** — like a monologue or conversation, with few pauses, so the model can capture tone effectively

## Guidelines for minimizing Latency

For optimal performance on-device:

1. Use the GGUF model backbones.
2. Pre-encode references.
3. Use the [ONNX codec decoder](https://huggingface.co/neuphonic/neucodec-onnx-decoder).

Take a look at the [examples README](examples/README.md###minimal-latency-example) to get started.

## Responsibility

Every audio file generated by NeuTTS Air includes [Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth).

## Disclaimer

Don't use this model to do bad things… please.

## Benchmarking ONNX providers

You can profile the available ONNX Runtime execution providers with the benchmarking helper:

```bash
python -m examples.provider_benchmark \
   --input_text "Benchmarking NeuTTS Air" \
   --ref_codes samples/dave.pt \
   --ref_text samples/dave.txt \
   --runs 3 \
   --output benchmark_results.json
```

The script automatically discovers reachable providers (CUDA, ROCm, DirectML and CPU), synthesises speech for each, and reports:

- model load time, inference time, and total wall-clock time;
- real-time factor (RTF = inference time ÷ audio duration);
- host RAM deltas and CUDA VRAM peaks (when applicable);
- the concrete provider order applied by ONNX Runtime.

Raw run data and aggregated statistics are saved to JSON when `--output` is provided. Use `--verbose` to see per-run metrics in the console. Add `--summary_output benchmark.txt` to persist the rendered table and detected system hardware in plain text.

Models are now instantiated once per device during benchmarking and reused for the measured runs. A single warm-up pass is executed by default before timings are recorded; customise this with `--warmup_runs`, or opt back into the legacy behaviour (fresh load per run) with `--no_reuse`.

The console output includes your system metadata above the table so you can attribute results to a specific machine.

The column values are reported as `mean ± standard deviation` across the configured runs.

Note for macOS / Apple Silicon users: On M1/M2/M3 machines the ONNX decoder will automatically use the CPU if no Metal/CoreML execution providers are available; install a native macOS ONNX package (see `examples/README.md`) to enable `MetalExecutionProvider`/`CoreMLExecutionProvider` and request them with `--codec_device metal` or `--codec_device coreml`.

Pass `--backbone_devices cpu,cuda` (or any comma-separated list) to benchmark the cross-product of backbone and codec execution targets—the resulting table reports both placements per row. You can also provide multiple repositories at once. For example, to compare the torch checkpoint with both GGUF variants (Q4 and Q8) across CPU/CUDA combinations:

```bash
python -m examples.provider_benchmark \
   --input_text "Benchmarking NeuTTS Air combos" \
   --ref_codes samples/dave.pt \
   --ref_text samples/dave.txt \
   --backbone_repos neuphonic/neutts-air,neuphonic/neutts-air-q4-gguf,neuphonic/neutts-air-q8-gguf \
   --backbone_devices cpu,cuda \
   --codec_devices cpu,cuda \
   --runs 3 \
   --warmup_runs 1 \
   --output gguf_comparison.json \
   --summary_output gguf_comparison.txt
```

Add `--codec_repos` if you want to include custom decoder builds alongside `neucodec-onnx-decoder` in the same sweep.
## Developer Requirements

To run the pre commit hooks to contribute to this project run:

```bash
pip install pre-commit
```

Then:

```bash
pre-commit install
```

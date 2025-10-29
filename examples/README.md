# Examples

### GGUF Backbones

To run the model with `llama-cpp-python` in GGUF format, select a GGUF backbone when initialising the example script. All examples default to automatic device selection: CUDA (or DirectML/MPS where available) is chosen when possible, otherwise the CPU fallback is used.

```bash
python -m examples.basic_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_audio ./samples/dave.wav \
  --ref_text ./samples/dave.txt \
  --backbone neuphonic/neutts-air-q4-gguf \
  --backbone_device auto \
  --codec_device auto
```

### Provider Benchmark

Benchmark the available ONNX Runtime execution providers and summarise their performance:

```bash
python -m examples.provider_benchmark \
  --input_text "Benchmarking NeuTTS Air" \
  --ref_codes samples/dave.pt \
  --ref_text samples/dave.txt \
  --runs 3 \
  --warmup_runs 1 \
  --output benchmark_results.json \
  --summary_output benchmark_summary.txt
```

Add `--verbose` for per-run stats. `--summary_output` captures the rendered table and hardware information.

Set `--backbone_devices cpu,cuda` to measure both placements of a single backbone. Combine that with `--backbone_repos` to sweep multiple checkpoints in one pass. For example, the command below compares the torch checkpoint with the Q4 and Q8 GGUF exports across CPU/CUDA combinations for both the backbone and codec:

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

Use `--codec_repos` if you need to include additional decoder builds alongside `neucodec-onnx-decoder`. Each device is instantiated once and a warm-up pass (configurable via `--warmup_runs`) is executed before measurements start. Add `--no_reuse` to revert to the legacy behaviour that reloads models for every run.

### Pre-encode a reference

Reference encoding can be done ahead of time to reduce latency whilst inferencing the model; to pre-encode a reference you only need to provide a reference audio, as in the following script:

```bash
python -m examples.encode_reference \
 --ref_audio  ./samples/dave.wav \
 --output_path encoded_reference.pt
 ```

### Minimal Latency Example

To take advantage of encoding references ahead of time, we have a compiled the codec decoder into an [onnx graph](https://huggingface.co/neuphonic/neucodec-onnx-decoder) that enables inferencing NeuTTS-Air without loading the encoder. 
This can be useful for running the model in resource-constrained environments where the encoder may add a large amount of extra latency/memory usage.

To test the decoder, install the appropriate ONNX Runtime build (e.g. `onnxruntime` for CPU or `onnxruntime-gpu` for CUDA) and run one of the following:

```bash
python -m examples.onnx_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_codes samples/dave.pt \
  --ref_text samples/dave.txt \
  --backbone neuphonic/neutts-air-q4-gguf
```

```bash
python -m examples.onnx_example_gpu \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_codes samples/dave.pt \
  --ref_text samples/dave.txt \
  --backbone neuphonic/neutts-air-q4-gguf \
  --codec_device cuda
```

Set `--codec_device` to `auto` (default) to pick the first available GPU provider with an automatic CPU fallback. Use `cuda`/`cuda:0` (or `directml`, `rocm`) to request a specific provider—each will still fall back to CPU if unavailable. Pass `cpu` to disable GPU execution entirely.

### Streaming Support 

To stream the model output in chunks, try out the `onnx_streaming.py` example. For streaming, only the GGUF backends are currently supported. Ensure you have `llama-cpp-python`, `onnxruntime` and `pyaudio` installed to run this example.

```bash
python -m examples.basic_streaming_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_codes samples/dave.pt \
  --ref_text samples/dave.txt \
  --backbone neuphonic/neutts-air-q4-gguf
```

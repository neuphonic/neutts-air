# Examples

### GGUF Backbones

To run the model with `llama-cpp-python` in GGUF format, select a GGUF backbone when intializing the example script.

All examples now default to automatic device selection: CUDA (or DirectML/MPS where supported) is used when available, with a safe fallback to CPU.

```bash
python -m examples.basic_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_audio ./samples/dave.wav \
  --ref_text ./samples/dave.txt \
  --backbone neuphonic/neutts-air-q4-gguf
```

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

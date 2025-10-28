from typing import Generator
from pathlib import Path
import librosa
import numpy as np
import torch
import re
import perth
from neucodec import NeuCodec, DistillNeuCodec
from phonemizer.backend import EspeakBackend
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
import logging
from types import SimpleNamespace
from pydantic import BaseModel, ConfigDict
import io
from scipy.io.wavfile import write
from functools import lru_cache
import os


def _linear_overlap_add(frames: list[np.ndarray], stride: int) -> np.ndarray:
    # original impl --> https://github.com/facebookresearch/encodec/blob/main/encodec/utils.py
    assert len(frames)
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]

    total_size = 0
    for i, frame in enumerate(frames):
        frame_end = stride * i + frame.shape[-1]
        total_size = max(total_size, frame_end)

    sum_weight = np.zeros(total_size, dtype=dtype)
    out = np.zeros(*shape, total_size, dtype=dtype)

    offset: int = 0
    for frame in frames:
        frame_length = frame.shape[-1]
        t = np.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
        weight = np.abs(0.5 - (t - 0.5))

        out[..., offset : offset + frame_length] += weight * frame
        sum_weight[offset : offset + frame_length] += weight
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


class NeuTTSAir:

    def __init__(
        self,
        backbone_repo="neuphonic/neutts-air",
        backbone_device="cpu",
        encoder_repo="neuphonic/neucodec",
        decoder_repo="neuphonic/neucodec",
        codec_device="cpu",
    ):

        # Consts
        self.sample_rate = 24_000
        self.max_context = 2048
        self.hop_length = 480
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 25
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        # ggml & onnx flags
        self._is_quantized_model = False
        self._is_onnx_decoder = False

        # HF tokenizer
        self.tokenizer = None

        # Load phonemizer + models
        print("Loading phonemizer...")
        self.phonemizer = EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )

        self._load_backbone(backbone_repo, backbone_device)

        self._load_codec(encoder_repo, codec_device)

        if "onnx" in decoder_repo:
            self._load_codec(decoder_repo, codec_device)
        else:
            self.decoder = self.encoder

        # Load watermarker
        self.watermarker = perth.PerthImplicitWatermarker()

    def _load_backbone(self, backbone_repo, backbone_device):
        print(f"Loading backbone from: {backbone_repo} on {backbone_device} ...")

        # GGUF loading
        if backbone_repo.endswith("gguf"):

            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "Failed to import `llama_cpp`. "
                    "Please install it with:\n"
                    "    pip install llama-cpp-python"
                ) from e

            self.backbone = Llama(
                model_path=backbone_repo,
                # filename="*.gguf",
                verbose=False,
                n_gpu_layers=-1 if backbone_device == "gpu" else 0,
                n_ctx=self.max_context,
                mlock=True,
                flash_attn=True if backbone_device == "gpu" else False,
            )
            self._is_quantized_model = True

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo)
            self.backbone = AutoModelForCausalLM.from_pretrained(backbone_repo).to(
                torch.device(backbone_device)
            )

    def _load_codec(self, codec_repo, codec_device):

        print(f"Loading codec from: {codec_repo} on {codec_device} ...")
        match codec_repo:
            case "neuphonic/neucodec":
                self.encoder = NeuCodec.from_pretrained(codec_repo)
                self.encoder.eval().to(codec_device)
        #     case "neuphonic/distill-neucodec":
        #         self.codec = DistillNeuCodec.from_pretrained(codec_repo)
        #         self.codec.eval().to(codec_device)
            case "neuphonic/neucodec-onnx-decoder":

                if codec_device != "cpu":
                    raise ValueError("Onnx decoder only currently runs on CPU.")

                try:
                    from neucodec import NeuCodecOnnxDecoder
                except ImportError as e:
                    raise ImportError(
                        "Failed to import the onnx decoder."
                        " Ensure you have onnxruntime installed as well as neucodec >= 0.0.4."
                    ) from e

                self.decoder = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
                self._is_onnx_decoder = True

    def infer(self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str) -> np.ndarray:
        """
        Perform inference to generate speech from text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio. Defaults to None.
        Returns:
            np.ndarray: Generated speech waveform.
        """

        # Generate tokens
        if self._is_quantized_model:
            output_str = self._infer_ggml(ref_codes, ref_text, text)
        else:
            prompt_ids = self._apply_chat_template(ref_codes, ref_text, text)
            output_str = self._infer_torch(prompt_ids)

        # Decode
        wav = self._decode(output_str)
        watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=24_000)

        return watermarked_wav
    
    def infer_stream(self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str) -> Generator[np.ndarray, None, None]:
        """
        Perform streaming inference to generate speech from text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio. Defaults to None.
        Yields:
            np.ndarray: Generated speech waveform.
        """ 

        if self._is_quantized_model:
            return self._infer_stream_ggml(ref_codes, ref_text, text)

        else:
            raise NotImplementedError("Streaming is not implemented for the torch backend!")

    def encode_reference(self, ref_audio_path: str | Path):
        ref_audio_path = Path(ref_audio_path)
        mtime = os.path.getmtime(ref_audio_path) # modification time for cache

        self._encode_reference_cached(str(ref_audio_path), mtime) # avoid encoding the reference as much 
    
    @lru_cache(maxsize=10)
    def _encode_reference_cached(self, ref_audio_path: str, mtime: float):
        ref_audio_path = Path(ref_audio_path)
        ref_audio_encoded_path = ref_audio_path.with_suffix(".pt")

        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        with torch.no_grad():
            ref_codes = self.encoder.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        torch.save(ref_codes, ref_audio_encoded_path)

    def _decode(self, codes: str):

        # Extract speech token IDs using regex
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes)]

        if len(speech_ids) > 0:

            # Onnx decode
            if self._is_onnx_decoder:
                codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
                recon = self.decoder.decode_code(codes)

            # Torch decode
            else:
                with torch.no_grad():
                    codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(
                        self.decoder.device
                    )
                    recon = self.decoder.decode_code(codes).cpu().numpy()

            return recon[0, 0, :]
        else:
            raise ValueError("No valid speech tokens found in the output.")

    def _to_phones(self, text: str) -> str:
        phones = self.phonemizer.phonemize([text])
        phones = phones[0].split()
        phones = " ".join(phones)
        return phones

    def _apply_chat_template(
        self, ref_codes: list[int], ref_text: str, input_text: str
    ) -> list[int]:

        input_text = self._to_phones(ref_text) + " " + self._to_phones(input_text)
        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        chat = """user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"""
        ids = self.tokenizer.encode(chat)

        text_replace_idx = ids.index(text_replace)
        ids = (
            ids[:text_replace_idx]
            + [text_prompt_start]
            + input_ids
            + [text_prompt_end]
            + ids[text_replace_idx + 1 :]  # noqa
        )

        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)

        return ids

    def _infer_torch(self, prompt_ids: list[int]) -> str:
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                prompt_tensor,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                use_cache=True,
                min_new_tokens=50,
            )
        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(
            output_tokens[0, input_length:].cpu().numpy().tolist(), add_special_tokens=False
        )
        return output_str
    
    def _infer_ggml(self, ref_codes: list[int], ref_text: str, input_text: str) -> str:
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)

        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )
        output = self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
        )
        output_str = output["choices"][0]["text"]
        return output_str

    def _infer_stream_ggml(self, ref_codes: torch.Tensor, ref_text: str, input_text: str) -> Generator[np.ndarray, None, None]:
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)

        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        audio_cache: list[np.ndarray] = []
        token_cache: list[str] = [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples: int = 0
        n_decoded_tokens: int = len(ref_codes)

        for item in self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
            stream=True
        ):
            output_str = item["choices"][0]["text"]
            token_cache.append(output_str)

            if len(token_cache[n_decoded_tokens:]) >= self.streaming_frames_per_chunk + self.streaming_lookforward:

                # decode chunk
                tokens_start = max(
                    n_decoded_tokens
                    - self.streaming_lookback
                    - self.streaming_overlap_frames,
                    0
                )
                tokens_end = (
                    n_decoded_tokens
                    + self.streaming_frames_per_chunk
                    + self.streaming_lookforward
                    + self.streaming_overlap_frames
                )
                sample_start = (
                    n_decoded_tokens - tokens_start
                ) * self.hop_length
                sample_end = (
                    sample_start
                    + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length
                )
                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = self.watermarker.apply_watermark(recon, sample_rate=24_000)
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)

                # postprocess
                processed_recon = _linear_overlap_add(
                    audio_cache, stride=self.streaming_stride_samples
                )
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[
                    n_decoded_samples:new_samples_end
                ]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk
                yield processed_recon

        # final decoding handled seperately as non-constant chunk size
        remaining_tokens = len(token_cache) - n_decoded_tokens
        if len(token_cache) > n_decoded_tokens:
            tokens_start = max(
                len(token_cache)
                - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens), 
                0
            )
            sample_start = (
                len(token_cache) 
                - tokens_start 
                - remaining_tokens 
                - self.streaming_overlap_frames
            ) * self.hop_length
            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = self.watermarker.apply_watermark(recon, sample_rate=24_000)
            recon = recon[sample_start:]
            audio_cache.append(recon)

            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]
            yield processed_recon


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle model loading on startup to save time on endpoint calls"""
    logging.info("Loading model...")

    app.state.llm = SimpleNamespace()

    app.state.llm.model = NeuTTSAir(
        backbone_repo='./models/llm',
        backbone_device="cpu",
        encoder_repo="neuphonic/neucodec",
        decoder_repo="neuphonic/neucodec-onnx-decoder",
        codec_device="cpu"
    )

    logging.info("Model loaded successfully!")
    yield

app = FastAPI(lifespan=lifespan)


class LLMRequest(BaseModel):
    text: str
    ref_audio_path: str | Path
    ref_text: str


@app.post("/generate")
def generate(request: LLMRequest):
  
    model = app.state.llm.model

    logging.info("Encoding reference audio...")
    model.encode_reference(request.ref_audio_path)
    ref_codes_path = Path(request.ref_audio_path).with_suffix('.pt')
    ref_codes = torch.load(ref_codes_path)

    logging.info("Generating audio...")
    wav = model.infer(request.text, ref_codes, request.ref_text)

    # Convert audio to WAV format
    wav_buffer = io.BytesIO()
    write(
        wav_buffer,
        24000,
        (wav * 32767).astype(np.int16),
    )
    wav_buffer.seek(0)

    return Response(
        content=wav_buffer.getvalue(),
        media_type="audio/wav"
    )
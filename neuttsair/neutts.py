from pathlib import Path
import librosa
import numpy as np
import torch
import re
import perth
import time
from neucodec import NeuCodec, DistillNeuCodec
from phonemizer.backend import EspeakBackend
from transformers import AutoTokenizer, AutoModelForCausalLM


class NeuTTSAir:

    def __init__(
        self,
        backbone_repo="neuphonic/neutts-air",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
    ):

        # Consts
        self.sample_rate = 24_000
        self.max_context = 2048

        # ggml & onnx flags
        self._grammar = None  # set with a ggml model
        self._is_quantized_model = False
        self._is_onnx_codec = False

        # HF tokenizer
        self.tokenizer = None

        # Load phonemizer + models
        print("Loading phonemizer...")
        phonemizer_start = time.perf_counter()
        self.phonemizer = EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        phonemizer_time = time.perf_counter() - phonemizer_start

        self.loading_metrics = {
            "phonemizer_seconds": phonemizer_time,
            "backbone_seconds": None,
            "codec_seconds": None,
            "watermarker_seconds": None,
        }

        self._load_backbone(backbone_repo, backbone_device)

        self._load_codec(codec_repo, codec_device)

        # Load watermarker
        watermarker_start = time.perf_counter()
        self.watermarker = perth.PerthImplicitWatermarker()
        self.loading_metrics["watermarker_seconds"] = time.perf_counter() - watermarker_start

        self.last_inference_metrics: dict[str, float | None] = {}

    def _load_backbone(self, backbone_repo, backbone_device):
        print(f"Loading backbone from: {backbone_repo} on {backbone_device} ...")
        start = time.perf_counter()

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

            self.backbone = Llama.from_pretrained(
                repo_id=backbone_repo,
                filename="*.gguf",
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

        self.loading_metrics["backbone_seconds"] = time.perf_counter() - start

    def _load_codec(self, codec_repo, codec_device):

        print(f"Loading codec from: {codec_repo} on {codec_device} ...")
        start = time.perf_counter()
        match codec_repo:
            case "neuphonic/neucodec":
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/distill-neucodec":
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
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

                self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
                self._is_onnx_codec = True

            case _:
                raise ValueError(
                    "Invalid codec repo! Must be one of:"
                    " 'neuphonic/neucodec', 'neuphonic/distill-neucodec',"
                    " 'neuphonic/neucodec-onnx-decoder'."
                )

        self.loading_metrics["codec_seconds"] = time.perf_counter() - start

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

        inference_start = time.perf_counter()

        if self._is_quantized_model:
            backbone_start = time.perf_counter()
            output_str = self._infer_ggml(ref_codes, ref_text, text)
            backbone_time = time.perf_counter() - backbone_start
        else:
            prompt_ids = self._apply_chat_template(ref_codes, ref_text, text)
            backbone_start = time.perf_counter()
            output_str = self._infer_torch(prompt_ids)
            backbone_time = time.perf_counter() - backbone_start

        decode_start = time.perf_counter()
        wav = self._decode(output_str)
        decode_time = time.perf_counter() - decode_start

        watermark_start = time.perf_counter()
        watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
        watermark_time = time.perf_counter() - watermark_start

        total_time = time.perf_counter() - inference_start
        audio_duration = len(watermarked_wav) / self.sample_rate if len(watermarked_wav) else 0.0
        rtf = total_time / audio_duration if audio_duration else None

        self.last_inference_metrics = {
            "backbone_seconds": backbone_time,
            "codec_decode_seconds": decode_time,
            "watermark_seconds": watermark_time,
            "total_seconds": total_time,
            "audio_seconds": audio_duration,
            "rtf": rtf,
        }

        return watermarked_wav

    def encode_reference(self, ref_audio_path: str | Path):
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes

    def _decode(self, codes: str):

        # Extract speech token IDs using regex
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes)]

        if len(speech_ids) > 0:

            # Onnx decode
            if self._is_onnx_codec:
                codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
                recon = self.codec.decode_code(codes)

            # Torch decode
            else:
                with torch.no_grad():
                    codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(
                        self.codec.device
                    )
                    recon = self.codec.decode_code(codes).cpu().numpy()

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

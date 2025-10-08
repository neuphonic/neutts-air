from pathlib import Path
import librosa
import numpy as np
import torch
import re
import perth
from neucodec import NeuCodec, DistillNeuCodec
from phonemizer.backend import EspeakBackend
from transformers import AutoTokenizer, AutoModelForCausalLM

from . import constants as const


class NeuTTSAir:

    def __init__(
        self,
        backbone_repo: str = const.DEFAULT_BACKBONE_REPO,
        backbone_device: str = const.CPU_DEVICE,
        codec_repo: str = const.DEFAULT_CODEC_REPO,
        codec_device: str = const.CPU_DEVICE,
        sample_rate: int = const.DEFAULT_SAMPLE_RATE,
        max_context: int = const.DEFAULT_MAX_CONTEXT,
    ):

        # Consts
        self.sample_rate = sample_rate
        self.max_context = max_context

        # ggml & onnx flags
        self._grammar = None  # set with a ggml model
        self._is_quantized_model = False
        self._is_onnx_codec = False

        # HF tokenizer
        self.tokenizer = None

        # Load phonemizer + models
        print("Loading phonemizer...")
        self.phonemizer = EspeakBackend(
            language=const.PHONEMIZER_LANGUAGE,
            preserve_punctuation=const.PHONEMIZER_PRESERVE_PUNCTUATION,
            with_stress=const.PHONEMIZER_WITH_STRESS,
        )

        self._load_backbone(backbone_repo, backbone_device)

        self._load_codec(codec_repo, codec_device)

        # Load watermarker
        self.watermarker = perth.PerthImplicitWatermarker()

    def _load_backbone(self, backbone_repo, backbone_device):
        backbone_msg = "Loading backbone from: %s on %s ..." % (
            backbone_repo,
            backbone_device,
        )
        print(backbone_msg)

        # GGUF loading
        if backbone_repo.endswith(const.GGUF_SUFFIX):

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
                filename=const.GGUF_FILENAME_PATTERN,
                verbose=False,
                n_gpu_layers=(
                    const.GGUF_GPU_ALL_LAYERS
                    if backbone_device == const.GPU_DEVICE
                    else const.GGUF_GPU_NO_LAYERS
                ),
                n_ctx=self.max_context,
                mlock=True,
                flash_attn=backbone_device == const.GPU_DEVICE,
            )
            self._is_quantized_model = True

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo)
            backbone_model = AutoModelForCausalLM.from_pretrained(backbone_repo)
            device = torch.device(backbone_device)
            self.backbone = backbone_model.to(device)

    def _load_codec(self, codec_repo, codec_device):

        codec_msg = "Loading codec from: %s on %s ..." % (
            codec_repo,
            codec_device,
        )
        print(codec_msg)
        match codec_repo:
            case const.DEFAULT_CODEC_REPO:
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case const.DISTILL_CODEC_REPO:
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case const.ONNX_CODEC_REPO:

                if codec_device != const.CPU_DEVICE:
                    raise ValueError("Onnx decoder only currently runs on CPU.")

                try:
                    from neucodec import NeuCodecOnnxDecoder
                except ImportError as e:
                    raise ImportError(
                        "Failed to import the onnx decoder. Ensure you have "
                        "onnxruntime installed as well as neucodec >= 0.0.4."
                    ) from e

                self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
                self._is_onnx_codec = True

            case _:
                allowed_repos = (
                    const.DEFAULT_CODEC_REPO,
                    const.DISTILL_CODEC_REPO,
                    const.ONNX_CODEC_REPO,
                )
                quoted_repos = (f"'{repo}'" for repo in allowed_repos)
                allowed_list = ", ".join(quoted_repos)
                message = f"Invalid codec repo! Must be one of: {allowed_list}."
                raise ValueError(message)

    def infer(
        self,
        text: str,
        ref_codes: np.ndarray | torch.Tensor | list[int],
        ref_text: str,
    ) -> np.ndarray:
        """
        Generate speech from text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.Tensor): Encoded reference.
            ref_text (str): Reference transcript for the reference audio.

        Returns:
            np.ndarray: Generated speech waveform.
        """

        # Normalize reference codes to plain Python ints
        ref_codes_list = self._flatten_codes(ref_codes)

        # Generate tokens
        if self._is_quantized_model:
            output_str = self._infer_ggml(ref_codes_list, ref_text, text)
        else:
            prompt_ids = self._apply_chat_template(
                ref_codes_list,
                ref_text,
                text,
            )
            output_str = self._infer_torch(prompt_ids)

        # Decode
        wav = self._decode(output_str)
        watermarked_wav = self.watermarker.apply_watermark(
            wav,
            sample_rate=self.sample_rate,
        )

        return watermarked_wav

    def encode_reference(self, ref_audio_path: str | Path):
        wav, _ = librosa.load(
            ref_audio_path,
            sr=const.REFERENCE_ENCODE_SAMPLE_RATE,
            mono=const.REFERENCE_MONO,
        )
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        with torch.no_grad():
            encoded_codes = self.codec.encode_code(audio_or_path=wav_tensor)
            ref_codes = encoded_codes.squeeze(0).squeeze(0)
        return self._flatten_codes(ref_codes)

    def _decode(self, codes: str):

        # Extract speech token IDs using regex
        matches = re.findall(const.SPEECH_TOKEN_REGEX, codes)
        speech_ids = [int(num) for num in matches]

        if len(speech_ids) > 0:

            # Onnx decode
            if self._is_onnx_codec:
                speech_array = np.array(speech_ids, dtype=np.int32)
                speech_array = speech_array[np.newaxis, np.newaxis, :]
                recon = self.codec.decode_code(speech_array)

            # Torch decode
            else:
                with torch.no_grad():
                    speech_tensor = torch.tensor(speech_ids, dtype=torch.long)
                    speech_tensor = speech_tensor[None, None, :].to(self.codec.device)
                    recon = self.codec.decode_code(speech_tensor).cpu().numpy()

            return recon[0, 0, :]
        else:
            raise ValueError("No valid speech tokens found in the output.")

    def _to_phones(self, text: str) -> str:
        phones = self.phonemizer.phonemize([text])
        phones = phones[0].split()
        phones = const.PHONEME_SEPARATOR.join(phones)
        return phones

    def _apply_chat_template(
        self, ref_codes: list[int], ref_text: str, input_text: str
    ) -> list[int]:

        input_text = (
            self._to_phones(ref_text) + const.PHONEME_SEPARATOR + self._to_phones(input_text)
        )
        speech_replace = self.tokenizer.convert_tokens_to_ids(const.TOKEN_SPEECH_REPLACE)
        speech_gen_start = self.tokenizer.convert_tokens_to_ids(const.TOKEN_SPEECH_GENERATION_START)
        text_replace = self.tokenizer.convert_tokens_to_ids(const.TOKEN_TEXT_REPLACE)
        text_prompt_start = self.tokenizer.convert_tokens_to_ids(const.TOKEN_TEXT_PROMPT_START)
        text_prompt_end = self.tokenizer.convert_tokens_to_ids(const.TOKEN_TEXT_PROMPT_END)

        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        chat_prompt = const.CHAT_TEMPLATE.format(
            text_token=const.TOKEN_TEXT_REPLACE,
            speech_token=const.TOKEN_SPEECH_REPLACE,
        )
        ids = self.tokenizer.encode(chat_prompt)

        text_replace_idx = ids.index(text_replace)
        ids = (
            ids[:text_replace_idx]
            + [text_prompt_start]
            + input_ids
            + [text_prompt_end]
            + ids[text_replace_idx + 1 :]  # noqa
        )

        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([const.SPEECH_TOKEN_FORMAT.format(idx=i) for i in ref_codes])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)

        return ids

    def _infer_torch(self, prompt_ids: list[int]) -> str:
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids(const.TOKEN_SPEECH_GENERATION_END)
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                prompt_tensor,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=const.TORCH_GENERATE_DO_SAMPLE,
                temperature=const.DEFAULT_TEMPERATURE,
                top_k=const.DEFAULT_TOP_K,
                use_cache=const.TORCH_GENERATE_USE_CACHE,
                min_new_tokens=const.DEFAULT_MIN_NEW_TOKENS,
            )
        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(
            output_tokens[0, input_length:].cpu().numpy().tolist(),
            add_special_tokens=False,
        )
        return output_str

    def _infer_ggml(self, ref_codes: list[int], ref_text: str, input_text: str) -> str:
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)

        prompt_body = const.PHONEME_SEPARATOR.join([ref_text, input_text])
        codes_str = "".join(const.SPEECH_TOKEN_FORMAT.format(idx=idx) for idx in ref_codes)
        prompt = const.GGML_PROMPT_TEMPLATE.format(
            text_start=const.TOKEN_TEXT_PROMPT_START,
            prompt_body=prompt_body,
            text_end=const.TOKEN_TEXT_PROMPT_END,
            speech_start=const.TOKEN_SPEECH_GENERATION_START,
            codes=codes_str,
        )
        output = self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=const.DEFAULT_TEMPERATURE,
            top_k=const.DEFAULT_TOP_K,
            stop=list(const.GGML_STOP_TOKENS),
        )
        choices = output[const.LLM_OUTPUT_CHOICES_KEY]
        output_str = choices[0][const.LLM_OUTPUT_TEXT_KEY]
        return output_str

    def _flatten_codes(
        self, codes: np.ndarray | torch.Tensor | list[int] | tuple[int, ...]
    ) -> list[int]:
        """
        Convert reference codes into a flat list of Python ints to avoid tensor
        wrappers when building prompts.
        """
        if isinstance(codes, torch.Tensor):
            codes = codes.detach().cpu().view(-1).tolist()
        elif isinstance(codes, np.ndarray):
            codes = codes.reshape(-1).tolist()
        elif isinstance(codes, (list, tuple)):
            codes = list(codes)
        else:
            codes = [codes]

        return [int(code) for code in codes]

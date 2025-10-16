# Model finetuning

NeuTTS-Air follows [Llasa](https://github.com/zhenye234/LLaSA_training) in its training and inference setup. In order to finetune a model, you can use the `transformers` library from Hugging Face. We have an [example script](/examples/finetune.py) for finetuning using the [Emilia-YODAS dataset](https://huggingface.co/datasets/neuphonic/emilia-yodas-english-neucodec) that is encoded with [NeuCodec](https://huggingface.co/neuphonic/neucodec).

> [!NOTE]
> We have an on-going discussion about finetuning [here](https://github.com/neuphonic/neutts-air/issues/7) where some users have reported success with finetuning using the example script.

# Finetuning on your own dataset

You can prepare your own dataset by following these steps:
1. Encode your audio files using the [NeuCodec](https://huggingface.co/neuphonic/neucodec) model into a format similar to the [Emilia-YODAS dataset](https://huggingface.co/datasets/neuphonic/emilia-yodas-english-neucodec).
2. Setup your configuration file similar to the [example config](finetuning_config.yaml).
3. Check and modify the phonemizer and the tokenizer in the script such that they suit your dataset/task.
4. Run the finetuning script with your dataset and configuration file.

# Training from scratch or using additional labels

The NeuTTS Air model is based on the Qwen2.5 0.5B model. You can change this in the config file to use this instead of the trained NeuTTS Air model. This means you would need to add the speech token tags to the vocabulary. You can also add additional labels to the dataset by modifying model vocabulary. Both of these steps can be done as such in the script:

```python
codec_special_tokens = [
    "<|TEXT_REPLACE|>",
    "<|TEXT_PROMPT_START|>",
    "<|TEXT_PROMPT_END|>",
    "<|SPEECH_REPLACE|>",
    "<|SPEECH_GENERATION_START|>",
    "<|SPEECH_GENERATION_END|>",
    "<|EN|>",
    "<|ZH|>",
]
codec_tokens = [f"<|speech_{idx}|>" for idx in range(config.codebook_size)]

new_tokens = codec_special_tokens + codec_tokens
n_added_tokens = tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model.vocab_size = len(tokenizer)
```

You can then modify the input to the model to include these additional labels. For example, if you have speaker IDs or emotion labels, you can concatenate them with the phoneme tokens before passing them to the model.

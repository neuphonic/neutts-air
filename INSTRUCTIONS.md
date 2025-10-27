# Setup

1. Pull docker container
2. Copy `save_audio.py` and `samples/` into base directory
3. If desired, add a 3-5 second `reference.wav` and corresponding transcript `reference.txt` to `samples/`
4. Ensure you are using python >= 3.8 with packages `requests`, `fire` installed

# Usage

1.  `cd` into base directory
2. Run container with `docker run --rm --name cloud-imperium-tts-demo-medium -p 8081:80 -v ./samples:/app/samples/ cloud-imperium-tts-demo-medium`
3. In a separate terminal, `cd` into base directory
4. Synthesise speech with the command `python -m save_audio "<text_to_generate>" <output_file_path>.wav <gguf> <ref_audio_path>.wav <ref_text_path>`
5. Play `<output_file_path>.wav`

## Usage Examples

* `python -m save_audio "This is a test sentence." example_1.wav False samples/jo.wav samples/jo.txt`
* `python -m save_audio "And here is another example sentence, this time one that is significantly longer." example_2.wav False samples/dave.wav samples/dave.txt`
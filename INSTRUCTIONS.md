# Setup

1. Pull docker container
2. Copy `save_audio.py` and `samples/` into base directory
3. If desired, add additional speaker reference audios `<speaker>.wav` and corresponding transcripts `<speaker>.txt` to `samples/`
4. Ensure you are using python >= 3.8 with packages `requests`, `fire` installed

## Guidelines for Best Voice Cloning Results

For optimal performance, reference audio samples should be:

1. Mono channel
2. 16-44 kHz sample rate
3. 3–15 seconds in length
4. Saved as a .wav file
5. Clean — minimal to no background noise
6. Natural, continuous speech — like a monologue or conversation, with few pauses, so the model can capture tone effectively

# Usage

1.  `cd` into base directory
2. Run container with `docker run --rm --name cloud-imperium-tts-demo-medium -p 8081:80 -v ./samples:/app/samples/ cloud-imperium-tts-demo-medium` and wait for application startup to complete
3. In a separate terminal, `cd` into base directory
4. Synthesise speech with the command `python -m save_audio "<text_to_generate>" <output_file_path>.wav <ref_audio_path>.wav <ref_text_path>`
5. Play `<output_file_path>.wav`

## Usage Examples

* `python -m save_audio "This is a test sentence." example_1.wav samples/jo.wav samples/jo.txt`
* `python -m save_audio "And here is another example sentence, this time one that is significantly longer." example_2.wav samples/dave.wav samples/dave.txt`
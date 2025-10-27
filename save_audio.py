import requests
from pathlib import Path
from fire import Fire

API_URL = "http://localhost:8081"


def save_generated_audio(
    text: str, output_path: str = "test.wav", 
    gguf: bool = False,
    ref_audio_path: str = "./samples/dave.wav", 
    ref_text: str = "./samples/dave.txt"
):
    with open(ref_text, "r") as f:
        ref_text = f.read().strip()

    # Prepare request data
    data = {"text": text, "ref_audio_path": ref_audio_path, "ref_text": ref_text, "gguf": gguf}

    # Make API request
    print(f"Generating audio for: '{text}'")
    response = requests.post(f"{API_URL}/generate", json=data)
    response.raise_for_status()

    # Save audio to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as f:
        f.write(response.content)

    print(f"Audio saved to: {output_file.absolute()}")
    return output_file


if __name__ == "__main__":
    Fire(save_generated_audio)
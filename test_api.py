import requests
import json

def test_api():
    base_url = "http://localhost:8080"
    try:
        response = requests.get(f"{base_url}/")
        print("✓ Server is running")
        print(f"Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("✗ Server is not running on http://localhost:8080")
        return
    try:
        response = requests.get(f"{base_url}/v1/models")
        print("\n✓ Models endpoint working")
        print(f"Available models: {response.json()}")
    except Exception as e:
        print(f"✗ Models endpoint failed: {e}")
    try:
        response = requests.get(f"{base_url}/v1/voices")
        print("\n✓ Voices list endpoint working")
        print(f"Available voices: {response.json()}")
    except Exception as e:
        print(f"✗ Voices list endpoint failed: {e}")
    try:
        with open("samples/dave.wav", "rb") as voice_file:
            with open("samples/dave.txt", "r") as transcript_file:
                transcript = transcript_file.read()
            files = {"voice": ("dave.wav", voice_file, "audio/wav")}
            data = {
                "name": "dave",
                "transcript": transcript
            }
            response = requests.post(f"{base_url}/v1/audio/voice", files=files, data=data)
            print("\n✓ Voice upload endpoint working")
            print(f"Upload response: {response.json()}")
            voice_id = response.json()["voice_id"]
    except FileNotFoundError:
        print("\n⚠ Sample files not found, skipping voice upload test")
        voice_id = None
    except Exception as e:
        print(f"\n✗ Voice upload failed: {e}")
        voice_id = None
    if voice_id:
        try:
            speech_data = {
                "model": "neutts-air",
                "input": "Hello, this is a test of the text to speech system.",
                "voice": "dave",
                "response_format": "wav"
            }
            response = requests.post(
                f"{base_url}/v1/audio/speech",
                json=speech_data
            )
            if response.status_code == 200:
                with open("test_output.wav", "wb") as f:
                    f.write(response.content)
                print("\n✓ Speech generation endpoint working")
                print("Generated audio saved as test_output.wav")
            else:
                print(f"\n✗ Speech generation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"\n✗ Speech generation failed: {e}")
    print("\nAPI testing complete!")

if __name__ == "__main__":
    test_api()
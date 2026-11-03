"""Quick diagnostic for audio/voice subsystem."""
import os, sys

print("=== Atlas Audio/Voice Diagnostic ===")
print(f"Python: {sys.executable}")
print(f"Platform: {sys.platform}")

# Check Vosk model
root = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(root, "models", "vosk-model-small-en-us-0.15")
print(f"Vosk model exists: {os.path.exists(model_path)}")

# Check required packages
try:
    import sounddevice as sd
    print(f"sounddevice: OK (version {sd.__version__})")
    devices = sd.query_devices()
    input_devices = [d for d in devices if d["max_input_channels"] > 0]
    print(f"Input devices found: {len(input_devices)}")
    for d in input_devices[:3]:
        print(f"  - {d['name']} (channels={d['max_input_channels']}, rate={d['default_samplerate']})")
except Exception as e:
    print(f"sounddevice: FAIL - {e}")

try:
    import vosk
    print(f"vosk: OK")
except Exception as e:
    print(f"vosk: FAIL - {e}")

try:
    import edge_tts
    print(f"edge_tts: OK")
except Exception as e:
    print(f"edge_tts: FAIL - {e}")

try:
    import pyttsx3
    print(f"pyttsx3: OK")
except Exception as e:
    print(f"pyttsx3: FAIL - {e}")

try:
    import win32com.client
    print(f"win32com: OK")
except Exception as e:
    print(f"win32com: FAIL - {e}")

# Check wake config
from atlas_ai.atlas_wake_config import load_atlas_wake_config
config = load_atlas_wake_config()
print(f"Wake phrase: {config.trigger_phrase}")
print(f"Sleep phrase: {config.sleep_phrase}")
print(f"Audio sample rate: {config.audio_sample_rate}")
print(f"Audio block size: {config.audio_block_size}")
print(f"Audio device: {config.audio_device_name or 'default'}")

# Check voice output
from atlas_ai.voice.output import SpeechOutput
speaker = SpeechOutput()
print(f"TTS enabled: {speaker.enabled}")
print(f"TTS backend: {speaker._backend}")

# Test speech
print("\nTesting speech...")
speaker.speak("Atlas audio diagnostic complete.")
print("Speech test sent.")

print("\n=== Diagnostic Complete ===")


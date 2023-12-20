# TTS is a library for advanced Text-to-Speech generation.

# https://github.com/coqui-ai/TTS

"""
Features:
Pretrained models in +1100 languages.
Tools for training new models and fine-tuning existing models in any language.
Utilities for dataset analysis and curation.
"""

# Python API
# Running a multi-speaker and multi-lingual model
import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS

# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language

wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")

# Text to speech to a file
tts.tts_to_file(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")

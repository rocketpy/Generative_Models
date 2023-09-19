# TTS is a library for advanced Text-to-Speech generation.

# https://github.com/coqui-ai/TTS


# Example
# Running a multi-speaker and multi-lingual model
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models and choose the first one
model_name = TTS().list_models()[0]
# Init TTS
tts = TTS(model_name).to(device)

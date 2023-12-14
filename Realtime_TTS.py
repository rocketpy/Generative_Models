# RealtimeTTS is a state-of-the-art text-to-speech (TTS) library designed for real-time applications.
# It stands out in its ability to convert text streams fast into high-quality auditory output with minimal latency.

# https://github.com/KoljaB/RealtimeTTS

# pip install RealtimeTTS

"""
Installation into virtual environment with GPU support:
python -m venv env_realtimetts
env_realtimetts\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install RealtimeTTS
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
"""

# The basic usage example:
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine() # replace with your TTS engine
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()

# Feed Text
stream.feed("Hello, this is a sentence.")

# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching


# https://github.com/SWivid/F5-TTS


# Installation
# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n f5-tts python=3.10
conda activate f5-tts

# Install pytorch with your CUDA version, e.g.
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

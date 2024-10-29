# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching


# https://github.com/SWivid/F5-TTS


# Installation
# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n f5-tts python=3.10
conda activate f5-tts

# Install pytorch with your CUDA version, e.g.
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118


# As a pip package (if just for inference)
pip install git+https://github.com/SWivid/F5-TTS.git


# Local editable (if also do training, finetuning)
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
pip install -e .

  
# Docker usage
# Build from Dockerfile
docker build -t f5tts:v1 .

# Or pull from GitHub Container Registry
docker pull ghcr.io/swivid/f5-tts:main

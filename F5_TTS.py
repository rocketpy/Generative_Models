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

# Gradio App
"""
Currently supported features:

Basic TTS with Chunk Inference
Multi-Style / Multi-Speaker Generation
Voice Chat powered by Qwen2.5-3B-Instruct
# Launch a Gradio app (web interface)
f5-tts_infer-gradio

# Specify the port/host
f5-tts_infer-gradio --port 7860 --host 0.0.0.0

# Launch a share link
f5-tts_infer-gradio --share
"""

# CLI Inference
# Run with flags
# Leave --ref_text "" will have ASR model transcribe (extra GPU memory usage)
f5-tts_infer-cli \
--model "F5-TTS" \
--ref_audio "ref_audio.wav" \
--ref_text "The content, subtitle or transcription of reference audio." \
--gen_text "Some text you want TTS model generate for you."

# Run with default setting. src/f5_tts/infer/examples/basic/basic.toml
f5-tts_infer-cli
# Or with your own .toml file
f5-tts_infer-cli -c custom.toml

# Multi voice. See src/f5_tts/infer/README.md
f5-tts_infer-cli -c src/f5_tts/infer/examples/multi/story.toml

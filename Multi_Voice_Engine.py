
# EmotiVoice - Multi-Voice and Prompt-Controlled TTS Engine.

# https://github.com/netease-youdao/EmotiVoice


# Quickstart
"""
EmotiVoice Docker image

The easiest way to try EmotiVoice is by running the docker image.
You need a machine with a NVidia GPU. If you have not done so, set up NVidia container toolkit by following the instructions for Linux or Windows WSL2.
Then EmotiVoice can be run with,

docker run -dp 127.0.0.1:8501:8501 syq163/emoti-voice:latest

Now open your browser and navigate to http://localhost:8501 to start using EmotiVoice's powerful TTS capabilities.
Full installation

conda create -n EmotiVoice python=3.8 -y
conda activate EmotiVoice
pip install torch torchaudio
pip install numpy numba scipy transformers==4.26.1 soundfile yacs g2p_en jieba pypinyin
"""

# MegaTTS3 - Official PyTorch Implementation


# https://github.com/bytedance/MegaTTS3

"""
Key features:
Lightweight and Efficient: The backbone of the TTS Diffusion Transformer has only 0.45B parameters.
Ultra High-Quality Voice Cloning: You can try our model at Huggingface Demo. 
The .wav and .npy files can be found at link1. 
Submit a sample (.wav format, < 24s, and please do not contain space in filename) on link2 to receive .npy voice latents you can use locally.
Bilingual Support: Supports both Chinese and English, and code-switching.
Controllable: Supports accent intensity control  and fine-grained pronunciation/duration adjustment (coming soon).


# Installation
# Clone the repository 

git clone https://github.com/bytedance/MegaTTS3

cd MegaTTS3

# Requirements (for Linux)

# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n megatts3-env python=3.10
conda activate megatts3-env
pip install -r requirements.txt

"""

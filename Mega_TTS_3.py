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

# Set the root directory
export PYTHONPATH="/path/to/MegaTTS3:$PYTHONPATH"

# [Optional] Set GPU
export CUDA_VISIBLE_DEVICES=0

# If you encounter bugs with pydantic in inference, you should check if the versions of pydantic and gradio are matched.
# [Note] if you encounter bugs related with httpx, please check that whether your environmental variable "no_proxy" has patterns like "::"


# Requirements (for Windows)

# [The Windows version is currently under testing]
# Comment below dependence in requirements.txt:
# # WeTextProcessing==1.0.4.1

# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n megatts3-env python=3.10
conda activate megatts3-env
pip install -r requirements.txt
conda install -y -c conda-forge pynini==2.1.5
pip install WeTextProcessing==1.0.3

# [Optional] If you want GPU inference, you may need to install specific version of PyTorch for your GPU from https://pytorch.org/.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

"""

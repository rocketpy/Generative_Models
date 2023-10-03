# Making large AI models cheaper, faster and more accessible 

# https://github.com/hpcaitech/ColossalAI
# www.colossalai.org

# Note: only Linux is supported for now!!!

# pip install colossalai
# CUDA_EXT=1 pip install colossalai
# pip install colossalai-nightly

# Download From Source
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# install colossalai
pip install .

# Requirements:
""""
PyTorch >= 1.11 (PyTorch 2.x in progress)
Python >= 3.7
CUDA >= 11.0
NVIDIA GPU Compute Capability >= 7.0 (V100/RTX20 and higher)
Linux OS
"""

# Open-Sora: Democratizing Efficient Video Production for All

# https://github.com/hpcaitech/Open-Sora


# Installation
"""
# create a virtual env
conda create -n opensora python=3.10

# install torch
# the command below is for CUDA 12.1, choose install commands from 
# https://pytorch.org/get-started/locally/ based on your own CUDA version
pip3 install torch torchvision

# install flash attention (optional)
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex (optional)
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

# install xformers
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

# install this project
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora
pip install -v .
"""


# Training
"""
To launch training, first download T5 weights into pretrained_models/t5_ckpts/t5-v1_1-xxl.
Then run the following commands to launch training on a single node.

# 1 GPU, 16x256x256
torchrun --nnodes=1 --nproc_per_node=1 scripts/train.py configs/opensora/train/16x256x256.py --data-path YOUR_CSV_PATH
# 8 GPUs, 64x512x512
torchrun --nnodes=1 --nproc_per_node=8 scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT
To launch training on multiple nodes, prepare a hostfile according to ColossalAI, and run the following commands.

colossalai run --nproc_per_node 8 --hostfile hostfile scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT
"""

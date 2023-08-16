# FaceChain - FaceChain is a deep-learning toolchain for generating your Digital-Twin.

# https://github.com/modelscope/facechain


# Installation
"""
Compatibility Verification

The following are the environment dependencies that have been verified:

    python: py3.8, py3.10
    pytorch: torch2.0.0, torch2.0.1
    tensorflow: 2.8.0, tensorflow-cpu
    CUDA: 11.7
    CUDNN: 8+
    OS: Ubuntu 20.04, CentOS 7.9
    GPU: Nvidia-A10 24G

Resource Usage:
    GPU: About 19G
    Disk: About 50GB

# Docker

If you are familiar with using docker, we recommend to use this way:

# Step1: Prepare the environment with GPU on local or cloud, we recommend to use Alibaba Cloud ECS, refer to: https://www.aliyun.com/product/ecs

# Step2: Download the docker image (for installing docker engine, refer to https://docs.docker.com/engine/install/）
docker pull registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0

# Step3: run the docker container
docker run -it --name facechain -p 7860:7860 --gpus all registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0 /bin/bash
(Note: you may need to install the nvidia-container-runtime, refer to https://github.com/NVIDIA/nvidia-container-runtime)

# Step4: Install the gradio in the docker container:
pip3 install gradio

# Step5 clone facechain from github
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/modelscope/facechain.git --depth 1
cd facechain
python3 app.py

# Step6
Run the app server: click "public URL" --> in the form of: https://xxx.gradio.live
"""

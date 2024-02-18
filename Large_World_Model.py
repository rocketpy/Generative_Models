# Large World Model (LWM) is a general-purpose large-context multimodal autoregressive model.
# It is trained on a large dataset of diverse long videos and books using RingAttention, and can perform language, image, and video understanding and generation.

# https://github.com/LargeWorldModel/LWM

# Setup
"""
Install the requirements with:

conda create -n lwm python=3.10
pip install -U "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
or set up TPU VM with:

sh tpu_requirements.sh
"""


# Gemma - Gemma is a family of lightweight, state-of-the art open models built from research and technology used to create Google Gemini models.

# https://github.com/google/gemma_pytorch


# Try it free on Colab
# https://ai.google.dev/gemma/docs/pytorch_gemma


"""
Download Gemma model checkpoint

Alternatively, you can find the model checkpoints on the Hugging Face Hub here.
To download the models, go the the model repository of the model of interest and click the Files and versions tab,
and download the model and tokenizer files. For programmatic downloading,
if you have huggingface_hub installed, you can also run:

huggingface-cli download google/gemma-7b-it-pytorch
Note that you can choose between the 2B, 7B, 7B int8 quantized variants.

VARIANT=<2b or 7b or 9b or 27b>
CKPT_PATH=<Insert ckpt path here>
Try it free on Colab
Follow the steps at https://ai.google.dev/gemma/docs/pytorch_gemma.

Try it out with PyTorch
Prerequisite: make sure you have setup docker permission properly as a non-root user.

sudo usermod -aG docker $USER
newgrp docker
Build the docker image.
DOCKER_URI=gemma:${USER}

docker build -f docker/Dockerfile ./ -t ${DOCKER_URI}
"""

# example

# Kaggle access
import kagglehub

kagglehub.login()

# Install dependencies
# pip install -q -U torch immutabledict sentencepiece

# Choose variant and machine type
VARIANT = '2b-it'
MACHINE_TYPE = 'cuda'

CONFIG = VARIANT[:2]
if CONFIG == '2b':
    CONFIG = '2b-v2'


import os
import kagglehub

# Load model weights
weights_dir = kagglehub.model_download(f'google/gemma-2-2b/pyTorch/gemma-2-{VARIANT}')

# Ensure that the tokenizer is present
tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'

# Ensure that the checkpoint is present
ckpt_path = os.path.join(weights_dir, f'model.ckpt')
assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'

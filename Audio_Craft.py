# AudioCraft is a PyTorch library for deep learning research on audio generation. 
# AudioCraft contains inference and training code for two state-of-the-art AI generative models producing high-quality audio: AudioGen and MusicGen.

# https://github.com/facebookresearch/audiocraft

# Installation
"""
AudioCraft requires Python 3.9, PyTorch 2.0.0.
To install AudioCraft, you can run the following:

# Best to make sure you have torch installed first, in particular before installing xformers.
# Don't run this if you already have PyTorch installed !!!
pip install 'torch>=2.0'

# Then proceed to one of the following
pip install -U audiocraft  # stable release
pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft  # bleeding edge
pip install -e .  # or if you cloned the repo locally (mandatory if you want to train).
"""

# AudioCraft training pipelines
# https://github.com/facebookresearch/audiocraft/blob/main/docs/TRAINING.md

# Models
"""
At the moment, AudioCraft contains the training code and inference code for:

MusicGen: A state-of-the-art controllable text-to-music model.
AudioGen: A state-of-the-art text-to-sound model.
EnCodec: A state-of-the-art high fidelity neural audio codec.
Multi Band Diffusion: An EnCodec compatible decoder using diffusion.
"""

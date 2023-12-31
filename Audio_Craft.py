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

# MusicGen: Simple and Controllable Music Generation
# https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md

# AudioCraft requires a GPU with at least 16 GB of memory for running inference with the medium-sized models (~1.5B parameters) !!!!!

# Usage
"""
We offer a number of way to interact with MusicGen:

A demo is also available on the facebook/MusicGen Hugging Face Space (huge thanks to all the HF team for their support).
You can run the extended demo on a Colab: colab notebook
You can use the gradio demo locally by running python -m demos.musicgen_app --share.
You can play with MusicGen by running the jupyter notebook at demos/musicgen_demo.ipynb locally (if you have a GPU).
Finally, checkout @camenduru Colab page which is regularly updated with contributions from @camenduru and the community.
"""

# quick example for using the API
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=8)  # generate 8 seconds.
wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
wav = model.generate(descriptions)  # generates 3 samples.

melody, sr = torchaudio.load('./assets/bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)


# Transformers Usage
# pip install git+https://github.com/huggingface/transformers.git

from transformers import AutoProcessor, MusicgenForConditionalGeneration


processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

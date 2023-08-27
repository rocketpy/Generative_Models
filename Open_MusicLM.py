# Open MusicLM - Pytorch implementation of MusicLM
# Implementation of MusicLM, a text to music model published by Google Research, with a few modifications.

# https://github.com/zhvng/open-musiclm

# Install
# conda env create -f environment.yaml
# conda activate open-musiclm

# Usage
# pip install musiclm-pytorch


# MuLaN first needs to be trained

import torch
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer

audio_transformer = AudioSpectrogramTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64,
    spec_n_fft = 128,
    spec_win_length = 24,
    spec_aug_stretch_factor = 0.8
)

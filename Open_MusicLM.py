# Open MusicLM - Pytorch implementation of MusicLM
# Implementation of MusicLM, a text to music model published by Google Research, with a few modifications.

# https://github.com/zhvng/open-musiclm
# https://github.com/lucidrains/musiclm-pytorch

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
text_transformer = TextTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64
)

mulan = MuLaN(
    audio_transformer = audio_transformer,
    text_transformer = text_transformer
)

# get a ton of <sound, text> pairs and train
wavs = torch.randn(2, 1024)
texts = torch.randint(0, 20000, (2, 256))
loss = mulan(wavs, texts)
loss.backward()
# after much training, you can embed sounds and text into a joint embedding space
# for conditioning the audio LM
embeds = mulan.get_audio_latents(wavs)  # during training
embeds = mulan.get_text_latents(texts)  # during inference


# To obtain the conditioning embeddings for the three transformers that are a part of AudioLM, you must use the MuLaNEmbedQuantizer
from musiclm_pytorch import MuLaNEmbedQuantizer

# setup the quantizer with the namespaced conditioning embeddings, unique per quantizer as well as namespace (per transformer)
quantizer = MuLaNEmbedQuantizer(
    mulan = mulan,                          # pass in trained mulan from above
    conditioning_dims = (1024, 1024, 1024), # say all three transformers have model dimensions of 1024
    namespaces = ('semantic', 'coarse', 'fine')
)
# now say you want the conditioning embeddings for semantic transformer
wavs = torch.randn(2, 1024)
conds = quantizer(wavs = wavs, namespace = 'semantic') # (2, 8, 1024) - 8 is number of quantizers

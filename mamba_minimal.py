# mamba-minimal - Simple, minimal implementation of Mamba in one file of PyTorch.

# https://github.com/johnma2006/mamba-minimal


from model import Mamba
from transformers import AutoTokenizer


model = Mamba.from_pretrained('state-spaces/mamba-370m')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
generate(model, tokenizer, 'Mamba is the')

# DeepFloyd IF - a novel state-of-the-art open-source text-to-image model with a high degree of photorealism and language understanding.

# https://github.com/deep-floyd/IF


# pip install deepfloyd_if==1.0.2rc0
# pip install xformers==0.0.16
# pip install git+https://github.com/openai/CLIP.git --no-deps

# Next we install diffusers and dependencies:
# pip install diffusers accelerate transformers safetensors


from huggingface_hub import login

login()


import torch
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil


# stage 1
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
stage_1.enable_model_cpu_offload()

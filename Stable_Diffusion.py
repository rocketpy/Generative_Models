# Stable Diffusion - High-Resolution Image Synthesis with Latent Diffusion Models

# https://github.com/CompVis/stable-diffusion

# Requirements
"""
conda env create -f environment.yaml
conda activate ldm

conda install pytorch torchvision -c pytorch
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
"""

# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
	use_auth_token=True
).to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]  
    
image.save("astronaut_rides_horse.png")


# The following describes an example where a rough sketch made in Pinta is converted into a detailed artwork.
# https://www.pinta-project.com/
# python scripts/img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img <path-to-img.jpg> --strength 0.8

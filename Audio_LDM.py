# Audio Generation with AudioLDM

# https://github.com/haoheliu/AudioLDM
# https://audioldm.github.io/

# Features:
"""
Generate speech, sound effects, music and beyond.
This repo currently support:
Text-to-Audio Generation: Generate audio given text input.
Audio-to-Audio Generation: Given an audio, generate another audio that contain the same type of sound.
Text-guided Audio-to-Audio Style Transfer: Transfer the sound of an audio into another one using the text description.
"""

# Commandline Usage
# https://github.com/haoheliu/AudioLDM#commandline-usage

# Web APP
"""
The web APP currently only support Text-to-Audio generation. For full functionality please refer to the Commandline Usage.

Prepare running environment
conda create -n audioldm python=3.8; conda activate audioldm
pip3 install audioldm
git clone https://github.com/haoheliu/AudioLDM; cd AudioLDM
Start the web application (powered by Gradio)
python3 app.py
A link will be printed out. Click the link to open the browser and play.
"""


from diffusers import AudioLDMPipeline
import torch

repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]

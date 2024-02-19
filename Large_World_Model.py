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


# Command-line usage
"""
In this section, we provide instructions on how to run each of the provided scripts.
For each script, you may need to fill in your own paths and values in the variables described in the beginning of each script.

To run each of the following scripts, use bash <script_name>.sh:

Language model training: bash scripts/run_train_text.sh
Vision-Language model training: bash scripts/run_train_vision_text.sh
Single Needle Evals (Language Model): bash scripts/run_eval_needle.sh
Multi Needle Evals (Language Model): bash scripts/run_eval_needle_multi.sh
Sampling images (Vision-Language Model): bash scripts/run_sample_image.sh
Sampling videos (Vision-LanguageModel): bash scripts/run_sample_video.sh
Image / Video understanding (Vision-Language Model): bash scripts/run_vision_chat.sh
"""

# Ferret - Refer and Ground Anything Anywhere at Any Granularity

# https://github.com/apple/ml-ferret

# Install
"""
Clone this repository and navigate to FERRET folder
git clone https://github.com/apple/ml-ferret
cd ml-ferret

Install Package
conda create -n ferret python=3.10 -y
conda activate ferret

pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install pycocotools
pip install protobuf==3.20.0

Install additional packages for training cases
pip install ninja
pip install flash-attn --no-build-isolation
"""

# Launch a controller
"""
python -m ferret.serve.controller --host 0.0.0.0 --port 10000

Launch a gradio web server.

python -m ferret.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload --add_region_feature

Launch a model worker

This is the worker that load the ckpt and do the inference on the GPU.
Each worker is responsible for a single model specified in --model-path.

CUDA_VISIBLE_DEVICES=0 python -m ferret.serve.model_worker --host 0.0.0.0 --controller
http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./checkpoints/FERRET-13B-v0 --add_region_feature
"""

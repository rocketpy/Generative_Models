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

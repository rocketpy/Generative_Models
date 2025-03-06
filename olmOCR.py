# olmOCR - A toolkit for training language models to work with PDF documents in the wild.

# https://github.com/allenai/olmocr


# Demo - https://olmocr.allenai.org/

# Installation
"""
Requirements:

Recent NVIDIA GPU (tested on RTX 4090, L40S, A100, H100) with at least 20 GB of GPU RAM
30GB of free disk space
You will need to install poppler-utils and additional fonts for rendering PDF images.

Install dependencies (Ubuntu/Debian)

sudo apt-get update
sudo apt-get install poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools
Set up a conda environment and install olmocr

conda create -n olmocr python=3.11
conda activate olmocr

git clone https://github.com/allenai/olmocr.git
cd olmocr
pip install -e .
"""

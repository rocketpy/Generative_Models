# SAMURAI - Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory

# https://github.com/yangchris11/samurai


# SAMURAI Installation
"""
SAM 2 needs to be installed first before use. The code requires python>=3.10, as well as torch>=2.3.1 and torchvision>=0.18.1. 
Please follow the instructions here to install both PyTorch and TorchVision dependencies. You can install the SAMURAI version of SAM 2 on a GPU machine using:

cd sam2
pip install -e .
pip install -e ".[notebooks]"
Please see INSTALL.md from the original SAM 2 repository for FAQs on potential issues and solutions.

Install other requirements:

pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru

# Checkpoint Download

cd checkpoints && \
./download_ckpts.sh && \
cd ..


# Data Preparation
Please prepare the data in the following format:

data/LaSOT
├── airplane/
│   ├── airplane-1/
│   │   ├── full_occlusion.txt
│   │   ├── groundtruth.txt
│   │   ├── img
│   │   ├── nlp.txt
│   │   └── out_of_view.txt
│   ├── airplane-2/
│   ├── airplane-3/
│   ├── ...
├── basketball
├── bear
├── bicycle
...
├── training_set.txt
└── testing_set.txt
"""

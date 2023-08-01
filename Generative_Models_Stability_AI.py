# Generative Models by Stability AI

# https://github.com/Stability-AI/generative-models

# Installation:
"""
# 1. Clone the repo
git clone git@github.com:Stability-AI/generative-models.git
cd generative-models

2. Setting up the virtualenv
This is assuming you have navigated to the generative-models root after cloning it.

NOTE: This is tested under python3.8 and python3.10. For other python versions, you might encounter version conflicts.

PyTorch 1.13

# install required packages from pypi
python3 -m venv .pt13
source .pt13/bin/activate
pip3 install -r requirements/pt13.txt
PyTorch 2.0

# install required packages from pypi
python3 -m venv .pt2
source .pt2/bin/activate
pip3 install -r requirements/pt2.txt

3. Install sgm
pip3 install .

4. Install sdata for training
pip3 install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata
"""

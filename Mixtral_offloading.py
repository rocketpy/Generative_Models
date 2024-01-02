# Mixtral offloading - Mistral AI continues its mission to deliver the best open models to the developer community.

# https://github.com/dvmazur/mixtral-offloading
# https://mistral.ai/news/mixtral-of-experts/

# the demo notebook: https://github.com/dvmazur/mixtral-offloading/blob/master/notebooks/demo.ipynb


import numpy
from IPython.display import clear_output

# fix triton in colab
!export LC_ALL="en_US.UTF-8"
!export LD_LIBRARY_PATH="/usr/lib64-nvidia"
!export LIBRARY_PATH="/usr/local/cuda/lib64/stubs"
!ldconfig /usr/lib64-nvidia

!git clone https://github.com/dvmazur/mixtral-offloading.git --quiet
!cd mixtral-offloading && pip install -q -r requirements.txt
clear_output()


import sys

sys.path.append("mixtral-offloading")
import torch
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig
from huggingface_hub import snapshot_download
from IPython.display import clear_output
from tqdm.auto import trange
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import logging as hf_logging

from src.build_model import OffloadConfig, QuantConfig, build_model

hf_logging.disable_progress_bar()

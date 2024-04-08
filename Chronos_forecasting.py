# Chronos: Learning the Language of Time Series

# https://github.com/amazon-science/chronos-forecasting


# Usage
# To perform inference with Chronos models, install this package by running:

# pip install git+https://github.com/amazon-science/chronos-forecasting.git
# pip install pandas matplotlib


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")


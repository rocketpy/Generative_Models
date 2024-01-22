# Generating new data with Generative Models

# Imports
import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU


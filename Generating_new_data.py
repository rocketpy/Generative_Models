# Generating new data with Generative Models

import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU


# Load data to train the generative model.
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train / 255.0

# Flatten data
x_train = x_train.reshape(x_train.shape[0], -1)

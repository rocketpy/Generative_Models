# Cog implementation of MusicGen

# Cog is an open-source tool that packages machine learning models in a standard, production-ready container.
# You can deploy your packaged model to your own infrastructure, or to Replicate, where users can interact with it via web interface or API.

# https://github.com/replicate/cog-musicgen
# https://github.com/replicate/cog


#Install
# For macOS using Homebrew:
# brew install cog

# Install the latest release of Cog directly from GitHub by running the following commands in a terminal:
# sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
# sudo chmod +x /usr/local/bin/cog


# Example of predictions a model
# Save this to predict.py:

import numpy as np
from typing import Any
from cog import BasePredictor, Input, Path
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels.h5')

     # Define the arguments and types the model takes as input
    def predict(self, image: Path = Input(description="Image to classify")) -> Any:
        """Run a single prediction on the model"""
        # Preprocess the image
        img = keras_image.load_img(image, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Run the prediction
        preds = self.model.predict(x)
        # Return the top 3 predictions
        return decode_predictions(preds, top=3)[0]

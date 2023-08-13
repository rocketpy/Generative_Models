# Transformers - provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio.

# https://github.com/huggingface/transformers

# Info:
"""
These models can be applied on:
Text - for tasks like text classification, information extraction, question answering, summarization, translation, text generation, in over 100 languages.
Images - for tasks like image classification, object detection, and segmentation.
Audio - for tasks like speech recognition and audio classification.
"""


from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
classifier = pipeline('sentiment-analysis')
classifier('We are very happy to introduce pipeline to the transformers repository.')
# [{'label': 'POSITIVE', 'score': 0.9996980428695679}]


# For example, extract detected objects in an image:
import requests
from PIL import Image
from transformers import pipeline

# Download an image with cute cats
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
image_data = requests.get(url, stream=True).raw
image = Image.open(image_data)

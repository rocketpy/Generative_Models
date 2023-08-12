# Transformers - provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio.

# https://github.com/huggingface/transformers

# Info:
These models can be applied on:
"""
Text - for tasks like text classification, information extraction, question answering, summarization, translation, text generation, in over 100 languages.
Images - for tasks like image classification, object detection, and segmentation.
Audio - for tasks like speech recognition and audio classification.
"""


from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
classifier = pipeline('sentiment-analysis')
classifier('We are very happy to introduce pipeline to the transformers repository.')
# [{'label': 'POSITIVE', 'score': 0.9996980428695679}]
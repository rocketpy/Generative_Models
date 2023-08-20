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

# Allocate a pipeline for object detection
object_detector = pipeline('object-detection')
object_detector(image)
[{'score': 0.9982201457023621,
  'label': 'remote',
  'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}},
 {'score': 0.9960021376609802,
  'label': 'remote',
  'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}},
 {'score': 0.9954745173454285,
  'label': 'couch',
  'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}},
 {'score': 0.9988006353378296,
  'label': 'cat',
  'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}},
 {'score': 0.9986783862113953,
  'label': 'cat',
  'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]


# to download and use any of the pretrained models on your given task, PyTorch version:
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)

# supervision

# https://github.com/roboflow/supervision
# https://supervision.roboflow.com/latest/

# pip install supervision

# Models
"""
Supervision was designed to be model agnostic. Just plug in any classification, detection, or segmentation model.
For your convenience, we have created connectors for the most popular libraries like Ultralytics, Transformers, or MMDetection.
"""

import cv2
import supervision as sv
from ultralytics import YOLO

image = cv2.imread(...)
model = YOLO('yolov8s.pt')
result = model(image)[0]
detections = sv.Detections.from_ultralytics(result)

# len(detections)
# 5


import cv2
import supervision as sv

image = cv2.imread(...)
detections = sv.Detections(...)

bounding_box_annotator = sv.BoundingBoxAnnotator()
annotated_frame = bounding_box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)


# Datasets

# Supervision provides a set of utils that allow you to load, split, merge,
# and save datasets in one of the supported formats.

import supervision as sv

dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=...,
    annotations_directory_path=...,
    data_yaml_path=...
)

# dataset.classes
# ['dog', 'person']

# len(dataset)
# 1000

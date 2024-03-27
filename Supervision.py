# supervision

# https://github.com/roboflow/supervision

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

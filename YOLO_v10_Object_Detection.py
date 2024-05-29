# YOLOv10: Real-Time End-to-End Object Detection

# https://github.com/THU-MIG/yolov10


# Installation
"""
conda virtual environment is recommended.

conda create -n yolov10 python=3.9
conda activate yolov10
pip install -r requirements.txt
pip install -e .


# Demo
https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt

python app.py

# Please visit http://127.0.0.1:7860


# Validation

yolov10n.pt yolov10s.pt yolov10m.pt yolov10b.pt yolov10l.pt yolov10x.pt

yolo val model=yolov10n/s/m/b/l/x.pt data=coco.yaml batch=256


# Training

yolo detect train data=coco.yaml model=yolov10n/s/m/b/l/x.yaml epochs=500 batch=256 imgsz=640 device=0,1,2,3,4,5,6,7


# Prediction

yolo predict model=yolov10n/s/m/b/l/x.pt
"""

import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ultralytics import YOLO

yolo = YOLO("yolov8n.pt",task="detect")

results = yolo(source="./ultralytics/assets/bus.jpg",save=True)


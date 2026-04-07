import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ultralytics import YOLO

model = YOLO('./runs/segment/train/weights/best.pt')

model.train(data="yolov8-bvn.yaml", epochs=500, workers=0, batch=24, imgsz=960, patience=50)
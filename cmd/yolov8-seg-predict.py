import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ultralytics import YOLO

model = YOLO("./runs/segment/train11/weights/best.pt")
results = model.predict(source="./datasets/bvn/images/val/206-0.jpeg", save=True)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

model = YOLO(str(ROOT / "runs/segment/loss_wiou3/weights/best.pt"))
results = model.predict(source=str(ROOT / "datasets/bvn/images/test/test-107-0.jpg"), save=True)

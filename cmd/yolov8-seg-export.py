import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import shutil
import yaml
from datetime import datetime
from ultralytics import YOLO

MODEL_PATH = "./runs/segment/loss_wiou3/weights/best.pt"
EXPORT_BASE = "./export"
DATASET_YAML = "./yolov8-bvn.yaml"

YAML_TEMPLATE = {
    "type": "yolov8_seg",
    "name": "",
    "provider": "Custom",
    "display_name": "YOLOv8n-Seg (Waybill)",    
    "model_path": "best.onnx",
    "iou_threshold": 0.45,
    "conf_threshold": 0.25,
    "max_det": 300,
    "epsilon_factor": 0.005,
    "classes": [],
}


def get_next_export_dir(base):
    i = 1
    while os.path.exists(os.path.join(base, f"export{i}")):
        i += 1
    return os.path.join(base, f"export{i}")


def load_classes(dataset_yaml):
    with open(dataset_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    return [names[k] for k in sorted(names.keys())]


def extract_train_name(model_path):
    """从模型路径中提取训练名称，如 runs/segment/train4/weights/best.pt -> train4"""
    parts = os.path.normpath(model_path).split(os.sep)
    for part in parts:
        if part.startswith("train"):
            return part
    return "unknown"


def main():
    export_dir = get_next_export_dir(EXPORT_BASE)
    os.makedirs(export_dir, exist_ok=True)

    train_name = extract_train_name(MODEL_PATH)

    model = YOLO(MODEL_PATH)
    model.export(format="onnx", imgsz=960, half=False, batch=1)

    onnx_src = MODEL_PATH.replace(".pt", ".onnx")
    onnx_dst = os.path.join(export_dir, "best.onnx")
    shutil.move(onnx_src, onnx_dst)

    classes = load_classes(DATASET_YAML)
    date_str = datetime.now().strftime("%Y%m%d")
    config = YAML_TEMPLATE.copy()
    config["name"] = f"yolov8n-seg-r{date_str}-{train_name}"
    config["display_name"] = f"YOLOv8n-Seg (Waybill) - {train_name}"
    config["classes"] = classes

    yaml_path = os.path.join(export_dir, "yolov8n-seg-waybill.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Export done: {export_dir}")
    print(f"  Source: {MODEL_PATH} ({train_name})")
    print(f"  ONNX:   {onnx_dst}")
    print(f"  YAML:   {yaml_path}")
    print(f"  Display: {config['display_name']}")


if __name__ == "__main__":
    main()

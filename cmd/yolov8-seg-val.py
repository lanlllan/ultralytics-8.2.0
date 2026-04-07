"""
批量模型评估脚本
用法: python cmd/yolov8-seg-val.py
按提示输入训练编号，如: 5,6,8 或 5 6 8
然后选择使用 val 或 test 集进行验证
"""

import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ultralytics import YOLO

DATA_YAML = "yolov8-bvn.yaml"
IMGSZ = 960


def get_model_path(train_id):
    if train_id == 1:
        return "./runs/segment/train/weights/best.pt"
    return f"./runs/segment/train{train_id}/weights/best.pt"


def val_single(model_path, data_yaml, split, imgsz, train_id):
    model = YOLO(model_path)
    project = "runs/segment"
    name = f"{split}_train{train_id}"
    metrics = model.val(
        data=data_yaml, split=split, imgsz=imgsz, verbose=False,
        project=project, name=name,
        workers=0,  # 禁用多进程，避免页面文件不足错误
    )
    return {
        "box_p": metrics.box.mp,
        "box_r": metrics.box.mr,
        "box_map50": metrics.box.map50,
        "box_map": metrics.box.map,
        "mask_p": metrics.seg.mp,
        "mask_r": metrics.seg.mr,
        "mask_map50": metrics.seg.map50,
        "mask_map": metrics.seg.map,
        "save_dir": f"{project}/{name}",
    }


def main():
    raw = input("train IDs (e.g. 5,6,8): ").strip()
    # 支持中文逗号、英文逗号和空格分隔
    raw = raw.replace("，", " ").replace(",", " ")
    ids = [int(x) for x in raw.split() if x.strip().isdigit()]

    if not ids:
        print("no valid IDs")
        return

    split_input = input("use val or test split? (default: val): ").strip().lower()
    split = "test" if split_input == "test" else "val"
    print(f"  using split: {split}")

    results = {}
    for tid in ids:
        path = get_model_path(tid)
        if not os.path.exists(path):
            print(f"  train{tid}: {path} not found, skipped")
            continue
        print(f"\n  evaluating train{tid}...")
        results[tid] = val_single(path, DATA_YAML, split, IMGSZ, tid)

    if not results:
        print("no models evaluated")
        return

    header = f"{'':>10} | {'Box P':>7} {'Box R':>7} {'B mAP50':>8} {'B mAP95':>8} | {'Msk P':>7} {'Msk R':>7} {'M mAP50':>8} {'M mAP95':>8}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(f"  data={DATA_YAML}  split={split}  imgsz={IMGSZ}")
    print(sep)
    print(header)
    print(sep)

    for tid in sorted(results.keys()):
        r = results[tid]
        print(
            f"{'train'+str(tid):>10} |"
            f" {r['box_p']:>7.4f} {r['box_r']:>7.4f} {r['box_map50']:>8.4f} {r['box_map']:>8.4f} |"
            f" {r['mask_p']:>7.4f} {r['mask_r']:>7.4f} {r['mask_map50']:>8.4f} {r['mask_map']:>8.4f}"
        )

    print(sep)

    best_id = max(results, key=lambda t: results[t]["mask_map"] + results[t]["box_map"])
    print(f"\n  best model: train{best_id} (highest combined mAP50-95)")

    print(f"\n  results saved to:")
    for tid in sorted(results.keys()):
        print(f"    train{tid} -> {results[tid]['save_dir']}")


if __name__ == "__main__":
    main()

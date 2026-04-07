"""
COCO 实例分割标注 -> YOLO Segment 标注格式转换脚本
用法: python cmd/coco2yolo_seg.py --input E:\code\deeplearning\train\_annotations.coco.json --output E:\code\deeplearning\train\labels
"""

import json
import os
import argparse
from pathlib import Path


def coco_to_yolo_seg(coco_json_path, output_dir):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    img_map = {img["id"]: img for img in coco["images"]}

    # COCO category_id 可能不从 0 开始，需要重映射到连续的 YOLO class_id
    cat_ids = sorted(set(ann["category_id"] for ann in coco["annotations"]))
    cat_id_to_yolo = {cid: idx for idx, cid in enumerate(cat_ids)}

    cat_names = {}
    for cat in coco["categories"]:
        if cat["id"] in cat_id_to_yolo:
            cat_names[cat_id_to_yolo[cat["id"]]] = cat["name"]

    ann_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)

    count = 0
    for img_id, img_info in img_map.items():
        w = img_info["width"]
        h = img_info["height"]
        fname = Path(img_info["file_name"]).stem

        lines = []
        if img_id in ann_by_image:
            for ann in ann_by_image[img_id]:
                cls_id = cat_id_to_yolo[ann["category_id"]]
                for seg in ann["segmentation"]:
                    if len(seg) < 6:
                        continue
                    # 归一化坐标: x/w, y/h
                    norm_pts = []
                    for i in range(0, len(seg), 2):
                        nx = float(seg[i]) / w
                        ny = float(seg[i + 1]) / h
                        nx = max(0.0, min(1.0, nx))
                        ny = max(0.0, min(1.0, ny))
                        norm_pts.extend([f"{nx:.6f}", f"{ny:.6f}"])
                    lines.append(f"{cls_id} " + " ".join(norm_pts))

        txt_path = os.path.join(output_dir, f"{fname}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        count += 1

    print(f"Done: {count} label files -> {output_dir}")
    print(f"Images: {len(img_map)}, Annotations: {len(coco['annotations'])}")
    print(f"Classes: {cat_names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="COCO JSON path")
    parser.add_argument("--output", required=True, help="Output labels dir")
    args = parser.parse_args()
    coco_to_yolo_seg(args.input, args.output)

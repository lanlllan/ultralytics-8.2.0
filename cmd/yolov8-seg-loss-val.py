"""
损失函数改进对比评估脚本

自动扫描 runs/segment/loss_* 下的训练结果，统一评估并生成对比表格。

产出目录: runs/segment/loss_analysis_{split}/
  table1_overview.csv       — 各实验核心指标对比
  table2_iou_ablation.csv   — 消融分析: IoU 损失类型对比

用法:
  python cmd/yolov8-seg-loss-val.py              # 评估所有 loss_* 实验
  python cmd/yolov8-seg-loss-val.py --split test  # 使用 test 集
"""

import argparse
import csv
import os
import time

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ultralytics import YOLO

DATA_YAML = "yolov8-bvn.yaml"
IMGSZ = 960
RUNS_DIR = "runs/segment"

EXPERIMENT_META = {
    "baseline": {"iou_type": "CIoU",  "desc": "CIoU（默认）"},
    "eiou":     {"iou_type": "EIoU",  "desc": "EIoU（宽高分离）"},
    "siou":     {"iou_type": "SIoU",  "desc": "SIoU（角度感知）"},
    "wiou":     {"iou_type": "WIoU",  "desc": "WIoU（动态聚焦）"},
}


ATTN_BASELINE_FALLBACK = os.path.join(RUNS_DIR, "attnv2_baseline", "weights", "best.pt")


def find_experiments():
    found = {}
    if not os.path.isdir(RUNS_DIR):
        return found

    for name in sorted(os.listdir(RUNS_DIR)):
        if not name.startswith("loss_"):
            continue
        weights = os.path.join(RUNS_DIR, name, "weights", "best.pt")
        if os.path.exists(weights):
            exp_name = name.replace("loss_", "", 1)
            found[exp_name] = weights

    if "baseline" not in found and os.path.exists(ATTN_BASELINE_FALLBACK):
        found["baseline"] = ATTN_BASELINE_FALLBACK
        print(f"  [INFO] baseline 使用注意力实验基线: {ATTN_BASELINE_FALLBACK}")

    return found


def eval_model(weights_path, exp_name, split):
    model = YOLO(weights_path)

    metrics = model.val(
        data=DATA_YAML,
        split=split,
        imgsz=IMGSZ,
        verbose=False,
        project=RUNS_DIR,
        name=f"loss_eval_{exp_name}_{split}",
        workers=0,
    )

    n_params = sum(p.numel() for p in model.model.parameters())
    speed = metrics.speed

    return {
        "params": n_params,
        "box_p": metrics.box.mp,
        "box_r": metrics.box.mr,
        "box_map50": metrics.box.map50,
        "box_map": metrics.box.map,
        "mask_p": metrics.seg.mp,
        "mask_r": metrics.seg.mr,
        "mask_map50": metrics.seg.map50,
        "mask_map": metrics.seg.map,
        "preprocess_ms": speed.get("preprocess", 0),
        "inference_ms": speed.get("inference", 0),
        "postprocess_ms": speed.get("postprocess", 0),
    }


def print_comparison(results, split, baseline_key="baseline"):
    baseline = results.get(baseline_key)

    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  损失函数改进对比评估  |  data={DATA_YAML}  split={split}  imgsz={IMGSZ}")
    print(sep)

    header = (
        f"{'实验':>12} | {'IoU类型':>8} {'说明':>16} |"
        f" {'M_mAP50':>8} {'M_mAP95':>8} {'提升':>8} |"
        f" {'B_mAP50':>8} {'B_mAP95':>8} |"
        f" {'推理ms':>7}"
    )
    print(header)
    print("-" * 120)

    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["mask_map"], reverse=True)

    for exp_name in sorted_keys:
        r = results[exp_name]
        meta = EXPERIMENT_META.get(exp_name, {"iou_type": "?", "desc": "?"})

        if baseline and exp_name != baseline_key:
            delta_map = r["mask_map"] - baseline["mask_map"]
            delta_map_str = f"{delta_map:+.4f}"
        else:
            delta_map_str = "基线"

        print(
            f"{exp_name:>12} | {meta['iou_type']:>8} {meta['desc']:>16} |"
            f" {r['mask_map50']:>8.4f} {r['mask_map']:>8.4f} {delta_map_str:>8} |"
            f" {r['box_map50']:>8.4f} {r['box_map']:>8.4f} |"
            f" {r['inference_ms']:>7.1f}"
        )

    print(sep)

    best_name = sorted_keys[0]
    if best_name != baseline_key and baseline:
        improve = results[best_name]["mask_map"] - baseline["mask_map"]
        print(f"\n  最优方案: {best_name} ({EXPERIMENT_META.get(best_name, {}).get('iou_type', '?')})")
        print(f"    mask_mAP50-95 提升: {improve:+.4f} ({improve/baseline['mask_map']*100:+.1f}%)")
    elif baseline:
        print(f"\n  注意: 所有改进方案均未超过基线 mask_mAP50-95")

    print()


def _make_output_dir(split):
    out_dir = os.path.join(RUNS_DIR, f"loss_analysis_{split}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _R(v, n=5):
    return round(v, n)


def save_table1(results, split, baseline_key="baseline"):
    out_dir = _make_output_dir(split)
    baseline = results.get(baseline_key)
    path = os.path.join(out_dir, "table1_overview.csv")

    fieldnames = [
        "实验", "IoU类型", "说明",
        "mask_Precision", "mask_Recall", "mask_mAP50", "mask_mAP50-95", "mask_mAP50-95_提升",
        "box_Precision", "box_Recall", "box_mAP50", "box_mAP50-95",
        "推理耗时(ms)",
    ]

    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["mask_map"], reverse=True)
    rows = []
    for exp in sorted_keys:
        r = results[exp]
        meta = EXPERIMENT_META.get(exp, {"iou_type": "?", "desc": "?"})
        dm = (r["mask_map"] - baseline["mask_map"]) if baseline and exp != baseline_key else 0.0

        rows.append({
            "实验": exp,
            "IoU类型": meta["iou_type"],
            "说明": meta["desc"],
            "mask_Precision": _R(r["mask_p"]),
            "mask_Recall": _R(r["mask_r"]),
            "mask_mAP50": _R(r["mask_map50"]),
            "mask_mAP50-95": _R(r["mask_map"]),
            "mask_mAP50-95_提升": _R(dm),
            "box_Precision": _R(r["box_p"]),
            "box_Recall": _R(r["box_r"]),
            "box_mAP50": _R(r["box_map50"]),
            "box_mAP50-95": _R(r["box_map"]),
            "推理耗时(ms)": _R(r["inference_ms"], 2),
        })

    _write_csv(path, fieldnames, rows)
    print(f"  表1 已保存: {path}")
    return path


def save_table2(results, split, baseline_key="baseline"):
    out_dir = _make_output_dir(split)
    baseline = results.get(baseline_key)
    if not baseline:
        return None
    path = os.path.join(out_dir, "table2_iou_ablation.csv")

    fieldnames = [
        "IoU类型", "说明",
        "mask_Precision", "mask_Recall", "mask_mAP50", "mask_mAP50-95",
        "box_Precision", "box_Recall", "box_mAP50", "box_mAP50-95",
        "mask_mAP50-95_vs_baseline", "mask_mAP50-95_vs_baseline(%)",
    ]

    rows = []
    for exp in ["baseline", "eiou", "siou"]:
        if exp not in results:
            continue
        r = results[exp]
        meta = EXPERIMENT_META.get(exp, {"iou_type": "?", "desc": "?"})
        dm = (r["mask_map"] - baseline["mask_map"]) if exp != baseline_key else 0.0
        dm_pct = dm / baseline["mask_map"] * 100 if baseline["mask_map"] and exp != baseline_key else 0.0

        rows.append({
            "IoU类型": meta["iou_type"],
            "说明": meta["desc"],
            "mask_Precision": _R(r["mask_p"]),
            "mask_Recall": _R(r["mask_r"]),
            "mask_mAP50": _R(r["mask_map50"]),
            "mask_mAP50-95": _R(r["mask_map"]),
            "box_Precision": _R(r["box_p"]),
            "box_Recall": _R(r["box_r"]),
            "box_mAP50": _R(r["box_map50"]),
            "box_mAP50-95": _R(r["box_map"]),
            "mask_mAP50-95_vs_baseline": _R(dm),
            "mask_mAP50-95_vs_baseline(%)": _R(dm_pct, 2),
        })

    _write_csv(path, fieldnames, rows)
    print(f"  表2 已保存: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="损失函数改进对比评估")
    parser.add_argument("--split", type=str, default="val", help="验证集 (val/test)")
    parser.add_argument("--data", type=str, default=None, help="数据集 YAML")
    args = parser.parse_args()

    global DATA_YAML
    if args.data:
        DATA_YAML = args.data

    experiments = find_experiments()
    if not experiments:
        print(f"  未找到实验结果，请先运行: python cmd/yolov8-seg-loss-train.py")
        print(f"  查找路径: {RUNS_DIR}/loss_*/weights/best.pt")
        return

    print(f"\n  找到 {len(experiments)} 个实验:")
    for name, path in experiments.items():
        print(f"    {name:>12} -> {path}")

    results = {}
    for exp_name, weights_path in experiments.items():
        print(f"\n  评估 {exp_name}...")
        start = time.time()
        results[exp_name] = eval_model(weights_path, exp_name, args.split)
        elapsed = time.time() - start
        print(f"    完成 ({elapsed:.1f}s) mask_mAP50={results[exp_name]['mask_map50']:.4f}")

    print_comparison(results, args.split)

    save_table1(results, args.split)
    save_table2(results, args.split)

    out_dir = _make_output_dir(args.split)
    print(f"\n  全部完成，共评估 {len(results)} 个实验")
    print(f"  结果目录: {out_dir}")


if __name__ == "__main__":
    main()

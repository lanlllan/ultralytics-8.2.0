"""
损失函数改进对比评估脚本

自动扫描 runs/segment/loss_* 下的训练结果，统一评估并生成对比表格。

产出目录: runs/segment/loss_analysis_{split}/
  table1_overview.csv       — 各实验核心指标对比
  table2_iou_ablation.csv   — 消融分析: IoU 损失类型对比
  table3_mask_ablation.csv  — 消融分析: 掩码损失类型对比

用法:
  python cmd/yolov8-seg-loss-val.py              # 评估所有 loss_* 实验
  python cmd/yolov8-seg-loss-val.py --split test  # 使用 test 集
"""

import argparse
import csv
import os
import time

import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ultralytics import YOLO
from ultralytics.utils.metrics import Metric

DATA_YAML = "yolov8-bvn.yaml"
IMGSZ = 960
RUNS_DIR = "runs/segment"

EXPERIMENT_META = {
    "baseline": {"改进类型": "基线",     "改进项": "CIoU + BCE",   "desc": "默认损失"},
    "eiou":     {"改进类型": "IoU损失",  "改进项": "EIoU",         "desc": "宽高分离惩罚"},
    "siou":     {"改进类型": "IoU损失",  "改进项": "SIoU",         "desc": "角度+形状感知"},
    "wiou":     {"改进类型": "IoU损失",  "改进项": "WIoU",         "desc": "动态聚焦加权"},
    "dice":     {"改进类型": "掩码损失", "改进项": "Dice",         "desc": "区域重叠优化"},
    "bce-dice": {"改进类型": "掩码损失", "改进项": "BCE+Dice",     "desc": "像素+区域联合"},
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


def _compute_fitness(metric_obj, fitness_type):
    preset = Metric.FITNESS_PRESETS.get(fitness_type, Metric.FITNESS_PRESETS["default"])
    w = preset["w"]
    if preset["keys"] == "recall_map75_map":
        vals = np.array([metric_obj.mr, metric_obj.map75, metric_obj.map])
    else:
        vals = np.array(metric_obj.mean_results())
    return float((vals * w).sum())


def eval_model(weights_path, exp_name, split, fitness_type="default"):
    model = YOLO(weights_path)

    metrics = model.val(
        data=DATA_YAML,
        split=split,
        imgsz=IMGSZ,
        verbose=False,
        project=RUNS_DIR,
        name=f"loss_eval_{exp_name}_{split}",
        fitness_type=fitness_type,
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
        "mask_map75": metrics.seg.map75,
        "mask_map": metrics.seg.map,
        "mask_fitness": _compute_fitness(metrics.seg, fitness_type),
        "mask_fitness_default": _compute_fitness(metrics.seg, "default"),
        "preprocess_ms": speed.get("preprocess", 0),
        "inference_ms": speed.get("inference", 0),
        "postprocess_ms": speed.get("postprocess", 0),
    }


def print_comparison(results, split, fitness_type="default", baseline_key="baseline"):
    baseline = results.get(baseline_key)
    use_new_fit = fitness_type != "default"
    fit_key = "mask_fitness"
    fit_label = fitness_type

    sep = "=" * 145
    print(f"\n{sep}")
    print(f"  损失函数改进对比评估  |  data={DATA_YAML}  split={split}  imgsz={IMGSZ}  fitness={fitness_type}")
    print(sep)

    header = (
        f"{'实验':>12} | {'类型':>8} {'改进项':>10} {'说明':>14} |"
        f" {'Recall':>7} {'mAP75':>7} {'mAP50':>8} {'mAP95':>8} {'Fit':>7} {'提升':>8} |"
        f" {'B_mAP50':>8} {'B_mAP95':>8} |"
        f" {'推理ms':>7}"
    )
    print(header)
    print("-" * 145)

    sorted_keys = sorted(results.keys(), key=lambda k: results[k][fit_key], reverse=True)

    for exp_name in sorted_keys:
        r = results[exp_name]
        meta = EXPERIMENT_META.get(exp_name, {"改进类型": "?", "改进项": "?", "desc": "?"})

        if baseline and exp_name != baseline_key:
            delta = r[fit_key] - baseline[fit_key]
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "基线"

        print(
            f"{exp_name:>12} | {meta['改进类型']:>8} {meta['改进项']:>10} {meta['desc']:>14} |"
            f" {r['mask_r']:>7.4f} {r['mask_map75']:>7.4f} {r['mask_map50']:>8.4f} {r['mask_map']:>8.4f}"
            f" {r[fit_key]:>7.4f} {delta_str:>8} |"
            f" {r['box_map50']:>8.4f} {r['box_map']:>8.4f} |"
            f" {r['inference_ms']:>7.1f}"
        )

    print(sep)

    best_name = sorted_keys[0]
    if best_name != baseline_key and baseline:
        improve = results[best_name][fit_key] - baseline[fit_key]
        best_meta = EXPERIMENT_META.get(best_name, {})
        print(f"\n  最优方案: {best_name} ({best_meta.get('改进项', '?')})")
        print(f"    Fitness({fit_label}) 提升: {improve:+.4f}")
    elif baseline:
        print(f"\n  注意: 所有改进方案均未超过基线 Fitness({fit_label})")

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


def save_table1(results, split, fitness_type="default", baseline_key="baseline"):
    out_dir = _make_output_dir(split)
    baseline = results.get(baseline_key)
    path = os.path.join(out_dir, "table1_overview.csv")

    fieldnames = [
        "实验", "改进类型", "改进项", "说明",
        "mask_Precision", "mask_Recall", "mask_mAP50", "mask_mAP75", "mask_mAP50-95",
        f"Fitness({fitness_type})", "Fitness(default)", "Fitness_提升",
        "box_Precision", "box_Recall", "box_mAP50", "box_mAP50-95",
        "推理耗时(ms)",
    ]

    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["mask_fitness"], reverse=True)
    rows = []
    for exp in sorted_keys:
        r = results[exp]
        meta = EXPERIMENT_META.get(exp, {"改进类型": "?", "改进项": "?", "desc": "?"})
        df = (r["mask_fitness"] - baseline["mask_fitness"]) if baseline and exp != baseline_key else 0.0

        rows.append({
            "实验": exp,
            "改进类型": meta["改进类型"],
            "改进项": meta["改进项"],
            "说明": meta["desc"],
            "mask_Precision": _R(r["mask_p"]),
            "mask_Recall": _R(r["mask_r"]),
            "mask_mAP50": _R(r["mask_map50"]),
            "mask_mAP75": _R(r["mask_map75"]),
            "mask_mAP50-95": _R(r["mask_map"]),
            f"Fitness({fitness_type})": _R(r["mask_fitness"]),
            "Fitness(default)": _R(r["mask_fitness_default"]),
            "Fitness_提升": _R(df),
            "box_Precision": _R(r["box_p"]),
            "box_Recall": _R(r["box_r"]),
            "box_mAP50": _R(r["box_map50"]),
            "box_mAP50-95": _R(r["box_map"]),
            "推理耗时(ms)": _R(r["inference_ms"], 2),
        })

    _write_csv(path, fieldnames, rows)
    print(f"  表1 已保存: {path}")
    return path


def _build_ablation_table(results, exp_list, split, filename, baseline_key="baseline"):
    out_dir = _make_output_dir(split)
    baseline = results.get(baseline_key)
    if not baseline:
        return None
    path = os.path.join(out_dir, filename)

    fieldnames = [
        "改进项", "说明",
        "mask_Precision", "mask_Recall", "mask_mAP50", "mask_mAP50-95",
        "box_Precision", "box_Recall", "box_mAP50", "box_mAP50-95",
        "mask_mAP50-95_vs_baseline", "mask_mAP50-95_vs_baseline(%)",
    ]

    rows = []
    for exp in exp_list:
        if exp not in results:
            continue
        r = results[exp]
        meta = EXPERIMENT_META.get(exp, {"改进项": "?", "desc": "?"})
        dm = (r["mask_map"] - baseline["mask_map"]) if exp != baseline_key else 0.0
        dm_pct = dm / baseline["mask_map"] * 100 if baseline["mask_map"] and exp != baseline_key else 0.0

        rows.append({
            "改进项": meta["改进项"],
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
    print(f"  已保存: {path}")
    return path


def save_table2(results, split, baseline_key="baseline"):
    return _build_ablation_table(
        results, ["baseline", "eiou", "siou", "wiou"], split,
        "table2_iou_ablation.csv", baseline_key,
    )


def save_table3(results, split, baseline_key="baseline"):
    return _build_ablation_table(
        results, ["baseline", "dice", "bce-dice"], split,
        "table3_mask_ablation.csv", baseline_key,
    )


def main():
    parser = argparse.ArgumentParser(description="损失函数改进对比评估")
    parser.add_argument("--split", type=str, default="val", help="验证集 (val/test)")
    parser.add_argument("--data", type=str, default=None, help="数据集 YAML")
    parser.add_argument("--fitness_type", type=str, default="default",
                        choices=list(Metric.FITNESS_PRESETS.keys()),
                        help="Fitness 函数类型 (default: 原版, recall_map75: OCR流水线)")
    args = parser.parse_args()

    global DATA_YAML
    if args.data:
        DATA_YAML = args.data

    experiments = find_experiments()
    if not experiments:
        print(f"  未找到实验结果，请先运行: python cmd/yolov8-seg-loss-train.py")
        print(f"  查找路径: {RUNS_DIR}/loss_*/weights/best.pt")
        return

    fit_label = f"{args.fitness_type} ({Metric.FITNESS_PRESETS[args.fitness_type]['w']})"
    print(f"\n  Fitness 函数: {fit_label}")
    print(f"  找到 {len(experiments)} 个实验:")
    for name, path in experiments.items():
        print(f"    {name:>12} -> {path}")

    results = {}
    for exp_name, weights_path in experiments.items():
        print(f"\n  评估 {exp_name}...")
        start = time.time()
        results[exp_name] = eval_model(weights_path, exp_name, args.split, args.fitness_type)
        elapsed = time.time() - start
        r = results[exp_name]
        print(f"    完成 ({elapsed:.1f}s) Recall={r['mask_r']:.4f} mAP75={r['mask_map75']:.4f} mAP95={r['mask_map']:.4f}")

    print_comparison(results, args.split, args.fitness_type)

    save_table1(results, args.split, args.fitness_type)
    save_table2(results, args.split)
    save_table3(results, args.split)

    out_dir = _make_output_dir(args.split)
    print(f"\n  全部完成，共评估 {len(results)} 个实验")
    print(f"  结果目录: {out_dir}")


if __name__ == "__main__":
    main()

"""
注意力机制对比评估脚本

自动扫描 runs/segment/attnv2_* 下的训练结果，统一评估并生成对比表格。

产出目录: runs/segment/attnv2_analysis_{split}_{fitness_type}/
  table1_overview.csv       — 各实验核心指标对比
  table2_ablation_type.csv  — 消融分析: 注意力类型对比（位置A）
  table3_ablation_pos.csv   — 消融分析: 插入位置对比（A vs C）

用法:
  python cmd/yolov8-seg-attention-val.py              # 评估所有 attnv2_* 实验
  python cmd/yolov8-seg-attention-val.py --split test  # 使用 test 集
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
    "baseline":  {"attention": "无",        "position": "-",            "params_note": "基线 3.4M"},
    "cbam-a":    {"attention": "CBAM",      "position": "A: SPPF后",    "params_note": "+~65K"},
    "simam-a":   {"attention": "SimAM",     "position": "A: SPPF后",    "params_note": "+0"},
    "ca-a":      {"attention": "CoordAtt",  "position": "A: SPPF后",    "params_note": "+~6K"},
    "cbam-c":    {"attention": "CBAM",      "position": "C: Neck C2f后", "params_note": "+~75K"},
    "ca-c":      {"attention": "CoordAtt",  "position": "C: Neck C2f后", "params_note": "+~18K"},
}


def find_experiments():
    """扫描 runs/segment/ 下所有 attnv2_* 目录。"""
    found = {}
    if not os.path.isdir(RUNS_DIR):
        return found

    for name in sorted(os.listdir(RUNS_DIR)):
        if not name.startswith("attnv2_"):
            continue
        weights = os.path.join(RUNS_DIR, name, "weights", "best.pt")
        if os.path.exists(weights):
            exp_name = name.replace("attnv2_", "", 1)
            found[exp_name] = weights
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
    """评估单个模型，返回指标字典。"""
    model = YOLO(weights_path)
    model.model.info(verbose=False)

    fitness_tag = fitness_type.replace("-", "_")
    metrics = model.val(
        data=DATA_YAML,
        split=split,
        imgsz=IMGSZ,
        verbose=False,
        project=RUNS_DIR,
        name=f"attnv2_eval_{exp_name}_{split}_{fitness_tag}",
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
    """打印对比表格。"""
    baseline = results.get(baseline_key)
    fit_key = "mask_fitness"

    sep = "=" * 150
    print(f"\n{sep}")
    print(f"  注意力机制对比评估  |  data={DATA_YAML}  split={split}  imgsz={IMGSZ}  fitness={fitness_type}")
    print(sep)

    header = (
        f"{'实验':>12} | {'注意力':>8} {'位置':>14} | {'参数量':>10} {'增量':>8} |"
        f" {'Recall':>7} {'mAP75':>7} {'mAP50':>8} {'mAP95':>8} {'Fit':>7} {'提升':>7} |"
        f" {'B_mAP50':>8} {'B_mAP95':>8} |"
        f" {'推理ms':>7}"
    )
    print(header)
    print("-" * 150)

    sorted_keys = sorted(results.keys(), key=lambda k: results[k][fit_key], reverse=True)

    for exp_name in sorted_keys:
        r = results[exp_name]
        meta = EXPERIMENT_META.get(exp_name, {"attention": "?", "position": "?", "params_note": "?"})

        params_str = f"{r['params']/1e6:.2f}M"

        if baseline and exp_name != baseline_key:
            delta_params = r["params"] - baseline["params"]
            delta_str = f"+{delta_params/1e3:.1f}K" if delta_params > 0 else f"{delta_params/1e3:.1f}K"
            delta_fit = r[fit_key] - baseline[fit_key]
            delta_fit_str = f"{delta_fit:+.4f}"
        else:
            delta_str = "-"
            delta_fit_str = "基线"

        print(
            f"{exp_name:>12} | {meta['attention']:>8} {meta['position']:>14} |"
            f" {params_str:>10} {delta_str:>8} |"
            f" {r['mask_r']:>7.4f} {r['mask_map75']:>7.4f} {r['mask_map50']:>8.4f} {r['mask_map']:>8.4f}"
            f" {r[fit_key]:>7.4f} {delta_fit_str:>7} |"
            f" {r['box_map50']:>8.4f} {r['box_map']:>8.4f} |"
            f" {r['inference_ms']:>7.1f}"
        )

    print(sep)

    best_name = sorted_keys[0]
    if best_name != baseline_key and baseline:
        improve = results[best_name][fit_key] - baseline[fit_key]
        print(f"\n  最优方案: {best_name}")
        print(f"    Fitness({fitness_type}) 提升: {improve:+.4f}")
        print(f"    参数量: {results[best_name]['params']/1e6:.2f}M")
    elif baseline:
        print(f"\n  注意: 所有注意力方案均未超过基线 Fitness({fitness_type})")

    print()


def print_ablation_type(results, baseline_key="baseline"):
    """表2: 消融分析——注意力类型对比（位置A），固定插入位置为 A，对比不同注意力机制。"""
    baseline = results.get(baseline_key)
    if not baseline:
        return

    position_a = ["baseline", "cbam-a", "simam-a", "ca-a"]
    available = [k for k in position_a if k in results]
    if len(available) < 2:
        return

    sep = "=" * 100
    print(f"\n{sep}")
    print("  表2: 消融分析——注意力类型对比（位置 A: SPPF 后）")
    print(sep)

    header = (
        f"{'机制':>12} | {'注意力维度':>12} | {'额外参数':>8} |"
        f" {'M_P':>7} {'M_R':>7} {'M_mAP50':>8} {'M_mAP95':>8} |"
        f" {'B_P':>7} {'B_R':>7} {'B_mAP50':>8} {'B_mAP95':>8} |"
        f" {'相对基线':>8}"
    )
    print(header)
    print("-" * 100)

    dimension_map = {
        "baseline": ("—", "—"),
        "cbam-a": ("CBAM", "通道+空间"),
        "simam-a": ("SimAM", "通道+空间"),
        "ca-a": ("CoordAtt", "通道+位置"),
    }

    for exp_name in available:
        r = results[exp_name]
        label, dim = dimension_map.get(exp_name, (exp_name, "?"))

        if exp_name == baseline_key:
            delta_params_str = "—"
            delta_map_str = "基线"
        else:
            dp = r["params"] - baseline["params"]
            delta_params_str = f"+{dp/1e3:.1f}K" if dp > 0 else ("0" if dp == 0 else f"{dp/1e3:.1f}K")
            delta = r["mask_map"] - baseline["mask_map"]
            delta_map_str = f"{delta:+.4f}"

        print(
            f"{label:>12} | {dim:>12} | {delta_params_str:>8} |"
            f" {r['mask_p']:>7.4f} {r['mask_r']:>7.4f} {r['mask_map50']:>8.4f} {r['mask_map']:>8.4f} |"
            f" {r['box_p']:>7.4f} {r['box_r']:>7.4f} {r['box_map50']:>8.4f} {r['box_map']:>8.4f} |"
            f" {delta_map_str:>8}"
        )

    print(sep)
    print()


def print_ablation_position(results, baseline_key="baseline"):
    """表3: 消融分析——插入位置对比，对比同一注意力机制在位置 A vs 位置 C 的效果。"""
    baseline = results.get(baseline_key)
    if not baseline:
        return

    pairs = [
        ("CBAM", "cbam-a", "cbam-c"),
        ("CoordAtt", "ca-a", "ca-c"),
    ]

    has_any = any(a in results and c in results for _, a, c in pairs)
    if not has_any:
        return

    sep = "=" * 110
    print(f"\n{sep}")
    print("  表3: 消融分析——插入位置对比（位置 A: SPPF后 1层  vs  位置 C: Neck C2f后 4层）")
    print(sep)

    header = (
        f"{'机制':>10} | {'位置':>8} | {'额外参数':>8} |"
        f" {'M_mAP50':>8} {'M_mAP95':>8} {'vs基线':>8} |"
        f" {'B_mAP50':>8} {'B_mAP95':>8} |"
        f" {'推理ms':>7} | {'位置C vs A':>10}"
    )
    print(header)
    print("-" * 110)

    br = baseline
    print(
        f"{'baseline':>10} | {'—':>8} | {'—':>8} |"
        f" {br['mask_map50']:>8.4f} {br['mask_map']:>8.4f} {'基线':>8} |"
        f" {br['box_map50']:>8.4f} {br['box_map']:>8.4f} |"
        f" {br['inference_ms']:>7.1f} | {'':>10}"
    )
    print("-" * 110)

    for mech, key_a, key_c in pairs:
        ra = results.get(key_a)
        rc = results.get(key_c)
        if not ra and not rc:
            continue

        for label_pos, key, r in [("A (1层)", key_a, ra), ("C (4层)", key_c, rc)]:
            if not r:
                continue
            dp = r["params"] - baseline["params"]
            dp_str = f"+{dp/1e3:.1f}K" if dp > 0 else ("0" if dp == 0 else f"{dp/1e3:.1f}K")
            delta = r["mask_map"] - baseline["mask_map"]
            delta_str = f"{delta:+.4f}"

            pos_compare = ""
            if label_pos.startswith("C") and ra:
                c_vs_a = r["mask_map"] - ra["mask_map"]
                pos_compare = f"{c_vs_a:+.4f}"

            print(
                f"{mech:>10} | {label_pos:>8} | {dp_str:>8} |"
                f" {r['mask_map50']:>8.4f} {r['mask_map']:>8.4f} {delta_str:>8} |"
                f" {r['box_map50']:>8.4f} {r['box_map']:>8.4f} |"
                f" {r['inference_ms']:>7.1f} | {pos_compare:>10}"
            )

        print("-" * 110)

    print(sep)
    print()


def _make_output_dir(split, fitness_type="default"):
    fitness_tag = fitness_type.replace("-", "_")
    out_dir = os.path.join(RUNS_DIR, f"attnv2_analysis_{split}_{fitness_tag}")
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
    """表1: 各实验核心指标对比 → table1_overview.csv"""
    out_dir = _make_output_dir(split, fitness_type)
    baseline = results.get(baseline_key)
    path = os.path.join(out_dir, "table1_overview.csv")

    fieldnames = [
        "实验", "注意力机制", "插入位置", "参数量", "参数增量", "参数增量(%)",
        "mask_Precision", "mask_Recall", "mask_mAP50", "mask_mAP75", "mask_mAP50-95",
        f"Fitness({fitness_type})", "Fitness(default)", "Fitness_提升",
        "box_Precision", "box_Recall", "box_mAP50", "box_mAP50-95",
        "推理耗时(ms)",
    ]

    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["mask_fitness"], reverse=True)
    rows = []
    for exp in sorted_keys:
        r = results[exp]
        meta = EXPERIMENT_META.get(exp, {"attention": "?", "position": "?"})
        dp = (r["params"] - baseline["params"]) if baseline and exp != baseline_key else 0
        df = (r["mask_fitness"] - baseline["mask_fitness"]) if baseline and exp != baseline_key else 0.0
        dp_pct = dp / baseline["params"] * 100 if baseline and baseline["params"] and exp != baseline_key else 0.0

        rows.append({
            "实验": exp,
            "注意力机制": meta["attention"],
            "插入位置": meta["position"],
            "参数量": r["params"],
            "参数增量": dp,
            "参数增量(%)": _R(dp_pct, 2),
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


def save_table2(results, split, baseline_key="baseline", fitness_type="default"):
    """表2: 消融分析——注意力类型对比（位置A） → table2_ablation_type.csv"""
    out_dir = _make_output_dir(split, fitness_type)
    baseline = results.get(baseline_key)
    if not baseline:
        return None
    path = os.path.join(out_dir, "table2_ablation_type.csv")

    position_a = ["baseline", "cbam-a", "simam-a", "ca-a"]
    available = [k for k in position_a if k in results]

    dimension_map = {
        "baseline": ("无", "—"),
        "cbam-a": ("CBAM", "通道+空间"),
        "simam-a": ("SimAM", "通道+空间(无参数)"),
        "ca-a": ("CoordAtt", "通道+位置"),
    }

    fieldnames = [
        "注意力机制", "注意力维度", "额外参数",
        "mask_Precision", "mask_Recall", "mask_mAP50", "mask_mAP50-95",
        "box_Precision", "box_Recall", "box_mAP50", "box_mAP50-95",
        "mask_mAP50-95_vs_baseline", "mask_mAP50-95_vs_baseline(%)",
    ]

    rows = []
    for exp in available:
        r = results[exp]
        label, dim = dimension_map.get(exp, (exp, "?"))
        dp = (r["params"] - baseline["params"]) if exp != baseline_key else 0
        dm = (r["mask_map"] - baseline["mask_map"]) if exp != baseline_key else 0.0
        dm_pct = dm / baseline["mask_map"] * 100 if baseline["mask_map"] and exp != baseline_key else 0.0

        rows.append({
            "注意力机制": label,
            "注意力维度": dim,
            "额外参数": dp,
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


def save_table3(results, split, baseline_key="baseline", fitness_type="default"):
    """表3: 消融分析——插入位置对比（A vs C） → table3_ablation_pos.csv"""
    out_dir = _make_output_dir(split, fitness_type)
    baseline = results.get(baseline_key)
    if not baseline:
        return None
    path = os.path.join(out_dir, "table3_ablation_pos.csv")

    pairs = [
        ("CBAM", "cbam-a", "cbam-c"),
        ("CoordAtt", "ca-a", "ca-c"),
    ]

    fieldnames = [
        "注意力机制", "插入位置", "插入层数", "额外参数",
        "mask_mAP50", "mask_mAP50-95", "mask_mAP50-95_vs_baseline",
        "box_mAP50", "box_mAP50-95",
        "推理耗时(ms)", "位置C_vs_A",
    ]

    rows = []
    br = baseline
    rows.append({
        "注意力机制": "无(baseline)",
        "插入位置": "—",
        "插入层数": 0,
        "额外参数": 0,
        "mask_mAP50": _R(br["mask_map50"]),
        "mask_mAP50-95": _R(br["mask_map"]),
        "mask_mAP50-95_vs_baseline": 0.0,
        "box_mAP50": _R(br["box_map50"]),
        "box_mAP50-95": _R(br["box_map"]),
        "推理耗时(ms)": _R(br["inference_ms"], 2),
        "位置C_vs_A": "",
    })

    for mech, key_a, key_c in pairs:
        ra = results.get(key_a)
        rc = results.get(key_c)

        for pos_label, layers, key, r in [("A: SPPF后", 1, key_a, ra), ("C: Neck C2f后", 4, key_c, rc)]:
            if not r:
                continue
            dp = r["params"] - baseline["params"]
            dm = r["mask_map"] - baseline["mask_map"]

            c_vs_a = ""
            if pos_label.startswith("C") and ra:
                c_vs_a = _R(r["mask_map"] - ra["mask_map"])

            rows.append({
                "注意力机制": mech,
                "插入位置": pos_label,
                "插入层数": layers,
                "额外参数": dp,
                "mask_mAP50": _R(r["mask_map50"]),
                "mask_mAP50-95": _R(r["mask_map"]),
                "mask_mAP50-95_vs_baseline": _R(dm),
                "box_mAP50": _R(r["box_map50"]),
                "box_mAP50-95": _R(r["box_map"]),
                "推理耗时(ms)": _R(r["inference_ms"], 2),
                "位置C_vs_A": c_vs_a,
            })

    _write_csv(path, fieldnames, rows)
    print(f"  表3 已保存: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="注意力机制对比评估")
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
        print(f"  未找到实验结果，请先运行: python cmd/yolov8-seg-attention-train.py")
        print(f"  查找路径: {RUNS_DIR}/attnv2_*/weights/best.pt")
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
    print_ablation_type(results)
    print_ablation_position(results)

    out_dir = _make_output_dir(args.split, args.fitness_type)
    save_table1(results, args.split, args.fitness_type)
    save_table2(results, args.split, fitness_type=args.fitness_type)
    save_table3(results, args.split, fitness_type=args.fitness_type)

    print(f"\n  全部完成，共评估 {len(results)} 个实验")
    print(f"  结果目录: {out_dir}")


if __name__ == "__main__":
    main()

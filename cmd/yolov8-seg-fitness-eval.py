"""
统一模型评估脚本 — 使用面向 OCR 流水线的新 Fitness 指标体系

====================== Fitness 函数说明 ======================

  通过 model.val() 获取原始评估指标（P/R/mAP 等），
  然后在脚本内基于 Metric.FITNESS_PRESETS 离线计算两种 Fitness
  用于新旧方案对比。model.val() 本身的 fitness 值不参与排名。

  预设:
    default       — 0.1×mAP50 + 0.9×mAP95 (原 Ultralytics 默认)
    recall_map75  — 0.2×Recall + 0.5×mAP@0.75 + 0.3×mAP@0.5:0.95

===============================================================

用法:
  1. 评估指定模型:
     python cmd/yolov8-seg-fitness-eval.py --model path/to/best.pt

  2. 评估多个模型，结果放同一目录:
     python cmd/yolov8-seg-fitness-eval.py --model a.pt b.pt c.pt --outdir runs/segment/my_compare

  3. 自动扫描全部实验 (attnv2_* + loss_*):
     python cmd/yolov8-seg-fitness-eval.py --scan

  4. 扫描 + 同时评估 best.pt 和 last.pt:
     python cmd/yolov8-seg-fitness-eval.py --scan --weights both

产出 (全部在 --outdir 指定的目录下):
  fitness_eval_results.csv  — 全部模型的新旧指标对比
  val_{model_name}/         — 每个模型的 YOLO 原生评估产物 (混淆矩阵、PR 曲线等)
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

FITNESS_PRESETS = Metric.FITNESS_PRESETS


def compute_fitness(metric_obj, fitness_type):
    """使用框架 Metric 的预设计算 fitness (与 Metric.fitness() 逻辑一致)。"""
    preset = FITNESS_PRESETS.get(fitness_type, FITNESS_PRESETS["default"])
    w = preset["w"]
    if preset["keys"] == "recall_map75_map":
        vals = np.array([metric_obj.mr, metric_obj.map75, metric_obj.map])
    else:
        vals = np.array(metric_obj.mean_results())
    return float((vals * w).sum())


def compute_f2(p, r):
    denom = 4 * p + r
    if denom < 1e-16:
        return 0.0
    return (5 * p * r) / denom


def eval_model(weights_path, split, eval_name, outdir):
    """
    评估单个模型，同时离线计算 default 和 recall_map75 两种 Fitness。

    model.val() 获取原始指标，两种 Fitness 均在脚本内从 metrics.seg
    离线重算（不依赖 model.val() 输出的 fitness 值）。
    """
    model = YOLO(weights_path)
    metrics = model.val(
        data=DATA_YAML,
        split=split,
        imgsz=IMGSZ,
        verbose=False,
        project=outdir,
        name=f"val_{eval_name}",
        exist_ok=True,
        workers=0,
    )

    n_params = sum(p.numel() for p in model.model.parameters())
    speed = metrics.speed

    seg = metrics.seg
    box = metrics.box

    seg_map75 = seg.map75
    seg_map90 = seg.all_ap[:, :9].mean() if len(seg.all_ap) else 0.0
    box_map75 = box.map75

    old_fit = compute_fitness(seg, "default")
    new_fit = compute_fitness(seg, "recall_map75")
    f2 = compute_f2(seg.mp, seg.mr)

    return {
        "weights": weights_path,
        "params": n_params,
        "mask_p": seg.mp,
        "mask_r": seg.mr,
        "mask_miss_rate": 1.0 - seg.mr,
        "mask_f2": f2,
        "mask_map50": seg.map50,
        "mask_map75": seg_map75,
        "mask_map90": seg_map90,
        "mask_map95": seg.map,
        "box_p": box.mp,
        "box_r": box.mr,
        "box_map50": box.map50,
        "box_map75": box_map75,
        "box_map95": box.map,
        "old_fitness": old_fit,
        "new_fitness": new_fit,
        "inference_ms": speed.get("inference", 0),
    }


def scan_experiments(weights_choice):
    found = {}
    if not os.path.isdir(RUNS_DIR):
        return found

    for name in sorted(os.listdir(RUNS_DIR)):
        is_attn = name.startswith("attnv2_") and not name.startswith("attnv2_eval") and not name.startswith("attnv2_analysis")
        is_loss = name.startswith("loss_") and not name.startswith("loss_eval") and not name.startswith("loss_analysis")
        if not (is_attn or is_loss):
            continue

        weights_dir = os.path.join(RUNS_DIR, name, "weights")
        if not os.path.isdir(weights_dir):
            continue

        targets = []
        if weights_choice in ("best", "both"):
            bp = os.path.join(weights_dir, "best.pt")
            if os.path.exists(bp):
                targets.append(("best", bp))
        if weights_choice in ("last", "both"):
            lp = os.path.join(weights_dir, "last.pt")
            if os.path.exists(lp):
                targets.append(("last", lp))

        for wtype, wpath in targets:
            label = f"{name}/{wtype}" if weights_choice == "both" else name
            found[label] = wpath

    return found


def print_results(results):
    sep = "=" * 155
    print(f"\n{sep}")
    print(f"  统一评估  data={DATA_YAML}  imgsz={IMGSZ}")
    print(f"  default  = {FITNESS_PRESETS['default']['w']}  ×  [P, R, mAP50, mAP95]")
    print(f"  recall_map75 = {FITNESS_PRESETS['recall_map75']['w']}  ×  [Recall, mAP75, mAP95]")
    print(sep)

    header = (
        f"{'模型':>28} | {'Recall':>7} {'漏检率':>7} {'mAP75':>7} {'mAP50':>7} {'mAP95':>7} |"
        f" {'新Fit':>7} {'旧Fit':>7} {'Δ排名':>6} |"
        f" {'F2':>7} {'P':>7} | {'推理ms':>7}"
    )
    print(header)
    print("-" * 155)

    by_new = sorted(results.keys(), key=lambda k: results[k]["new_fitness"], reverse=True)
    by_old = sorted(results.keys(), key=lambda k: results[k]["old_fitness"], reverse=True)

    old_rank = {k: i + 1 for i, k in enumerate(by_old)}
    new_rank = {k: i + 1 for i, k in enumerate(by_new)}

    for name in by_new:
        r = results[name]
        rank_delta = old_rank[name] - new_rank[name]
        if rank_delta > 0:
            rank_str = f"↑{rank_delta}"
        elif rank_delta < 0:
            rank_str = f"↓{-rank_delta}"
        else:
            rank_str = "="

        print(
            f"{name:>28} | {r['mask_r']:>7.4f} {r['mask_miss_rate']:>7.4f} "
            f"{r['mask_map75']:>7.4f} {r['mask_map50']:>7.4f} {r['mask_map95']:>7.4f} |"
            f" {r['new_fitness']:>7.4f} {r['old_fitness']:>7.4f} {rank_str:>6} |"
            f" {r['mask_f2']:>7.4f} {r['mask_p']:>7.4f} | {r['inference_ms']:>7.1f}"
        )

    print(sep)

    best_new = by_new[0]
    best_old = by_old[0]
    if best_new != best_old:
        print(f"\n  recall_map75 最优: {best_new}  (Recall={results[best_new]['mask_r']:.4f}, mAP75={results[best_new]['mask_map75']:.4f})")
        print(f"  default 最优:      {best_old}  (mAP95={results[best_old]['mask_map95']:.4f})")
        print(f"  → 新旧 Fitness 选出了不同的最优模型！")
    else:
        print(f"\n  新旧 Fitness 最优一致: {best_new}")

    print()


def save_csv(results, outdir):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "fitness_eval_results.csv")

    fieldnames = [
        "模型", "权重路径", "参数量",
        "mask_Recall", "mask_漏检率", "mask_F2",
        "mask_Precision", "mask_mAP50", "mask_mAP75", "mask_mAP50-90", "mask_mAP50-95",
        "box_Recall", "box_Precision", "box_mAP50", "box_mAP75", "box_mAP50-95",
        "Fitness(recall_map75)", "Fitness(default)", "推理耗时(ms)",
    ]

    R = lambda v, n=5: round(v, n)

    by_new = sorted(results.keys(), key=lambda k: results[k]["new_fitness"], reverse=True)
    rows = []
    for name in by_new:
        r = results[name]
        rows.append({
            "模型": name,
            "权重路径": r["weights"],
            "参数量": r["params"],
            "mask_Recall": R(r["mask_r"]),
            "mask_漏检率": R(r["mask_miss_rate"]),
            "mask_F2": R(r["mask_f2"]),
            "mask_Precision": R(r["mask_p"]),
            "mask_mAP50": R(r["mask_map50"]),
            "mask_mAP75": R(r["mask_map75"]),
            "mask_mAP50-90": R(r["mask_map90"]),
            "mask_mAP50-95": R(r["mask_map95"]),
            "box_Recall": R(r["box_r"]),
            "box_Precision": R(r["box_p"]),
            "box_mAP50": R(r["box_map50"]),
            "box_mAP75": R(r["box_map75"]),
            "box_mAP50-95": R(r["box_map95"]),
            "Fitness(recall_map75)": R(r["new_fitness"]),
            "Fitness(default)": R(r["old_fitness"]),
            "推理耗时(ms)": R(r["inference_ms"], 2),
        })

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  CSV 已保存: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="统一模型评估 — 新旧 Fitness 离线重算对比 (基于框架预设权重)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python cmd/yolov8-seg-fitness-eval.py --model runs/segment/attnv2_baseline/weights/best.pt
  python cmd/yolov8-seg-fitness-eval.py --model a.pt b.pt --outdir runs/segment/my_compare
  python cmd/yolov8-seg-fitness-eval.py --scan
  python cmd/yolov8-seg-fitness-eval.py --scan --weights both --split test""",
    )
    parser.add_argument("--model", nargs="+", default=None, help="模型权重路径 (可指定多个)")
    parser.add_argument("--scan", action="store_true", help="自动扫描 attnv2_* 和 loss_* 实验")
    parser.add_argument("--weights", choices=["best", "last", "both"], default="best",
                        help="扫描时评估哪些权重 (default: best)")
    parser.add_argument("--split", default="val", help="验证集 (val/test)")
    parser.add_argument("--data", default=None, help="数据集 YAML (default: yolov8-bvn.yaml)")
    parser.add_argument("--outdir", default=None,
                        help="所有结果的统一输出目录 (default: runs/segment/fitness_eval_{split})")
    args = parser.parse_args()

    global DATA_YAML
    if args.data:
        DATA_YAML = args.data

    outdir = args.outdir or os.path.join(RUNS_DIR, f"fitness_eval_{args.split}")
    os.makedirs(outdir, exist_ok=True)

    models = {}

    if args.model:
        for p in args.model:
            if not os.path.exists(p):
                print(f"  [WARN] 文件不存在: {p}")
                continue
            parent = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(p))))
            wtype = os.path.splitext(os.path.basename(p))[0]
            label = f"{parent}/{wtype}" if parent != "weights" else os.path.basename(p)
            models[label] = p

    if args.scan:
        scanned = scan_experiments(args.weights)
        models.update(scanned)

    if not models:
        print("  未指定模型。使用 --model <路径> 或 --scan 自动扫描。")
        print("  示例: python cmd/yolov8-seg-fitness-eval.py --scan")
        return

    print(f"\n  输出目录: {outdir}")
    print(f"  Fitness 预设: {list(FITNESS_PRESETS.keys())}")
    print(f"  待评估模型 ({len(models)} 个):")
    for name, path in models.items():
        print(f"    {name:>30} -> {path}")

    results = {}
    for name, wpath in models.items():
        safe_name = name.replace("/", "_").replace("\\", "_")
        print(f"\n  评估 {name} ...")
        start = time.time()
        try:
            results[name] = eval_model(wpath, args.split, safe_name, outdir)
            elapsed = time.time() - start
            r = results[name]
            print(f"    完成 ({elapsed:.1f}s)  Recall={r['mask_r']:.4f}  mAP75={r['mask_map75']:.4f}  mAP95={r['mask_map95']:.4f}")
        except Exception as e:
            print(f"    失败: {e}")

    if not results:
        print("  无模型成功评估。")
        return

    print_results(results)
    save_csv(results, outdir)

    print(f"\n  全部完成，共评估 {len(results)} 个模型")
    print(f"  结果目录: {outdir}")
    print(f"    fitness_eval_results.csv  — 全部指标对比表")
    for name in results:
        safe = name.replace("/", "_").replace("\\", "_")
        print(f"    val_{safe}/  — {name} 的 YOLO 评估产物")


if __name__ == "__main__":
    main()

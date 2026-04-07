"""
损失函数改进对比实验训练脚本

工作流程:
  1. 从 yolov8n-seg.pt COCO 预训练权重出发
  2. baseline 使用默认 CIoU（对照组）
  3. 各实验仅切换 iou_type 参数，模型架构完全一致
  4. 所有实验统一配置，确保公平对比

实验矩阵:
  baseline   - CIoU（默认，对照组）
  eiou       - EIoU（宽高分离惩罚）
  siou       - SIoU（角度感知）
  wiou       - WIoU（动态聚焦加权）

用法:
  python cmd/yolov8-seg-loss-train.py                   # 交互选择
  python cmd/yolov8-seg-loss-train.py --exp eiou         # 直接运行
  python cmd/yolov8-seg-loss-train.py --exp all          # 全部实验
  python cmd/yolov8-seg-loss-train.py --exp all --resume # 从中断处恢复
"""

import argparse
import gc
import os
import time

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from ultralytics import YOLO

# ======================== 配置区 ========================

BASE_MODEL = "yolov8n-seg.pt"

DATA_YAML = "yolov8-bvn.yaml"

EXP_PREFIX = "loss"

EXPERIMENTS = {
    "baseline": {
        "desc": "CIoU 默认损失（对照组）",
        "iou_type": "CIoU",
    },
    "eiou": {
        "desc": "EIoU（宽高分离惩罚，2022）",
        "iou_type": "EIoU",
    },
    "siou": {
        "desc": "SIoU（角度+形状感知，2022）",
        "iou_type": "SIoU",
    },
    "wiou": {
        "desc": "WIoU（动态聚焦加权，2023）",
        "iou_type": "WIoU",
    },
}

TRAIN_PARAMS = {
    "epochs": 500,
    "batch": 16,
    "imgsz": 960,
    "patience": 50,
    "workers": 0,
    "optimizer": "auto",
}

# ======================== 核心逻辑 ========================

STATUS_COMPLETED = "completed"
STATUS_INTERRUPTED = "interrupted"
STATUS_NOT_STARTED = "not_started"


def get_exp_dir(exp_name):
    return os.path.join("runs", "segment", f"{EXP_PREFIX}_{exp_name}")


def _count_completed_epochs(exp_dir):
    results_csv = os.path.join(exp_dir, "results.csv")
    if not os.path.exists(results_csv):
        return 0
    try:
        with open(results_csv, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return max(len(lines) - 1, 0)
    except Exception:
        return 0


def get_experiment_status(exp_name):
    exp_dir = get_exp_dir(exp_name)
    results_png = os.path.join(exp_dir, "results.png")
    best_pt = os.path.join(exp_dir, "weights", "best.pt")
    last_pt = os.path.join(exp_dir, "weights", "last.pt")

    completed_epochs = _count_completed_epochs(exp_dir)

    if os.path.exists(results_png):
        return STATUS_COMPLETED, best_pt, f"{completed_epochs} epochs 已完成"

    if os.path.exists(last_pt):
        return STATUS_INTERRUPTED, last_pt, f"中断于 epoch {completed_epochs}"

    return STATUS_NOT_STARTED, None, ""


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_experiment(exp_name, base_model, lr0, resume=False):
    exp = EXPERIMENTS[exp_name]

    status, ckpt_path, info = get_experiment_status(exp_name)

    if resume and status == STATUS_COMPLETED:
        print(f"\n  [跳过] {exp_name} 已完成（{info}）")
        return 0.0

    is_resuming = resume and status == STATUS_INTERRUPTED

    print(f"\n{'='*60}")
    print(f"  实验: {exp_name}")
    print(f"  说明: {exp['desc']}")
    print(f"  IoU 类型: {exp['iou_type']}")
    if is_resuming:
        print(f"  ** 恢复模式: 从 {ckpt_path} 继续训练（{info}）**")
    else:
        print(f"  基础权重: {base_model}")
    print(f"  学习率: lr0={lr0}  optimizer={TRAIN_PARAMS['optimizer']}")
    print(f"{'='*60}\n")

    cleanup_memory()

    if is_resuming:
        model = YOLO(ckpt_path)
        start = time.time()
        model.train(resume=True)
    else:
        model = YOLO(base_model)
        start = time.time()
        model.train(
            data=DATA_YAML,
            project="runs/segment",
            name=f"{EXP_PREFIX}_{exp_name}",
            lr0=lr0,
            iou_type=exp["iou_type"],
            **TRAIN_PARAMS,
        )

    elapsed = time.time() - start

    del model
    cleanup_memory()

    mode_str = "恢复" if is_resuming else "完成"
    print(f"\n  {exp_name} {mode_str}训练，耗时 {elapsed/60:.1f} 分钟")
    print(f"  权重: {get_exp_dir(exp_name)}/weights/best.pt")
    return elapsed


def select_experiment():
    print("\n可用实验:")
    for i, (name, exp) in enumerate(EXPERIMENTS.items(), 1):
        print(f"  {i}. {name:>12}  -  {exp['desc']}")
    print(f"  {len(EXPERIMENTS)+1}. {'all':>12}  -  按顺序运行全部实验")

    choice = input("\n选择实验编号或名称: ").strip().lower()

    keys = list(EXPERIMENTS.keys())
    if choice == "all" or choice == str(len(EXPERIMENTS) + 1):
        return keys
    if choice in EXPERIMENTS:
        return [choice]
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(keys):
            return [keys[idx]]
    except ValueError:
        pass

    print("  无效选择，使用 baseline")
    return ["baseline"]


def main():
    parser = argparse.ArgumentParser(description="YOLOv8-seg 损失函数改进对比实验")
    parser.add_argument("--exp", type=str, default=None,
                        help="实验名称 (baseline/eiou/siou/wiou/all)")
    parser.add_argument("--base-model", type=str, default=None,
                        help=f"基础模型权重路径 (default: {BASE_MODEL})")
    parser.add_argument("--data", type=str, default=None, help="数据集 YAML")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="初始学习率 (default: 0.01)")
    parser.add_argument("--resume", action="store_true",
                        help="恢复模式: 跳过已完成，从 last.pt 断点续训")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="跳过 baseline 训练（复用注意力实验的 attnv2_baseline）")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    args = parser.parse_args()

    base_model = args.base_model or BASE_MODEL
    if args.data:
        global DATA_YAML
        DATA_YAML = args.data
    if args.epochs:
        TRAIN_PARAMS["epochs"] = args.epochs
    if args.batch:
        TRAIN_PARAMS["batch"] = args.batch
    if args.imgsz:
        TRAIN_PARAMS["imgsz"] = args.imgsz

    if not os.path.exists(base_model):
        print(f"  错误: 基础模型不存在: {base_model}")
        print(f"  请检查路径或使用 --base-model 指定")
        return

    if args.exp:
        if args.exp == "all":
            to_run = list(EXPERIMENTS.keys())
        elif args.exp in EXPERIMENTS:
            to_run = [args.exp]
        else:
            print(f"未知实验: {args.exp}")
            print(f"可选: {', '.join(EXPERIMENTS.keys())}, all")
            return
    else:
        to_run = select_experiment()

    print(f"\n  基础模型: {base_model}")
    print(f"  数据集: {DATA_YAML}")
    print(f"  优化器: {TRAIN_PARAMS['optimizer']}  lr0={args.lr0}")
    print(f"  参数: epochs={TRAIN_PARAMS['epochs']} batch={TRAIN_PARAMS['batch']} imgsz={TRAIN_PARAMS['imgsz']}")
    print(f"  输出前缀: {EXP_PREFIX}_")
    print(f"  恢复模式: {'开启' if args.resume else '关闭'}")
    print(f"  待运行: {', '.join(to_run)}")

    if args.resume:
        print(f"\n  --- 实验状态扫描 ---")
        for exp_name in to_run:
            status, _, info = get_experiment_status(exp_name)
            label = {
                STATUS_COMPLETED: "已完成 → 跳过",
                STATUS_INTERRUPTED: "已中断 → 从 last.pt 恢复",
                STATUS_NOT_STARTED: "未开始 → 全新训练",
            }[status]
            suffix = f"  ({info})" if info else ""
            print(f"    {exp_name:>12}: {label}{suffix}")
        print()

    timings = {}
    skipped = []
    failed = []
    for exp_name in to_run:
        if exp_name == "baseline" and args.skip_baseline:
            attn_baseline = os.path.join("runs", "segment", "attnv2_baseline", "weights", "best.pt")
            if os.path.exists(attn_baseline):
                print(f"\n  [跳过] baseline 训练（复用注意力实验基线: {attn_baseline}）")
                skipped.append(exp_name)
                continue
            else:
                print(f"\n  [警告] --skip-baseline 但未找到 {attn_baseline}，将正常训练 baseline")
        try:
            elapsed = run_experiment(exp_name, base_model, args.lr0, resume=args.resume)
            if args.resume and elapsed == 0.0:
                skipped.append(exp_name)
            else:
                timings[exp_name] = elapsed
        except Exception as e:
            print(f"\n  !! {exp_name} 失败: {e}")
            print(f"  !! 跳过，继续下一个实验...\n")
            failed.append(exp_name)
            cleanup_memory()

    print(f"\n{'='*60}")
    if skipped:
        print(f"  已跳过（之前已完成）: {', '.join(skipped)}")
    if timings:
        print("  本次完成的实验:")
        for name, t in timings.items():
            print(f"    {name:>12}: {t/60:.1f} 分钟")
    if failed:
        print(f"\n  失败的实验: {', '.join(failed)}")
        print(f"  恢复训练: python cmd/yolov8-seg-loss-train.py --exp all --resume")
    if timings or skipped:
        print(f"\n  下一步: python cmd/yolov8-seg-loss-val.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

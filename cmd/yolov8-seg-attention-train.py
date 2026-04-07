"""
注意力机制对比实验训练脚本（从 yolov8n-seg.pt 预训练权重全新训练）

工作流程:
  1. 从 yolov8n-seg.pt COCO 预训练权重出发
  2. baseline 直接训练原始架构（对照组）
  3. 注意力实验: 构建新架构 → 迁移预训练权重 → 全量训练
  4. 所有实验统一配置，确保公平对比

实验矩阵:
  baseline   - yolov8n-seg 原始架构（对照组）
  cbam-a     - CBAM + 位置A（Backbone末尾）
  simam-a    - SimAM + 位置A（零参数3D注意力）
  ca-a       - CoordAtt + 位置A（位置编码）
  cbam-c     - CBAM + 位置C（Neck 4处增强）
  ca-c       - CoordAtt + 位置C（Neck 4处增强）

用法:
  python cmd/yolov8-seg-attention-train.py                   # 交互选择
  python cmd/yolov8-seg-attention-train.py --exp cbam-a      # 直接运行
  python cmd/yolov8-seg-attention-train.py --exp all         # 全部实验
  python cmd/yolov8-seg-attention-train.py --exp all --resume # 从中断处恢复
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

SCALE = "n"

EXP_PREFIX = "attnv2"

EXPERIMENTS = {
    "baseline": {
        "desc": "yolov8n-seg 原始架构（对照组）",
        "yaml": None,
        "attention": "无",
        "position": "-",
    },
    "cbam-a": {
        "desc": "CBAM + Backbone末尾（通道+空间注意力）",
        "yaml": f"yolov8{SCALE}-seg-cbam-a.yaml",
        "attention": "CBAM",
        "position": "A: SPPF后",
    },
    "simam-a": {
        "desc": "SimAM + Backbone末尾（零参数3D注意力）",
        "yaml": f"yolov8{SCALE}-seg-simam-a.yaml",
        "attention": "SimAM",
        "position": "A: SPPF后",
    },
    "ca-a": {
        "desc": "CoordAtt + Backbone末尾（坐标注意力）",
        "yaml": f"yolov8{SCALE}-seg-ca-a.yaml",
        "attention": "CoordAtt",
        "position": "A: SPPF后",
    },
    "cbam-c": {
        "desc": "CBAM + Neck增强（4处C2f后）",
        "yaml": f"yolov8{SCALE}-seg-cbam-c.yaml",
        "attention": "CBAM",
        "position": "C: Neck C2f后",
    },
    "ca-c": {
        "desc": "CoordAtt + Neck增强（4处C2f后）",
        "yaml": f"yolov8{SCALE}-seg-ca-c.yaml",
        "attention": "CoordAtt",
        "position": "C: Neck C2f后",
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
    """从 results.csv 统计已完成的 epoch 数（仅用于显示进度）。"""
    results_csv = os.path.join(exp_dir, "results.csv")
    if not os.path.exists(results_csv):
        return 0
    try:
        with open(results_csv, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return max(len(lines) - 1, 0)
    except Exception:
        return 0


def get_experiment_status(exp_name):
    """
    检测实验状态，返回 (status, checkpoint_path, info_str)。

    判定逻辑:
      1. results.png 存在 → completed（YOLO 仅在训练正常结束后绘制）
      2. last.pt 存在 → interrupted（可从断点恢复）
      3. 以上都不满足 → not_started
    """
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
    """释放 GPU 和系统内存，防止连续实验时 OOM。"""
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
    print(f"  注意力: {exp['attention']}  位置: {exp['position']}")
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
        if exp["yaml"] is None:
            model = YOLO(base_model)
        else:
            model = YOLO(exp["yaml"])
            ckpt = YOLO(base_model)
            src_state = ckpt.model.state_dict()
            dst_state = model.model.state_dict()

            matched, skipped = 0, 0
            new_state = {}
            for k, v in dst_state.items():
                if k in src_state and src_state[k].shape == v.shape:
                    new_state[k] = src_state[k]
                    matched += 1
                else:
                    new_state[k] = v
                    skipped += 1

            model.model.load_state_dict(new_state)
            print(f"  权重迁移: {matched} 层匹配加载, {skipped} 层随机初始化（注意力模块）")
            del ckpt, src_state, dst_state, new_state
            cleanup_memory()

        start = time.time()
        model.train(
            data=DATA_YAML,
            project="runs/segment",
            name=f"{EXP_PREFIX}_{exp_name}",
            lr0=lr0,
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
    parser = argparse.ArgumentParser(description="YOLOv8-seg 注意力机制对比实验（从预训练全新训练）")
    parser.add_argument("--exp", type=str, default=None,
                        help="实验名称 (baseline/cbam-a/simam-a/ca-a/cbam-c/ca-c/all)")
    parser.add_argument("--base-model", type=str, default=None,
                        help=f"基础模型权重路径 (default: {BASE_MODEL})")
    parser.add_argument("--data", type=str, default=None, help="数据集 YAML")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="初始学习率 (default: 0.01)")
    parser.add_argument("--resume", action="store_true",
                        help="恢复模式: 跳过已完成实验，从 last.pt 断点续训未完成实验")
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
        print(f"  恢复训练: python cmd/yolov8-seg-attention-train.py --exp all --resume")
        print(f"  单独重试: python cmd/yolov8-seg-attention-train.py --exp <name> --resume")
    if timings or skipped:
        print(f"\n  下一步: python cmd/yolov8-seg-attention-val.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

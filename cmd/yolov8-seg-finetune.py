"""
场景补充微调训练脚本

适用场景:
  在已有模型基础上，补充新场景数据进行微调训练。
  新旧数据一起训练，通过参数调整平衡新旧数据的学习。

使用前准备:
  1. 采集新场景图片并用 X-AnyLabeling 标注（多边形，贴合不规则轮廓）
  2. 对新数据执行增强: python cmd/batch_image_augmentation.py
  3. 将新数据（含增强）放入 datasets/bvn/images/train 和 labels/train（与旧数据合并）
  4. 新场景的 20% 放入 val 集（不做增强）
  5. 删除 datasets/bvn/labels/ 下的 .cache 文件（强制重新扫描数据集）

数据比例建议:
  新旧数据比例尽量保持在 1:3 ~ 1:1 之间
  如果新数据太少，用 cmd/batch_image_augmentation.py 增强到合理比例

用法:
  python cmd/yolov8-seg-finetune.py
  按提示选择场景模式，或直接设置 MODE 变量
"""

import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ultralytics import YOLO

# ======================== 配置区 ========================

# 基础模型: 使用历史最优模型作为起点
MODEL_PATH = "./runs/segment/train11/weights/best.pt"

# 数据集配置
DATA_YAML = "yolov8-bvn.yaml"

# 场景模式: 设为 None 则运行时交互选择
# "deform"   - 折叠/扭曲/不规则（强变形增强，标准分辨率）
# "closeup"  - 近距离大目标（弱变形增强，高分辨率，低 mosaic）
# "smallobj" - 远景小目标（标准增强，高分辨率，强 mosaic）
# "general"  - 通用混合场景（均衡参数）
MODE = "general"

# ======================== 场景预设 ========================
#
# 每个场景的参数说明:
#   epochs    - 最大训练轮数（通常被 patience 提前终止）
#   batch     - 批大小（显存不足时减小）
#   imgsz     - 输入尺寸
#   patience  - 连续 N 轮无提升则早停
#   lr0       - 初始学习率（微调用低值，保护已有特征）
#   degrees   - 随机旋转角度
#   shear     - 剪切变换强度（模拟平行四边形变形）
#   perspective - 透视变换强度（模拟拍摄角度变形）
#   scale     - 随机缩放范围
#   mosaic    - 4 图拼接概率（大目标场景要降低，避免裁切）
#   flipud    - 上下翻转概率
#   freeze    - 冻结前 N 层（None=不冻结，新数据极少时设为 10）

PRESETS = {
    "deform": {
        "desc": "折叠/扭曲/不规则: 强变形增强，标准分辨率",
        "epochs": 200, "batch": 24, "imgsz": 960, "patience": 50,
        "lr0": 0.005,
        "degrees": 15.0,      # 较大旋转，模拟倾斜放置
        "shear": 5.0,         # 剪切变换，模拟折叠变形
        "perspective": 0.001, # 透视变换，模拟不同角度
        "scale": 0.7,
        "mosaic": 0.8,
        "flipud": 0.1,
        "freeze": None,
    },
    "closeup": {
        "desc": "近距离大目标: 高分辨率，低 mosaic，弱变形",
        "epochs": 200, "batch": 8, "imgsz": 1280, "patience": 50,
        "lr0": 0.003,         # 更低学习率，精细调整边缘特征
        "degrees": 10.0,      # 近距离旋转角度较小
        "shear": 0.0,         # 不需要剪切，大目标边缘特征重要
        "perspective": 0.0,   # 近距离透视变化小
        "scale": 0.9,         # 大缩放范围，适应占满画面的情况
        "mosaic": 0.3,        # 大幅降低，避免大目标被裁切变不完整
        "flipud": 0.0,        # 近距离通常不会倒置
        "freeze": None,
    },
    "smallobj": {
        "desc": "远景小目标: 高分辨率，强 mosaic 增加密度",
        "epochs": 200, "batch": 8, "imgsz": 1280, "patience": 50,
        "lr0": 0.005,
        "degrees": 10.0,
        "shear": 2.0,
        "perspective": 0.0005,
        "scale": 0.5,         # 标准缩放，不要过度放大小目标
        "mosaic": 1.0,        # 满 mosaic，增加单图中的小目标密度
        "flipud": 0.1,
        "freeze": None,
    },
    "general": {
        "desc": "通用混合场景: 均衡参数，适用于多种场景混合补充",
        "epochs": 200, "batch": 16, "imgsz": 960, "patience": 50,
        "lr0": 0.005,
        "degrees": 10.0,
        "shear": 2.0,
        "perspective": 0.0005,
        "scale": 0.6,
        "mosaic": 0.8,
        "flipud": 0.05,
        "freeze": None,
    },
}

# ======================== 训练 ========================

def select_mode():
    print("\nscene modes:")
    for key, preset in PRESETS.items():
        print(f"  {key:>10}  -  {preset['desc']}")
    choice = input(f"\nselect mode ({'/'.join(PRESETS.keys())}): ").strip().lower()
    if choice not in PRESETS:
        print(f"  invalid, using 'general'")
        choice = "general"
    return choice


mode = MODE or select_mode()
cfg = PRESETS[mode]

print(f"\n  mode: {mode} - {cfg['desc']}")
print(f"  model: {MODEL_PATH}")
print(f"  imgsz={cfg['imgsz']} batch={cfg['batch']} lr0={cfg['lr0']}")
print(f"  degrees={cfg['degrees']} shear={cfg['shear']} perspective={cfg['perspective']}")
print(f"  scale={cfg['scale']} mosaic={cfg['mosaic']} flipud={cfg['flipud']}")
print()

model = YOLO(MODEL_PATH)

model.train(
    data=DATA_YAML,
    epochs=cfg["epochs"],
    batch=cfg["batch"],
    imgsz=cfg["imgsz"],
    workers=0,
    patience=cfg["patience"],
    lr0=cfg["lr0"],
    degrees=cfg["degrees"],
    shear=cfg["shear"],
    perspective=cfg["perspective"],
    scale=cfg["scale"],
    mosaic=cfg["mosaic"],
    flipud=cfg["flipud"],
    freeze=cfg["freeze"],
)

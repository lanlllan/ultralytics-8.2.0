"""
断点续训脚本

适用场景:
  训练过程被中断（手动 Ctrl+C、显卡崩溃、断电等），需要从中断处继续训练。
  会完整恢复: 模型权重、优化器状态、epoch 计数、学习率调度等。

原理:
  YOLOv8 每个 epoch 结束时自动保存 last.pt，其中包含完整的训练状态。
  使用 resume=True 加载 last.pt 后，训练会从中断的 epoch 继续，
  所有参数（epochs、lr0、batch 等）沿用原始训练时的设定，无需手动指定。

用法:
  1. 将 LAST_PT_PATH 改为要恢复的 last.pt 路径
  2. 运行: python cmd/yolov8-seg-resume.py

注意:
  - 必须使用 last.pt，不能用 best.pt（best.pt 不含优化器状态和 epoch 信息）
  - resume=True 时不能传入其他训练参数，所有参数从 checkpoint 自动恢复
  - 如果想在中断后调整参数（如降低学习率），请用 cmd/yolov8-seg-finetune.py 加载 last.pt
"""

import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ultralytics import YOLO

# ======================== 配置区 ========================

# 指向要恢复的训练的 last.pt
# 例: "./runs/segment/train8/weights/last.pt"
LAST_PT_PATH = "./runs/segment/train8/weights/last.pt"

# ======================== 断点续训 ========================

model = YOLO(LAST_PT_PATH)
model.train(resume=True)

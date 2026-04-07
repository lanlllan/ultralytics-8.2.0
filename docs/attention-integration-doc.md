# YOLOv8-Seg 注意力机制集成改造技术文档

> 适用于论文方法论（Method）章节的写作参考
>
> 目标模型：YOLOv8n-seg | 任务：单类运单（waybill）实例分割

---

## 目录

1. [研究动机与问题分析](#1-研究动机与问题分析)
2. [基线模型架构分析](#2-基线模型架构分析)
3. [注意力机制选型与设计](#3-注意力机制选型与设计)
4. [网络架构改造方案](#4-网络架构改造方案)
5. [代码工程实现](#5-代码工程实现)
6. [实验设计](#6-实验设计)
7. [实验工具链与执行方式](#7-实验工具链与执行方式)
8. [改造前后参数量对比](#8-改造前后参数量对比)

---

## 1. 研究动机与问题分析

### 1.1 任务特点

运单（waybill）实例分割任务具有以下特殊性：

- **形状不规则**：运单可能存在折叠、弯曲、部分遮挡等变形，边缘轮廓复杂
- **尺度变化大**：近景拍摄时运单占满画面，远景时可能仅占图像一小部分
- **背景干扰**：运单可能放置在包裹、桌面、传送带等复杂背景上，与周围区域对比度不一

### 1.2 引入注意力机制的动机

标准 YOLOv8n-seg 对所有空间位置和通道特征等权处理。在运单分割场景中，引入注意力机制的预期收益：

- **通道注意力**：增强与运单纹理/颜色相关的特征通道响应，抑制背景噪声通道
- **空间注意力**：聚焦运单所在区域的空间特征，提升边缘定位精度
- **位置编码**：编码水平/垂直方向的位置信息，有助于矩形/近矩形运单的边界回归

### 1.3 设计原则

- **轻量化**：在 yolov8n-seg（3.4M 参数）基础上，参数增量控制在 3% 以内
- **即插即用**：注意力模块不改变特征图尺寸，可灵活插入网络任意位置
- **与预训练兼容**：改造后的网络可复用原始 Backbone 的预训练权重

---

## 2. 基线模型架构分析

### 2.1 YOLOv8n-seg 网络结构

YOLOv8n-seg 采用 CSPDarknet Backbone + PANet Neck + Segment Head 的三段式结构：

```
输入 (3, H, W)
│
├── Backbone（CSPDarknet 特征提取）
│   Layer 0:  Conv(3→16, k=3, s=2)         →  (16, H/2, W/2)      P1
│   Layer 1:  Conv(16→32, k=3, s=2)        →  (32, H/4, W/4)      P2
│   Layer 2:  C2f(32→32, n=1)              →  (32, H/4, W/4)
│   Layer 3:  Conv(32→64, k=3, s=2)        →  (64, H/8, W/8)      P3
│   Layer 4:  C2f(64→64, n=2)              →  (64, H/8, W/8)      ──→ Neck
│   Layer 5:  Conv(64→128, k=3, s=2)       →  (128, H/16, W/16)   P4
│   Layer 6:  C2f(128→128, n=2)            →  (128, H/16, W/16)   ──→ Neck
│   Layer 7:  Conv(128→256, k=3, s=2)      →  (256, H/32, W/32)   P5
│   Layer 8:  C2f(256→256, n=1)            →  (256, H/32, W/32)
│   Layer 9:  SPPF(256→256, k=5)           →  (256, H/32, W/32)
│
├── Neck（PANet 双向特征金字塔）
│   Layer 10: Upsample ×2                   →  (256, H/16, W/16)
│   Layer 11: Concat([#10, #6])             →  (384, H/16, W/16)
│   Layer 12: C2f(384→128, n=1)             →  (128, H/16, W/16)
│   Layer 13: Upsample ×2                   →  (128, H/8, W/8)
│   Layer 14: Concat([#13, #4])             →  (192, H/8, W/8)
│   Layer 15: C2f(192→64, n=1)              →  (64, H/8, W/8)     P3 输出
│   Layer 16: Conv(64→64, k=3, s=2)         →  (64, H/16, W/16)
│   Layer 17: Concat([#16, #12])            →  (192, H/16, W/16)
│   Layer 18: C2f(192→128, n=1)             →  (128, H/16, W/16)  P4 输出
│   Layer 19: Conv(128→128, k=3, s=2)       →  (128, H/32, W/32)
│   Layer 20: Concat([#19, #9])             →  (384, H/32, W/32)
│   Layer 21: C2f(384→256, n=1)             →  (256, H/32, W/32)  P5 输出
│
└── Head
    Layer 22: Segment([P3, P4, P5])          →  (nc, 32 proto masks, 256 ch)
```

**缩放参数**（n 级别）：depth_multiple=0.33, width_multiple=0.25, max_channels=1024

### 2.2 基线性能指标


| 指标                 | 值                 |
| ------------------ | ----------------- |
| 总参数量               | 3,409,968 (3.41M) |
| GFLOPs (imgsz=640) | ~12.6             |
| 输入分辨率              | 960×960           |


---

## 3. 注意力机制选型与设计

本文选取三种具有代表性的注意力机制进行对比实验，覆盖不同的注意力建模维度：

### 3.1 CBAM（Convolutional Block Attention Module）

**来源**：Woo S, Park J, Lee J Y, et al. CBAM: Convolutional block attention module[C]. ECCV, 2018.

**设计思想**：CBAM 串联两个子模块——通道注意力模块（Channel Attention Module）和空间注意力模块（Spatial Attention Module），分别在通道维度和空间维度对特征进行自适应校准。

**通道注意力子模块**：

对输入特征 $F \in \mathbb{R}^{C \times H \times W}$，先通过全局平均池化（Global Average Pooling）压缩空间维度得到通道描述符 $\mathbf{z} \in \mathbb{R}^{C \times 1 \times 1}$，再经过 $1 \times 1$ 卷积和 Sigmoid 激活生成通道注意力权重：

$$M_c(F) = \sigma(\text{Conv}_{1 \times 1}(\text{GAP}(F)))$$

$$F' = F \otimes M_c(F)$$

其中 $\sigma$ 为 Sigmoid 函数，$\otimes$ 为逐元素乘法。

**空间注意力子模块**：

对通道注意力加权后的特征 $F'$，沿通道维度分别做平均池化和最大池化，拼接后经 $7 \times 7$ 卷积和 Sigmoid 生成空间注意力图：

$$M_s(F') = \sigma(\text{Conv}_{7 \times 7}([\text{AvgPool}(F'); \text{MaxPool}(F')]))$$

$$\hat{F} = F' \otimes M_s(F')$$

**实现代码**（Ultralytics 代码库已有实现，位于 `ultralytics/nn/modules/conv.py`）：

```python
class CBAM(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))
```

**参数量**：$C^2 + 1$（通道注意力 $1 \times 1$ 卷积）$+ 7 \times 7 \times 2$（空间注意力 $7 \times 7$ 卷积）$= C^2 + 99$

---

### 3.2 SimAM（Simple, Parameter-Free Attention Module）

**来源**：Yang L, Zhang R Y, Li L, et al. SimAM: A simple, parameter-free attention module for convolutional neural networks[C]. ICML, 2021.

**设计思想**：SimAM 从神经科学中的能量函数出发，无需任何可学习参数即可实现 3D 注意力（同时覆盖通道和空间维度）。其核心思想是：对每个神经元评估其相对于同通道其他神经元的"显著性"——越偏离均值的神经元被赋予越高的注意力权重。

**数学表达**：

对于输入特征图中位于通道 $c$、空间位置 $(i, j)$ 的激活值 $x_{c,i,j}$：

$$e_{c,i,j} = \frac{(x_{c,i,j} - \mu_c)^2}{\sigma_c^2 + \epsilon}$$

其中 $\mu_c$ 和 $\sigma_c^2$ 分别为通道 $c$ 上所有空间位置的均值和方差，$\epsilon$ 为稳定性常数（默认 $10^{-4}$）。

最终注意力加权：

$$\hat{x} = x \odot \sigma\left(\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\right)$$

**实现代码**（本文新增）：

```python
class SimAM(nn.Module):
    def __init__(self, c1, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = ((x - mean) ** 2).sum(dim=[2, 3], keepdim=True) / n
        y = (x - mean) / (var + self.e_lambda).sqrt()
        return x * y.sigmoid()
```

**参数量**：**0**（完全无可学习参数）

**优势**：作为消融实验的理想候选——如果零参数的 SimAM 能带来提升，说明注意力机制的"特征选择"效应确实对该任务有帮助，而非仅仅由于额外参数带来的容量提升。

---

### 3.3 CoordAtt（Coordinate Attention）

**来源**：Hou Q, Zhou D, Feng J. Coordinate attention for efficient mobile network design[C]. CVPR, 2021.

**设计思想**：CoordAtt 将位置信息显式编码到通道注意力中。与 CBAM 的全局池化不同，CoordAtt 沿水平方向和垂直方向分别做池化，保留了精确的一维位置信息，然后通过共享变换进行融合。

**计算流程**：

1. **方向解耦池化**：

$$z_c^h(h) = \frac{1}{W}\sum_{0 \leq i < W} x_c(h, i) \quad \in \mathbb{R}^{C \times H \times 1}$$

$$z_c^w(w) = \frac{1}{H}\sum_{0 \leq j < H} x_c(j, w) \quad \in \mathbb{R}^{C \times 1 \times W}$$

1. **拼接与共享变换**：

$$f = \delta(\text{BN}(\text{Conv}_{1 \times 1}([\mathbf{z}^h; \mathbf{z}^w])))$$

其中 $\delta$ 为 SiLU 激活函数，$[\cdot; \cdot]$ 表示沿空间维度拼接，降维比例 $r=32$。

1. **拆分与独立映射**：

$$g^h = \sigma(\text{Conv}_{1 \times 1}^h(f^h)) \quad \in \mathbb{R}^{C \times H \times 1}$$

$$g^w = \sigma(\text{Conv}_{1 \times 1}^w(f^w)) \quad \in \mathbb{R}^{C \times 1 \times W}$$

1. **注意力加权**：

$$\hat{x} = x \odot g^h \odot g^w$$

**实现代码**（本文新增）：

```python
class CoordAtt(nn.Module):
    def __init__(self, c1, reduction=32):
        super().__init__()
        mid = max(c1 // reduction, 8)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(c1, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, c1, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, c1, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = y.split([h, w], dim=2)
        x_h = self.conv_h(x_h).sigmoid()
        x_w = self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()
        return x * x_h * x_w
```

**参数量**：$3 \times C^2 / r + \text{BN}$，其中 $r=32$

**优势**：对运单这类具有明显方向性边缘的目标，水平/垂直方向的位置编码有助于精确定位边界。

### 3.4 三种机制对比


| 机制       | 注意力维度        | 可学习参数 (C=256) | 核心特点             |
| -------- | ------------ | ------------- | ---------------- |
| CBAM     | 通道 + 空间      | ~65,890       | 双维度串联建模，经典稳定     |
| SimAM    | 通道 + 空间 (3D) | **0**         | 无参数，基于能量函数的显著性评估 |
| CoordAtt | 通道 + 位置      | ~6,160        | 编码水平/垂直位置信息，边缘敏感 |


---

## 4. 网络架构改造方案

### 4.1 改造策略

本文采用**即插即用**的改造策略：注意力模块接收特征图 $F \in \mathbb{R}^{C \times H \times W}$，输出相同尺寸的加权特征 $\hat{F} \in \mathbb{R}^{C \times H \times W}$，不改变通道数和空间分辨率。因此可在网络中任意位置插入，无需修改相邻层的参数配置。

### 4.2 插入位置设计

本文设计两种插入策略，从简到繁逐步验证：

#### 位置 A：Backbone 末尾（SPPF 之后）

在 Backbone 的 SPPF 层之后插入一个注意力模块。这是最小改动方案——仅新增 1 层，对 P5 最深层特征进行注意力加权后，通过 FPN 自顶向下传播到 P3/P4 各层。

**改造后的 Backbone 结构**：

```
Layer 0-8:  （与原版完全相同）
Layer 9:    SPPF(256→256, k=5)
Layer 10:   Attention(256)        ← 新增注意力模块
```

**设计考量**：

- 改动量最小（1 层），便于快速验证注意力机制的有效性
- Backbone 前 10 层权重与预训练模型完全兼容
- P5 特征经注意力加权后，语义信息通过 FPN 间接增强 P3/P4

#### 位置 C：Neck 特征融合后（4 处 C2f 之后）

在 Neck 部分每个 C2f 特征融合层之后各插入一个注意力模块（共 4 处），直接增强多尺度融合特征。

**改造后的 Neck 结构**：

```
Layer 12: C2f(384→128)    →  Layer 13: Attention(128)    P4 融合
Layer 15→16: C2f(192→64)  →  Layer 17: Attention(64)     P3 融合（小目标）
Layer 18→20: C2f(192→128) →  Layer 21: Attention(128)    P4 输出
Layer 21→24: C2f(384→256) →  Layer 25: Attention(256)    P5 输出
```

**设计考量**：

- Backbone 完全不变，预训练权重 100% 兼容
- 在特征融合阶段施加注意力，直接作用于输入检测头的最终特征
- 多尺度的注意力加权有助于平衡不同尺度目标的检测精度

### 4.3 两种策略对比


| 策略   | 新增层数 | 预训练兼容         | 影响范围             | 适用场景        |
| ---- | ---- | ------------- | ---------------- | ----------- |
| 位置 A | 1 层  | Backbone 完全兼容 | 仅 P5，间接影响 P3/P4  | 快速验证，最小侵入   |
| 位置 C | 4 层  | Backbone 完全兼容 | 全部 P3/P4/P5 融合特征 | 全面增强，效果上限更高 |


---

## 5. 代码工程实现

### 5.1 技术架构

改造基于 Ultralytics YOLOv8 (v8.2.0) 的模块化架构。YOLOv8 采用 YAML 配置文件定义网络结构，通过 `parse_model()` 函数将配置文件中的模块名映射到 Python 类进行实例化。

### 5.2 注意力模块实现

所有注意力模块统一定义在 `ultralytics/nn/modules/conv.py` 中，遵循以下接口约定：

- `__init__(self, c1, ...)`: 第一个参数为输入通道数
- `forward(self, x) → Tensor`: 输入输出张量形状不变（$B \times C \times H \times W$）

#### 新增模块列表


| 模块类名           | 注意力类型    | 新增文件位置               |
| -------------- | -------- | -------------------- |
| `CBAM`         | 通道+空间    | 已有（conv.py L309-320） |
| `SimAM`        | 3D 无参数   | 新增（conv.py）          |
| `CoordAtt`     | 坐标注意力    | 新增（conv.py）          |
| `SEAttention`  | SE 通道注意力 | 新增（conv.py，备用）       |
| `ECAAttention` | 高效通道注意力  | 新增（conv.py，备用）       |


### 5.3 模型解析器适配

在 `ultralytics/nn/tasks.py` 的 `parse_model()` 函数中，为注意力模块添加通道自动适配逻辑：

```python
elif m in {CBAM, SEAttention, ECAAttention, SimAM, CoordAtt}:
    c1 = ch[f]       # 从上一层获取实际通道数
    c2 = c1           # 注意力模块不改变通道数
    args = [c1, *args[1:]]  # 替换 YAML 中的通道占位符为实际值
```

该处理确保注意力模块在任何模型缩放比例（n/s/m/l/x）下均能正确获取通道数，无需在 YAML 中硬编码实际通道值。

### 5.4 模块注册

注意力模块通过以下注册链可被 YAML 配置文件引用：

```
conv.py (类定义)
  → __init__.py (模块导出)
    → tasks.py (import 到 globals 命名空间)
      → parse_model() (YAML 名称 → Python 类映射)
```

### 5.5 YAML 配置文件定义

以 CBAM + 位置 A 为例（`yolov8-seg-cbam-a.yaml`）：

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # 0
  - [-1, 1, Conv, [128, 3, 2]]       # 1
  - [-1, 3, C2f, [128, True]]        # 2
  - [-1, 1, Conv, [256, 3, 2]]       # 3
  - [-1, 6, C2f, [256, True]]        # 4
  - [-1, 1, Conv, [512, 3, 2]]       # 5
  - [-1, 6, C2f, [512, True]]        # 6
  - [-1, 1, Conv, [1024, 3, 2]]      # 7
  - [-1, 3, C2f, [1024, True]]       # 8
  - [-1, 1, SPPF, [1024, 5]]         # 9
  - [-1, 1, CBAM, [1024]]            # 10 ← 新增注意力层

head:
  # ... Head 层索引相应偏移 +1 ...
  - [[16, 19, 22], 1, Segment, [nc, 32, 256]]
```

### 5.6 预训练权重迁移

改造后的模型通过 Ultralytics 的 `intersect_dicts()` 机制实现权重自动迁移：

1. 构建新架构模型（含注意力层）
2. 加载已有训练权重（如 train11/best.pt）
3. 按 **参数名称 + 张量形状** 逐层匹配
4. 匹配成功的层直接加载权重，未匹配的层（即新增的注意力层）保持随机初始化

```python
# 权重迁移实现
src_state = pretrained_model.state_dict()
dst_state = new_model.state_dict()
for k, v in dst_state.items():
    if k in src_state and src_state[k].shape == v.shape:
        new_state[k] = src_state[k]   # 匹配层：加载预训练权重
    else:
        new_state[k] = v               # 新增层：保持随机初始化
```

---

## 6. 实验设计

### 6.1 实验矩阵

本文设计 6 组对比实验，覆盖 **3 种注意力机制 × 2 种插入位置**，加 1 组对照基线：


| 实验编号  | 实验名称     | 注意力机制    | 插入位置               | YAML 配置文件               | 目的                   |
| ----- | -------- | -------- | ------------------ | ----------------------- | -------------------- |
| Exp-0 | baseline | 无        | -                  | 原版 yolov8n-seg          | 对照组（同等条件继续训练）        |
| Exp-1 | cbam-a   | CBAM     | A: SPPF 后          | yolov8-seg-cbam-a.yaml  | 验证通道+空间注意力的有效性       |
| Exp-2 | simam-a  | SimAM    | A: SPPF 后          | yolov8-seg-simam-a.yaml | 零参数消融，排除容量因素         |
| Exp-3 | ca-a     | CoordAtt | A: SPPF 后          | yolov8-seg-ca-a.yaml    | 验证位置编码对边缘分割的增益       |
| Exp-4 | cbam-c   | CBAM     | C: Neck C2f 后 (×4) | yolov8-seg-cbam-c.yaml  | 验证 Neck 全面增强的效果上限    |
| Exp-5 | ca-c     | CoordAtt | C: Neck C2f 后 (×4) | yolov8-seg-ca-c.yaml    | 验证位置编码 + Neck 增强综合效果 |


#### 各实验详细说明

**Exp-0 baseline（对照组）**

直接加载已有最优模型（train11/best.pt）在相同超参数下继续训练。目的是排除"多训练几轮就能提升"的干扰因素，建立公平的对比基准。基线模型为标准 yolov8n-seg，不引入任何注意力模块。

**Exp-1 cbam-a（CBAM + 位置 A）**

在 Backbone 末尾（SPPF 之后）插入 1 个 CBAM 模块。CBAM 串联通道注意力和空间注意力，通道注意力通过全局池化→$1 \times 1$卷积学习各通道的重要性权重，空间注意力通过$7 \times 7$卷积学习各空间位置的重要性。本实验是改动最小的方案（仅新增 1 层），作为注意力机制有效性的首次验证。

**Exp-2 simam-a（SimAM + 位置 A）**

在相同位置（SPPF 后）插入 SimAM。SimAM 完全不引入可学习参数，通过统计每个神经元相对于同通道均值的偏离程度来计算注意力权重。本实验作为**消融对照**——若零参数的 SimAM 也能带来提升，则证明注意力机制的"特征选择"本身对运单分割有帮助，而非额外参数带来的模型容量提升。

**Exp-3 ca-a（CoordAtt + 位置 A）**

在相同位置（SPPF 后）插入 CoordAtt。CoordAtt 沿水平和垂直方向分别池化并编码位置信息，对具有明显方向性边缘的运单目标尤其有利。本实验验证位置编码信息对分割边缘精度的增益。

**Exp-4 cbam-c（CBAM + 位置 C）**

在 Neck 的每个 C2f 特征融合层之后各插入 1 个 CBAM（共 4 处），直接增强 P3/P4/P5 三个尺度的融合特征。Backbone 完全不变，预训练权重 100% 兼容。与 Exp-1 对比可验证：在特征融合阶段施加注意力是否优于仅增强深层特征。

**Exp-5 ca-c（CoordAtt + 位置 C）**

在 Neck 的 4 处 C2f 之后各插入 CoordAtt。与 Exp-4 对比可验证 CoordAtt 在多尺度增强场景下的表现；与 Exp-3 对比可验证插入更多注意力层是否带来边际收益。

### 6.2 训练策略

#### 6.2.1 统一超参数

所有实验采用严格相同的训练超参数，确保对比公平：


| 参数         | 值               | 说明                              |
| ---------- | --------------- | ------------------------------- |
| 基础权重       | train11/best.pt | 已在目标数据集上充分训练的最优权重               |
| optimizer  | SGD             | 显式指定，确保 lr0 生效                  |
| lr0        | 0.005           | 初始学习率（微调场景用较低值保护已有特征）           |
| lrf        | 0.01            | 最终学习率衰减因子（lr_final = lr0 × lrf） |
| epochs     | 200             | 最大训练轮次                          |
| patience   | 50              | 连续 50 轮无改善则早停                   |
| batch size | 16              | 适配 6GB GPU 显存                   |
| imgsz      | 960             | 输入分辨率（与基线训练一致）                  |
| workers    | 0               | 禁用多进程数据加载（Windows 兼容性）          |


> **optimizer 选择说明**：Ultralytics 默认 `optimizer=auto` 会自动选择 AdamW 并覆盖用户指定的 lr0。为确保各实验使用严格一致的学习率策略，本文显式指定 `optimizer=SGD`。

#### 6.2.2 权重迁移策略

对于引入注意力机制的实验（Exp-1~5），采用以下权重迁移流程：

1. 从注意力 YAML 配置构建新架构（含注意力层）
2. 加载 train11/best.pt 的参数字典
3. 按**参数名 + 张量形状**逐层匹配
4. 兼容层（Backbone/Neck 中与基线相同的层）直接加载预训练权重
5. 不兼容层（新增的注意力模块）保持 PyTorch 默认随机初始化

各实验的权重迁移情况：


| 实验           | 新增层数 | 匹配加载的层                   | 随机初始化的层  |
| ------------ | ---- | ------------------------ | -------- |
| baseline     | 0    | 全部 (417/417)             | 0        |
| 位置 A 实验 (×3) | 1    | Backbone 全部 + Head 全部    | 注意力层 1 个 |
| 位置 C 实验 (×2) | 4    | Backbone 全部 + Head C2f 层 | 注意力层 4 个 |


#### 6.2.3 训练时间预估

基于实测数据（NVIDIA RTX 3060 Laptop 6GB, batch=16, imgsz=960, 330 张训练图/99 张验证图）：


| 阶段             | 批次数  | 每批耗时  | 每 epoch 耗时       |
| -------------- | ---- | ----- | ---------------- |
| 训练（330 张 / 16） | 21 批 | ~5.5s | ~2.0 min         |
| 验证（99 张 / 16）  | 7 批  | ~5s   | ~0.6 min         |
| **单 epoch 合计** |      |       | **~2.5-3.0 min** |



| 收敛场景             | 估计 epoch 数 | 单实验耗时   | 6 个实验总计 |
| ---------------- | ---------- | ------- | ------- |
| 早停触发（~50 epoch）  | 50         | ~2.5 小时 | ~15 小时  |
| 中等收敛（~100 epoch） | 100        | ~5 小时   | ~30 小时  |
| 跑满 200 epoch     | 200        | ~10 小时  | ~60 小时  |


由于基于已充分训练的 train11 模型微调，预期大部分实验在 50-80 epoch 内早停，**单实验约 2-4 小时，全部 6 个实验约 15-25 小时**。

#### 6.2.4 内存管理

在 6GB 显存的 GPU 上进行连续实验训练，需要注意内存管理。训练脚本在实验之间执行以下清理操作：

1. 删除模型对象（`del model`）释放 GPU 显存
2. 调用 `gc.collect()` 强制 Python 垃圾回收
3. 调用 `torch.cuda.empty_cache()` 释放 CUDA 缓存
4. 单个实验失败不中断后续实验（try-except 容错）

### 6.3 评估指标


| 指标                                  | 说明                      | 重要程度       |
| ----------------------------------- | ----------------------- | ---------- |
| mask [mAP@0.5](mailto:mAP@0.5)      | IoU 阈值 0.5 的掩码平均精度      | ★★★★★ 核心指标 |
| mask [mAP@0.5](mailto:mAP@0.5):0.95 | IoU 阈值 0.5-0.95 的掩码平均精度 | ★★★★★ 核心指标 |
| box [mAP@0.5](mailto:mAP@0.5)       | 检测框平均精度                 | ★★★☆☆ 辅助参考 |
| box [mAP@0.5](mailto:mAP@0.5):0.95  | 检测框平均精度（严格）             | ★★★☆☆ 辅助参考 |
| Parameters                          | 模型参数量                   | ★★★★☆ 部署约束 |
| Inference (ms)                      | 单帧推理时间                  | ★★★★☆ 部署约束 |


### 6.4 消融实验逻辑

实验设计包含以下两个消融维度：

**维度一：注意力机制类型消融**（Exp-1 vs Exp-2 vs Exp-3，相同位置 A）

在相同的插入位置（Backbone 末尾）对比三种不同注意力机制的效果差异：


| 对比组                               | 对比内容             | 验证假设          |
| --------------------------------- | ---------------- | ------------- |
| Exp-1 (CBAM) vs Exp-0 (baseline)  | 通道+空间注意力 vs 无注意力 | 注意力机制能否提升分割精度 |
| Exp-2 (SimAM) vs Exp-0 (baseline) | 零参数注意力 vs 无注意力   | 排除额外参数的容量因素   |
| Exp-3 (CA) vs Exp-1 (CBAM)        | 位置编码 vs 通道+空间    | 位置信息对分割边缘的增益  |
| Exp-2 (SimAM) vs Exp-1 (CBAM)     | 0 参数 vs 65K 参数   | 参数量与效果是否正相关   |


**维度二：插入位置消融**（相同机制，位置 A vs 位置 C）


| 对比组                              | 对比内容                         | 验证假设            |
| -------------------------------- | ---------------------------- | --------------- |
| Exp-4 (cbam-c) vs Exp-1 (cbam-a) | CBAM: Neck 4处 vs Backbone 1处 | 多尺度增强是否优于单点增强   |
| Exp-5 (ca-c) vs Exp-3 (ca-a)     | CA: Neck 4处 vs Backbone 1处   | 位置编码在多尺度下的边际收益  |
| Exp-4 (cbam-c) vs Exp-5 (ca-c)   | CBAM vs CA 在位置 C             | 哪种机制更适合 Neck 增强 |


### 6.5 预期结果对比表格

训练完成后，使用评估脚本（`cmd/yolov8-seg-attention-val.py`）自动生成以下对比表格：

#### 表 1：各实验核心指标对比


| 实验       | 注意力      | 位置  | 参数量   | 增量               | mask mAP@50 | mask mAP@50-95 | 提升  | box mAP@50 | 推理(ms) |
| -------- | -------- | --- | ----- | ---------------- | ----------- | -------------- | --- | ---------- | ------ |
| baseline | 无        | -   | 3.41M | -                | -           | -              | 基线  | -          | -      |
| cbam-a   | CBAM     | A   | 3.48M | +65.9K (+1.93%)  | -           | -              | -   | -          | -      |
| simam-a  | SimAM    | A   | 3.41M | +0 (+0.00%)      | -           | -              | -   | -          | -      |
| ca-a     | CoordAtt | A   | 3.42M | +6.2K (+0.18%)   | -           | -              | -   | -          | -      |
| cbam-c   | CBAM     | C   | 3.51M | +103.4K (+3.03%) | -           | -              | -   | -          | -      |
| ca-c     | CoordAtt | C   | 3.42M | +13.9K (+0.41%)  | -           | -              | -   | -          | -      |


> 注："-"表示待填入实验结果。mask mAP@50-95 为最核心评价指标。

#### 表 2：消融分析——注意力类型对比（位置 A）


| 机制          | 注意力维度   | 额外参数  | mask mAP@50-95 | 相对基线提升 |
| ----------- | ------- | ----- | -------------- | ------ |
| 无（baseline） | -       | 0     | -              | -      |
| CBAM        | 通道 + 空间 | 65.9K | -              | -      |
| SimAM       | 3D（无参数） | 0     | -              | -      |
| CoordAtt    | 通道 + 位置 | 6.2K  | -              | -      |


#### 表 3：消融分析——插入位置对比


| 机制       | 位置 A (1 层) | 位置 C (4 层) | 位置 C 提升幅度 |
| -------- | ---------- | ---------- | --------- |
| CBAM     | -          | -          | -         |
| CoordAtt | -          | -          | -         |


---

## 7. 实验工具链与执行方式

### 7.1 训练脚本

训练脚本 `cmd/yolov8-seg-attention-train.py` 统一管理所有 6 组实验，支持三种执行方式：

**方式一：交互式选择**

```bash
python cmd/yolov8-seg-attention-train.py
```

脚本列出所有可用实验，用户选择编号或名称执行。

**方式二：指定单个实验**

```bash
python cmd/yolov8-seg-attention-train.py --exp cbam-a
```

**方式三：批量执行全部实验**

```bash
python cmd/yolov8-seg-attention-train.py --exp all
```

按 baseline → cbam-a → simam-a → ca-a → cbam-c → ca-c 顺序依次执行，实验间自动清理内存。单个实验失败不影响后续实验。

**自定义参数覆盖**

```bash
python cmd/yolov8-seg-attention-train.py --exp cbam-a \
    --base-model ./runs/segment/train11/weights/best.pt \
    --lr0 0.003 --epochs 150 --batch 8 --imgsz 1280
```

### 7.2 评估脚本

评估脚本 `cmd/yolov8-seg-attention-val.py` 自动扫描 `runs/segment/attn_*` 下的训练结果，对所有实验进行统一评估并输出对比表格：

```bash
python cmd/yolov8-seg-attention-val.py              # 使用 val 集
python cmd/yolov8-seg-attention-val.py --split test  # 使用 test 集
```

输出内容包括：

- 各实验的 mask/box mAP 指标
- 参数量增量和增幅
- 推理速度对比
- 自动标注最优方案及相对基线提升幅度

### 7.3 训练产出目录

```
runs/segment/
├── attn_baseline/       # Exp-0 对照组
│   ├── weights/
│   │   ├── best.pt      # 最优权重
│   │   └── last.pt      # 最后一轮权重
│   ├── results.csv      # 逐 epoch 指标记录
│   └── results.png      # 训练曲线图
├── attn_cbam-a/         # Exp-1 CBAM + 位置A
├── attn_simam-a/        # Exp-2 SimAM + 位置A
├── attn_ca-a/           # Exp-3 CoordAtt + 位置A
├── attn_cbam-c/         # Exp-4 CBAM + 位置C
└── attn_ca-c/           # Exp-5 CoordAtt + 位置C
```

---

## 8. 改造前后参数量对比

以下为 yolov8n-seg（scale=n, width_multiple=0.25）实际验证的参数量统计：


| 配置                     | 总参数量      | 增量       | 增幅     |
| ---------------------- | --------- | -------- | ------ |
| baseline (yolov8n-seg) | 3,409,968 | -        | -      |
| CBAM + 位置A             | 3,475,858 | +65,890  | +1.93% |
| SimAM + 位置A            | 3,409,968 | +0       | +0.00% |
| CoordAtt + 位置A         | 3,416,128 | +6,160   | +0.18% |
| CBAM + 位置C             | 3,513,336 | +103,368 | +3.03% |
| CoordAtt + 位置C         | 3,423,856 | +13,888  | +0.41% |


所有注意力方案的参数增量均控制在 **3.1% 以内**，符合轻量化设计原则。

---

## 附录：文件变更清单


| 文件                                                  | 变更类型 | 说明                                                |
| --------------------------------------------------- | ---- | ------------------------------------------------- |
| `ultralytics/nn/modules/conv.py`                    | 修改   | 新增 SEAttention, ECAAttention, SimAM, CoordAtt 四个类 |
| `ultralytics/nn/modules/__init__.py`                | 修改   | 导出新增模块                                            |
| `ultralytics/nn/tasks.py`                           | 修改   | import 新模块 + parse_model 通道适配                     |
| `ultralytics/cfg/models/v8/yolov8-seg-cbam-a.yaml`  | 新增   | CBAM + 位置A 架构配置                                   |
| `ultralytics/cfg/models/v8/yolov8-seg-simam-a.yaml` | 新增   | SimAM + 位置A 架构配置                                  |
| `ultralytics/cfg/models/v8/yolov8-seg-ca-a.yaml`    | 新增   | CoordAtt + 位置A 架构配置                               |
| `ultralytics/cfg/models/v8/yolov8-seg-cbam-c.yaml`  | 新增   | CBAM + 位置C 架构配置                                   |
| `ultralytics/cfg/models/v8/yolov8-seg-ca-c.yaml`    | 新增   | CoordAtt + 位置C 架构配置                               |
| `cmd/yolov8-seg-attention-train.py`                 | 新增   | 注意力实验训练脚本                                         |
| `cmd/yolov8-seg-attention-val.py`                   | 新增   | 注意力实验对比评估脚本                                       |


---

## 参考文献

1. Woo S, Park J, Lee J Y, et al. CBAM: Convolutional block attention module[C]//European Conference on Computer Vision (ECCV). 2018: 3-19.
2. Yang L, Zhang R Y, Li L, et al. SimAM: A simple, parameter-free attention module for convolutional neural networks[C]//International Conference on Machine Learning (ICML). 2021: 11863-11874.
3. Hou Q, Zhou D, Feng J. Coordinate attention for efficient mobile network design[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2021: 13713-13722.
4. Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018: 7132-7141.
5. Wang Q, Wu B, Zhu P, et al. ECA-Net: Efficient channel attention for deep convolutional neural networks[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2020: 11534-11542.


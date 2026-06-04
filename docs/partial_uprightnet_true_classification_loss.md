# 残缺点云 upright estimation 的真正分类式损失设计

## 1. 设计修正

之前把 upright direction 离散成球面 anchor，再用 cross-entropy 分类。这个方案不应该作为主线，因为它本质上仍然是在逼近一个连续方向变量：

```text
连续方向 y in S^2
-> 离散成 K 个方向 bin
-> 用 CE 训练
-> 推理时再把 bin 还原成方向
```

这只是伪分类。类别没有真实语义，类别之间仍然由角距离组织，本质仍是方向回归的离散近似。

我们需要的“真正分类”应该是语义分类，而不是空间离散化。核心问题应改写为：

```text
给定一个残缺点云 P 和一个候选 upright hypothesis h，
判断 h 是否是该物体的真实直立解释。
```

也就是训练一个 uprightness classifier：

```text
C(P, h) -> probability that h is a valid upright explanation
```

这里分类标签是：

```text
valid upright / invalid upright
```

这是一个真实二分类任务，不是把方向空间切 bin。

## 2. 为什么这才是分类

### 2.1 类别有语义

二分类标签不是“第 37 个方向 anchor”，而是：

```text
1: 这个候选姿态把物体解释为合理直立状态
0: 这个候选姿态会导致倒置、侧翻、错误支撑或不稳定解释
```

类别含义和物理/语义状态直接相关，因此是严格意义上的分类。

### 2.2 候选方向不是类别

候选 `h` 是分类器的输入条件，不是类别编号。它可以来自：

```text
PCA 轴
可见平面法向
原 UprightNet/RANSAC 候选
随机负样本方向
180° flip hard negative
局部可见结构生成的候选
```

这些候选可以是连续的、动态生成的、每个样本不同的。训练目标始终只是判断：

```text
这个候选对不对？
```

因此它不是空间离散化分类。

### 2.3 推理可以用搜索，但训练不是回归

推理时可以生成多个候选并打分：

```text
h* = argmax_h C(P, h)
```

这一步可能需要候选采样或局部优化，但这不是训练损失的性质。训练损失仍然是二分类 CE / focal BCE / pairwise classification。

## 3. 输入表示

推荐不要直接把候选方向 `h` 作为一个裸 3D 向量拼到特征里。更好的做法是把点云规范化到候选姿态坐标系中：

```text
P_h = AlignCandidateUpToY(P, h)
```

也就是将候选 upright 方向 `h` 旋到 canonical `+Y`，然后让分类器判断：

```text
P_h 看起来像一个真实直立物体吗？
```

这样分类器学到的是“候选姿态下的物体状态”：

```text
底部是否像支撑区域
重心是否在支撑上方
可见结构是否符合上下语义
是否像倒置物体
是否像侧翻物体
是否只有不可判别的残缺外壳
```

这比直接回归 `h` 更符合残缺点云问题。

## 4. 训练样本构造

对每个原始样本，有 GT upright direction `y`。构造若干候选 `h`，并给每个候选一个分类标签。

### 4.1 正样本

正样本候选：

```text
h_pos = y
```

也可以加入小扰动正样本，提高分类器容忍度：

```text
angle(h, y) <= tau_pos
```

建议：

```text
tau_pos = 5° or 10°
```

标签：

```text
label = 1
```

### 4.2 负样本

负样本必须包含 hard negatives，而不是只采随机方向。

推荐负样本类型：

| 类型 | 构造 | 目的 |
|---|---|---|
| Flip negative | `h = -y` | 专门打击 180° flip |
| Tilt negative | `angle(h, y) in [30°, 90°]` | 学侧翻/倾斜 |
| Random negative | 随机 SO(3) 方向 | 覆盖普通错误 |
| PCA negative | 使用 PCA 主轴/次轴中的错误方向 | 模拟对称物体误导 |
| Plane negative | 使用可见局部平面法向 | 模拟错误支撑面 |
| RANSAC negative | 原 UprightNet 在 partial 上的错误输出 | 直接针对失败模式 |

标签：

```text
label = 0
```

### 4.3 残缺样本的关键 hard negative

我们的实验表明，partial 数据上的主要失败是：

```text
support mask 为空
或者 support mask 落到错误可见平面
```

因此 hard negative 应该优先来自“看起来像支撑面但实际上不是支撑面”的候选。例如：

```text
桌面上方可见平面
椅背平面
灯罩边缘
瓶口/杯口
床垫上表面
桌面板上表面
```

这些候选比纯随机负样本重要得多。

## 5. 主损失

### 5.1 Binary uprightness classification

分类器输出：

```text
s = C(P, h)
```

其中 `s` 是 raw logit。

标签：

```text
z = 1 if h is a valid upright hypothesis else 0
```

主损失：

```text
L_upright_bce = BCEWithLogits(s, z)
```

这是第一版最核心的损失。

### 5.2 Focal BCE

由于 easy negative 很多，建议使用 focal BCE 强化 hard negative：

```text
p = sigmoid(s)
p_t = p       if z = 1
      1 - p   if z = 0

L_focal = - alpha_t * (1 - p_t)^gamma * log(p_t)
```

建议初始值：

```text
gamma = 2
alpha_pos = 0.5
alpha_neg = 0.5
```

如果 flip negative 仍然难，可以给 flip negative 单独更高权重。

### 5.3 Pairwise classification

也可以把训练写成“正候选是否比负候选更像 upright”的二分类：

```text
s_pos = C(P, h_pos)
s_neg = C(P, h_neg)
L_pair = BCEWithLogits(s_pos - s_neg, 1)
```

等价于：

```text
L_pair = -log sigmoid(s_pos - s_neg)
```

这个损失的优点是只要求模型比较两个候选，不要求 score 绝对校准。对候选生成质量不稳定的早期阶段，它可能比独立 BCE 更稳。

推荐第一版同时记录两种配置：

```text
A1: independent BCE
A2: pairwise BCE
```

## 6. 辅助分类任务

### 6.1 Visibility state classification

残缺点云下，分类器还应判断当前视角是否真的看到底部支撑证据：

```text
visibility_state in {support_visible, support_weak, support_hidden}
```

损失：

```text
L_vis = CE(logits_vis, visibility_state)
```

这个任务不是预测方向，而是预测证据状态。它是一个真正的语义分类任务。

### 6.2 Candidate error type classification

对负样本，可以进一步预测错误类型：

```text
error_type in {flip, tilt, wrong_plane, random, ambiguous}
```

损失：

```text
L_type = CE(logits_type, error_type)
```

这个辅助任务能迫使模型区分不同失败机制。例如 `flip` 和 `wrong_plane` 在角度上可能都很大，但语义原因不同。

### 6.3 Point support classification

point-wise support BCE 可以保留，但它不是主线：

```text
L_sup = BCEWithLogits(support_logits_i, support_label_i)
```

它的作用是提供局部支撑语义，不再作为最终 upright 的唯一来源。

## 7. 推荐总损失

第一版推荐：

```text
L_total =
    1.0 * L_upright_bce
  + 0.3 * L_vis
  + 0.2 * L_type
  + 0.1 * L_sup
```

如果使用 pairwise classification：

```text
L_total =
    1.0 * L_pair
  + 0.3 * L_vis
  + 0.2 * L_type
  + 0.1 * L_sup
```

这里没有任何方向回归项，没有 geodesic loss，没有 vMF NLL，也没有球面 anchor CE。

## 8. 模型结构建议

### 8.1 Candidate-conditioned classifier

模型输入：

```text
P_h: candidate-normalized point cloud
optional candidate metadata: source type, visibility stats
```

模型输出：

```text
upright_logit: scalar
visibility_logits: 3 classes
error_type_logits: K classes
support_logits: per-point optional
```

核心结构：

```text
P, h
  -> AlignCandidateUpToY(P, h)
  -> PointNet/DGCNN/Equivariant trunk
  -> global feature
  -> uprightness classifier
```

如果使用 DGCNN trunk，可以复用现有 UprightNet backbone，但 head 必须改为分类 head，而不是输出 3D direction。

### 8.2 为什么候选归一化很重要

如果直接把 `h` 拼到全局特征里，模型仍可能学成隐式方向回归。把点云旋到候选坐标系后，分类器只需要判断：

```text
这个规范化后的物体是不是 upright？
```

这更接近图像分类中的“这张图是不是正放”的任务，而不是回归角度。

## 9. 推理流程

推理不是一次前向直接输出方向，而是：

```text
1. 从 partial point cloud 生成候选集合 H
2. 对每个 h in H，计算 score C(P, h)
3. 选择 score 最大的候选
4. 若 top scores 接近或 entropy 高，输出不确定性
```

候选集合可以来自：

```text
PCA axes
visible plane normals
原 UprightNet/RANSAC proposal
随机扰动
multi-start local refinement
```

关键点：候选生成不是分类类别。分类器学习的是 `valid / invalid` predicate。

## 10. 当前代码状态

当前代码还没有实现这个真正分类范式。

已有实现：

```text
network_shs.py:
  DecomposedDirectionHead -> axis: (B,3), sign_logit: (B,)

Common/loss_shs.py:
  L_axis = antipodal_geodesic_loss(axis, gt)
  L_sign = BCEWithLogits(sign_logit, sign_gt)
  L_sup  = BCE
  L_stab = ReLU stability
```

这只能算：

```text
连续轴回归 + 符号二分类
```

其中 `L_sign` 是真正的二分类，但 `L_axis` 仍然是回归。因此它不满足“主损失必须是真分类”的要求。

也没有看到：

```text
candidate-conditioned uprightness classifier
positive/negative candidate generation
flip / wrong-plane / random hard negative labels
visibility state classification training path
pairwise BCE ranking loss
```

所以需要新增模型、数据集采样器和损失。

## 11. 最小实现目标

第一阶段只做最小可行版本：

```text
Dataset:
  每个原始/残缺点云生成 1 个正候选 + N 个负候选

Model:
  CandidateUprightClassifier(P, h) -> upright_logit

Loss:
  BCEWithLogits(upright_logit, valid_label)

Negatives:
  -h_gt
  random direction
  PCA wrong axis
  original UprightNet failed proposal
```

只要这个版本能在 partial 数据上把 `flip rate` 明显压下去，就说明“真正分类”路线成立。

# 在 UprightNet 上做等变性创新——候选故事线

> 约束：保留 "E(3)-等变点云网络估计直立方向" 这条主线，基于现有 e3nn/MACE 栈做出研究意义上的新贡献。
> 目标：给出 4-5 个可独立成文的创新点，每个都有明确 novelty gap、实现路径、实验故事。

---

## 核心观察：当前实现的两个"空位"正好是好故事的起点

1. **置信度单峰假设**：vMF / Watson 都是 S² 上的**单峰**分布。对圆柱、花瓶、哑铃、轮子这类**旋转对称物体**，存在多个等价合理的"直立方向"，单峰分布被迫在它们之间平均 ⇒ 大角度错误。数据集里这类物体约占 20-30%，是当前方法的系统性 failure mode。

2. **几何先验与等变骨干解耦**：`weighted_plane_normal`（可微加权 SVD 求平面法向）已经写在 `Common/geometric_utils.py:118-168`，**但从未被调用**。原 UprightNet 的物理归纳偏置（支撑面法向 = 重力方向）完全没接入到等变模型的方向输出上——`SupportHead` 和 `VMFDirectionHead` 是两条互不相关的分支。

这两个空位，各自对应一个独立可发表的创新方向，也可以**合并成一个"多峰 + 物理先验"的大故事**。

---

## 创新点 1：Equivariant Implicit Density on S² ——多峰等变方向分布

### 一句话 pitch
把 Implicit-PDF（Murphy ICML 2021，针对 SO(3) 从图像）搬到 S²，**并用等变特征参数化**，让网络天然表达对称物体的多模态直立方向分布。

### Novelty gap
- Implicit-PDF 原作：SO(3) 上 + 图像输入 + **非等变** backbone；
- 本任务：S² 上 + 点云输入 + **E(3)-等变** backbone；
- 据我所知**没有工作把 implicit density 放在等变特征之上**。等变约束会让 implicit field 天然对称、训练更稳定。

### 数学骨架
- 输入：等变 backbone 的 l=0 和 l=1 全局特征 `f = (s ∈ R^C, v_1, ..., v_K ∈ R^3)`（`s` 不变，`v_i` 共变）。
- 对任意 query 方向 `u ∈ S²`，构造**不变**标量集合 `{s, ⟨u, v_1⟩, ..., ⟨u, v_K⟩, ⟨u, v_i⟩⟨u, v_j⟩, ...}`——前两项已经是完备的 l=0/l=1 标量化方式，外积项可选加入高阶项。
- 把这组不变标量送进 MLP → 输出 `log p(u | PC)`。
- 等变性：`log p(Ru | R·PC) = log p(u | PC)` 自动成立（因为所有输入标量都是 R-不变的）。
- 训练损失：`-log p(u_gt | PC) + log ∫_{S²} exp(log p(u|PC)) du`，积分通过 HEALPix / Fibonacci 格点 128~512 个采样近似；推理时在同一格点上取 argmax 或前 K 峰。

### 需要改什么代码
- 新加一个 `ImplicitS2DensityHead`：输入 `(node_features, batch, query_directions)`，输出 `log p(query)`。
- 训练时每个样本再采样 N_neg=64 个 query 方向估算归一化常数。
- 现有 `VMFDirectionHead` 退化为**它的单峰特例**（`log p(u) = κ·⟨u, μ⟩ + const`），可以作为 warm-start。

### 实验故事
- 按类别拆分 error：对称类（花瓶、杯、圆桌、哑铃、轮子）vs 非对称类（椅子、飞机、汽车）。
- 现有 vMF 在对称类上 mean error > 45°；新方法应显著下降，同时在非对称类保持或略优。
- Ablation：`K=1`（退化到 vMF）、`K=2`、`K=4`，看内部 l=1 通道数对多峰表达力的影响。
- 可视化：对哑铃、轮子画 S² 上的 `p(u)` heatmap，展示两极双峰。

### 难度 / 风险
中等。归一化常数估计是主要数值难点（可以借鉴 Implicit-PDF 的做法）。可发表性强——"equivariant + multimodal" 是两个干净的卖点相乘。

---

## 创新点 2：End-to-End 等变可微分支撑平面回归（ECSP）

### 一句话 pitch
把 Upright-Net 的 "支撑点 → RANSAC 平面 → 法向" 流程全部重写成 **E(3)-等变 + 端到端可微** 版本，让 Upright-Net 的物理归纳偏置与等变网络无缝融合。

### Novelty gap
- Upright-Net 原方法：RANSAC 不可微，平面拟合硬阈值，对非平面底不友好；
- 现有等变实现：直接全局池化 l=1 输出 mu，**完全不用**支撑面信息（SupportHead 是独立分支）；
- 真空：**没有工作同时做到"等变 + 可微 + 保留 Upright-Net 物理先验"**。

### 架构骨架
```
等变 backbone → node_features
  ├── SupportHead   → soft_weights ∈ [0,1]^N     (l=0 不变)
  └── VMFDirectionHead → mu_vMF, κ_vMF           (l=1 + l=0)

weighted_plane_normal(points, soft_weights)       (已有实现)
  → normal_SVD ∈ R³                              (l=1, 严格等变)

Losses:
  L_vMF    = vMF_NLL(mu_vMF, κ_vMF, y_up)
  L_SVD    = geodesic(normal_SVD, y_up)
  L_consist = geodesic(mu_vMF, normal_SVD)       ← 新
  L_sup     = BCE(soft_weights, y_support)       ← 可选，弱监督
  L_total   = L_vMF + α·L_SVD + β·L_consist + γ·L_sup
```

### 进一步的"纯语义"变体（更强的创新点）
**去掉 y_support 监督**，把 `soft_weights` 当成完全**从方向一致性损失倒逼出来**的 emergent 表示：让网络自己学到"哪些点对直立方向最有信息量"，不需要人工标注 `y < 0.05` 的伪标签。这一变体消除了 Upright-Net 对"点云已经大致对齐"的隐含假设——ModelNet 对齐良好才能定义支撑点，真实扫描没有这种特权。

### 需要改什么代码
- 调用 `weighted_plane_normal`（现成），把输出作为第二路 l=1 输出；
- 在 `EquivariantLoss` 里加 `L_SVD` 和 `L_consist`；
- 对 torch.linalg.eigh 的梯度在 cov 退化（点云共面）时会爆炸——加 `eps * I` 正则（现有实现已有）。

### 实验故事
- 消融：纯 L_vMF / 纯 L_SVD / 两者 / 两者 + L_consist。
- Robustness 实验：对真实扫描 partial / noisy 数据（ScanObjectNN），展示物理先验路径（L_SVD）比纯回归（L_vMF）更鲁棒。
- Interpretability：可视化 `soft_weights` 热力图，证明网络在无标签情况下也能自发识别支撑区域。
- 去 y_support 监督的变体可以单独作为 "self-emergent support learning" 卖点。

### 难度 / 风险
低。代码大部分已经写好（`weighted_plane_normal` 现成、`SupportHead` 现成），只需要接线和调参。**建议先做这个**——一两天就能出第一版结果。

---

## 创新点 3：Partial / Real-world 点云的等变直立估计

### 一句话 pitch
把战场从 "ModelNet 完整点云 + 合成 SO(3) 旋转" 搬到 **ScanObjectNN / ScanNet / 真实 LiDAR 扫描**，论证等变性的**真正价值不是"省掉数据增广"，而是 "partial / noisy 数据下的 OOD 鲁棒性"**。

### Novelty gap
- 几乎所有等变点云论文（TFN, VN, EPN, 本仓库）都在完整合成数据上评测；
- 原 UprightNet 也只在 ModelNet 上评测；
- 真实场景中 upright 估计最有用的就是 "仅看到部分扫描" 时。等变网络天然该在这里赢，但**没有工作系统评测**。

### 数据集选型
- **ScanObjectNN**（Uy ICCV 2019）：15 类真实扫描物体，含部分遮挡、背景噪声；和 UprightNet15 类别对齐方便。
- **OmniObject3D / Ocullar-scan**：更多类别和真实 noise profile。
- **SUN RGB-D / ScanNet 物体级 crop**：真正的 LiDAR-like partial scan。

### 实验故事
- Train on ModelNet15 → Test on ScanObjectNN15（OOD）：原 UprightNet 崩，等变网络保住；
- 分析：partial 扫描时支撑面往往缺失，vMF κ 自然降低——展示 κ 和真实 OOD 严重度的相关。
- 可以和 Idea 2 联动：partial 扫描下物理先验（支撑平面）缺失时 SVD 分支退化，vMF 分支兜底，`L_consist` 自动成为"两条分支谁更可信"的门控。

### 难度 / 风险
中。主要是数据对齐和 per-class GT up 的获取——ScanObjectNN 每个类的 canonical up 需要人工确认。

---

## 创新点 4：自监督等变直立估计（物理先验代替标签）

### 一句话 pitch
用 **"质心应在支撑面上方 + 重力方向在 Hessian 最小方向附近"** 作为**自监督信号**，训一个 E(3)-等变网络输出 up 方向，完全不需要人工 up 标签。

### Novelty gap
- Upright-Net 需要逐物体的 up 标签 + 支撑点标签；
- 现有自监督 canonicalization（Canonical Capsules, Compass）学的是"某个一致姿态"，但不一定是**语义上正确的 up**；
- **没人**用显式物理先验（静态稳定性）来做 upright 自监督。

### 损失构造
- **稳定性损失**：对当前预测的 up 方向 `u`，找投影到垂直于 `u` 的平面上的 **2D 凸包**；要求质心投影在凸包内部（可微实现：soft minimum distance to convex hull edges）。
- **势能最低损失**：在 `u` 方向下，重心高度 = `⟨u, c⟩` 最小；而其他方向 `u'` 的重心高度高。用对比损失：`L = -log σ(⟨u, c⟩_min - ⟨u', c⟩)` 在随机 `u'` 上。
- **等变一致性损失**：对同一物体的两个随机旋转 `R₁·PC` 和 `R₂·PC`，输出应满足 `u(R₁·PC) = R₁·R₂⁻¹·u(R₂·PC)`（等变架构自动保证，这条主要是 sanity check）。

### 实验故事
- 完全无 up 标签训练，和有标签的 Upright-Net 对比。
- Ablation：只用稳定性 / 只用势能最低 / 两者。
- 跨数据集：在 ShapeNet 上自监督训，在 ModelNet / ScanObjectNN 测零样本 up。

### 难度 / 风险
高。物理损失的可微实现（凸包投影的可微版本）需要仔细设计，训练可能陷入平凡解（比如输出 `u = -mass_center` 就能让"势能最低"成立）。但一旦跑通，创新点非常强。

---

## 创新点 5：预训练非等变骨干 + 微型等变 canonicalization 头

### 一句话 pitch
沿 **Kaba ICML 2023 / Mondal NeurIPS 2023** 的思路：用 Point-MAE / PointNeXt / PointTransformerV3 的**冻结**预训练表示，只在前端加一个**小的 E(3)-等变 canonicalization 头**，输出旋转 R，把点云正位后喂给预训练骨干。

### Novelty gap
- Mondal NeurIPS 2023 只做了图像；
- 点云上的 "pretrained + equivariant adapter" 没人做；
- 本任务恰好可以把"canonicalization 头的输出旋转 `R[:, 1]`"直接作为 up 方向。

### 好处
- 训练成本极低（冻结骨干，只训数 M 参数的等变头）；
- 自然受益于大预训练；
- 可以复用任何未来的点云 foundation model。

### 难度 / 风险
中。需要 Point-MAE / PointTransformerV3 的 checkpoint，工程链路较长。

---

## 推荐路径

### 阶段 1（2 周）：Idea 2 作为**安全的第一篇**
`weighted_plane_normal` 接回前向，跑 L_vMF + L_SVD + L_consist 消融——这套东西**一定有结果**，消融图清楚，故事直白（"我们把 Upright-Net 可微化 + 等变化，并且证明 support 信号可以 emergent"）。两周内应该能出 workshop 级别结果。

### 阶段 2（1-2 个月）：Idea 1 叠加成**主刊故事**
在 Idea 2 的基础上把 vMF 头换成 Equivariant Implicit Density Head。新 selling point："等变 + 多峰 + 可微物理先验"，对对称物体系统性改善。可以冲 CVPR/ICCV。

### 阶段 3（看情况）：Idea 3 作为**实用性章节**
把 Idea 1+2 的模型在 ScanObjectNN partial 数据上评，把 "equivariance 的真正价值" 这条 narrative 写实。这会让整篇论文从"又一个等变架构"升级为"在真实场景显著有用的等变架构"。

### 不推荐做第一篇
- Idea 4（自监督物理先验）：创新点最强但实现风险最高，适合作为 follow-up 论文。
- Idea 5（预训练 + canonicalization 适配）：更像工程整合，novelty 相对弱，适合 short paper。

---

## 一句话总结

**最稳的第一步是 Idea 2**（端到端等变可微支撑平面），一两天就能接通代码，消融清晰；**最有学术野心的合并路线是 Idea 2 + Idea 1**（可微支撑 + 多峰等变密度），真正能解决当前方法的两个系统性失效模式，是"等变 + upright 估计"方向上有原创贡献的论文骨架。

# UprightNet 代码库调研与替代方案分析

> 范围：仓库 `/home/yujian_shi/uprightnet` 的全部训练 / 测试 / 模型代码。
> 目标：(1) 梳理这个代码库到底在做什么；(2) 评价当前架构的每一个设计决策；(3) 指出可能更好 / 更简单 / 更现代的做法。
> 配套文档：`docs/upright_direction_literature_review.md`（按任务维度的文献综述，本文最后整合）。

---

## 目录

1. 任务定义
2. 数据与标签
3. 原始基线（UprightNet / CVPR 2022）
4. 新实现：E(3)-等变网络
5. 损失函数
6. 训练与评估
7. 设计评价（逐点）
8. 替代方案对比
9. 具体改造建议

---

## 1. 任务定义

仓库解决单一任务：**给定一个 SO(3) 随机旋转后的 3D 点云，估计其"直立方向 / 重力向上方向" ∈ S²。**

- 输入：`(N, 3)` 点云，N = 2048。
- 输出：`mu ∈ R³`（单位向量），可选 `kappa > 0`（vMF 置信度）。
- 评估指标：`angular_error_deg`（度）、Acc@5 / Acc@10（预测与 GT 夹角小于 5° / 10° 的比例），见 `model.py:499`、`model.py:506-508`。
- Antipodal 对称：评估和损失里用 `|cos|`（`Common/geometric_utils.py:37`、`Common/loss_equivariant.py:47`、`:94`），**即训练时视 n 与 -n 等价**。

> **观察 1**：原始 Upright-Net（CVPR 2022）的设定是 "有向" 的（支撑面法向朝远离质心），但本仓库在训练 / 评估环节统一使用 antipodal，二者语义不完全一致（详见 §7）。

---

## 2. 数据与标签

### 2.1 两个数据路径并存

- **原始基线路径**（`Common/RobustPointSetDataLoader.py`）：需要预生成的 `{train,test}_original.npy`、`{train,test}_rotation.npy`、`rotm_*.npy`、`pid_*.npy`、`{train,test}_d.npy`。原论文对每个物体离线生成 100 次随机旋转（README 注明）。
- **等变路径**（`Common/EquivariantDataLoader.py`）：只加载 `*_noaug_original.npy` + `labels_*_noaug.npy` + `pid_*_noaug.npy`；旋转**在线**生成（`Rotation.random()`），每次 `get()` 调用一个新旋转矩阵 R，GT 方向取 `R[:, 1]`（因为标准姿态下"up = y 轴"）。

### 2.2 支撑点标签

`pid` 是每点是否属于"底面支撑点"的二分类标签；README 说法是 ModelNet40 对齐后 `y < 0.05` 视为支撑点。本仓库把它作为**辅助损失**（BCE），通过 `--beta` 调权（默认 0.5）。

### 2.3 数据规模

- 训练集 1110 个对象，测试集 370 个对象（UprightNet15，15 类 ModelNet 子集）。
- 等变路径因在线旋转，**实际训练轨迹等价于"无限增广"**（每个 epoch 1110 样本，每次重随机 R）。

> **观察 2**：原始路径依赖离线 100 倍增广文件，磁盘成本 100×；等变路径彻底不再需要增广预处理——这是等变方法的一项实际工程收益，但前提是模型**真正严格等变**（在线旋转 + 等变架构 ⇒ 任何 epoch 见过的有效"朝向分布"都是 SO(3) 均匀）。

---

## 3. 原始基线（UprightNet / CVPR 2022）

### 3.1 网络结构 `network.py`

```
input: (B, 3, N)
  └─ EdgeConv(k=20) ×3，通道 3→32→64→128        # DGCNN 式
  └─ SelfAttention ×4（每层 128d）
  └─ Conv1d 128*4 → 1024 → global max pool
  └─ 拼接 (x_a, x_b, x_c, x_global) → Conv1d → 1
  └─ sigmoid → 每点支撑点概率 (B, 1, N)
```

### 3.2 后处理 `model.py:230-262`

对每个样本：
1. 二值化：`pred > 0.5` 选出支撑点；
2. 用 `sklearn.RANSACRegressor`（残差阈值 0.03）在 x/z 平面拟合 `y = a*x + c*z + d`；
3. 法向 `(a, -1, c)` 归一化；
4. **符号消歧**：乘以 `sign(a*cx - cy + c*cz + d)` 让法向指向质心所在半空间；
5. 支撑点不足 3 时退化为 "支撑点质心 → 整体质心" 的连线；支撑点为 0 时直接用 GT 逆旋转的第二列（**这是对 GT 的隐式泄漏**）。

### 3.3 损失 `Common/loss_utils.py`

- `bce_loss`：对支撑点二分类标签的 BCE。
- `fr_loss`：fitting-residual 损失——选出预测为支撑点的点，与真实底面 y 值 `coef_d` 的偏差通过 SmoothL1 惩罚，再归一化到"被选中点数"。

### 3.4 问题

- 两个不可微步骤（二值化、RANSAC）；
- 后处理假设目标类有明确的平面底，对花瓶等圆形底、对动物类效果预期差；
- 数据路径依赖大量离线增广（100× 存储，训练时间原始 README 估算 ~86 小时）；
- 当支撑点识别失败，fallback 会**泄漏 GT 旋转矩阵的第二列**（`model.py:260`），让汇报的"正确率"有结构性偏差。

---

## 4. 新实现：E(3)-等变网络

位置：`network_equivariant.py`（790 行）。依赖：`e3nn`、`torch-cluster` 的 `knn_graph`、`torch-scatter`。

### 4.1 总览

```
Input: (N, 3) 点云
 → knn_graph(k=num_neighbors)                      # 图构建
 → edge_vec = pos[dst] - pos[src]
 → edge_sh = Y^l(edge_vec)  for l ∈ {0..lmax}     # 球谐
 → edge_length_embedded = soft_one_hot_linspace(  # 径向基
      edge_len, basis='smooth_finite')
 → node_features = scatter(edge_sh, dst, 'sum') / √k   # 初始节点特征
 → input_linear:  irreps_sh → irreps_hidden
 → EfficientConvLayer × num_layers
 → VMFDirectionHead   → (mu, kappa)
 → SupportHead        → per-point support logit
```

### 4.2 `EfficientConvLayer`（MACE 风格 depthwise-separable TP）

`network_equivariant.py:331-461` 是重点。一层等变卷积被拆成：

```
linear_up (channel mixing, 节点特征 shared)
depthwise TP (uvu 模式，每路径 1 权重，per-edge MLP 给权重)
scatter → aggregate
linear_down (channel mixing)
+ self-connection (skip)
S² 激活（separable S² activation，iSHT → pointwise SiLU → SHT）
e3nn BatchNorm
```

关键点：

- **depthwise TP 指令构建**（`_build_depthwise_tp_instructions`，`:178-214`）：遍历 `irreps_in × irreps_sh`，按 CG 选择规则枚举输出不可约，过滤到目标 irreps；每条路径用 `"uvu"` 模式 + 1 权重。
- **torch.compile(self.tp)**（`:404`）：对 e3nn 的 TensorProduct 编译，README 称 ~2× 加速。
- **S² 激活**（`SeparableS2Activation`，`:222-328`）：`lmax ≥ 1` 时把高阶系数通过 `ToS2Grid` 投射到球面网格做 SiLU，再 `FromS2Grid` 投回；**l=0 通道直接 SiLU**（fast path）。思路来自 EquiformerV2（Liao ICLR 2024）。
- **e3nn BatchNorm**（`:419`）：对不同 l 分别做等变归一化。

### 4.3 `EquivariantConvLayer`（FCTP 变体）

`:47-170` 是对比 baseline：用 `o3.FullyConnectedTensorProduct` 直接耦合。所有通道全连接，权重矩阵巨大（~3456 weights/edge for L=2）。训练成本远高于 depthwise，用于消融。

### 4.4 `VMFDirectionHead`

`:469-525`：

1. `direction_proj: irreps_hidden → 1x1o`（投到 l=1 向量）；
2. `kappa_proj: irreps_hidden → 1x0e`（投到 l=0 标量）；
3. 全图均值池化（按 `batch` index）；
4. `mu = normalize(vec_global)`；
5. `kappa = softplus(scalar_global * 0.1 + kappa_bias).clamp(0.1, 500)`。

> **观察 3**：mu 来自 l=1 ⇒ 自然对旋转共变；kappa 来自 l=0 ⇒ 对旋转不变。这是"概率等变"的典型结构。`kappa_bias` 用逆 softplus 初始化到 `kappa_init`（默认 1.0）。

### 4.5 `SupportHead`

`:533-593`：只取 l=0 分量（`_scalar_indices` 索引），过 3 层 MLP → 每点一个 logit（支撑点二分类）。

### 4.6 初始节点特征

仓库采用**轻量初始化**（`:736-744`）：把边的球谐向量按终点 scatter-sum，再线性投到 `irreps_hidden`——节点没有任何其他特征（不含坐标、不含类别 embedding、不含密度）。这是纯几何、严格等变的初始化。

### 4.7 图构建

默认 `knn_graph(k=num_neighbors)`（而非 `radius_graph`）。脚本 `train_l1.sh`、`train_l2.sh` 里 `num_neighbors=32`、`max_radius=0.1`——注意 `max_radius` 其实**未被用到**，因为 `knn_graph` 不看半径，`max_radius` 只在 `soft_one_hot_linspace` 的 `end` 被 `max_edge_len` 动态覆盖（`:724-730`）。

> **观察 4**：代码里 `max_radius` 是 dead argument（只被 `EfficientConvLayer.__init__` 接收但没参与前向），`radius_graph` 被 import 但未调用。如果未来想切回 radius-based 图，需要同时改 `forward` 的图构建分支。

### 4.8 等变性验证

`scripts/verify_equivariance.py` 随机生成点云 + SO(3) 旋转，对比 `f(R·P)` 和 `R·f(P)`。README 声明误差 ~1e-5（fp32 数值极限）。**前提**：knn_graph 的边集合在小旋转下也可能 flip（当两个候选等距），所以测试用高斯点云 + 10 次旋转平均，阈值 1e-3。

---

## 5. 损失函数

### 5.1 Geodesic loss（`loss_equivariant.py:24`）

`L = arccos(|cos<mu, gt>|)`，arccos 前 clamp 到 `[-1+1e-7, 1-1e-7]` 防梯度爆炸。

### 5.2 vMF NLL（`loss_equivariant.py:65`）

`L = -kappa * |cos| - log C_3(kappa)`，其中

```
log C_3(kappa) = log(kappa) - log(4π) - log(sinh(kappa))
log sinh(kappa) = kappa + log1p(-exp(-2*kappa)) - log 2   # 大 kappa 稳定
```

见 `Common/geometric_utils.py:63-89`。

> **观察 5**：配 `antipodal=True` 时用 `|cos|`，这在 vMF 的原始定义下**不是一个合法分布**（vMF 是 S² 上的有向分布）。"`|cos|` + vMF 归一化常数" 等价于把 μ 和 -μ 的两个 mode 合成的**对称化 vMF**，对应分布是 **Watson 分布（Fisher-Bingham 的一种）**。这一点代码没说明，但数学上是成立的近似：只在用户真的"不知道 up/down 方向"时用。

### 5.3 辅助 BCE（`:114`）

`binary_cross_entropy_with_logits(support_pred, support_gt)`；`--beta` 默认 0.5 控制权重。

### 5.4 组合损失

`L_total = L_direction + β · L_support`，`L_direction ∈ {vmf, geodesic}`。

> **观察 6**：原论文的 `fr_loss`（fitting residual）**在等变路径里被完全放弃**。一方面它依赖于"y 是上方向"这个 fixed 坐标系约定，对 SO(3) 随机旋转后的点云没有意义；另一方面它假设底面平面的特殊几何，非平面类就失效。放弃它是合理的。

---

## 6. 训练与评估

### 6.1 训练循环（`model.py:268-426`）

- 优化器：**AdamW**（原始路径是 Adam），lr=1e-3，wd=1e-5（脚本里）。
- 调度：**CosineAnnealingLR**（原始路径是 StepLR）。
- Epochs：脚本默认 **200**，bs=8 (L=1) 或 4 (L=2)。
- 每 5 epoch 评估一次，保存当前最优 `equivariant_best_{timestamp}.pth`（`models/` 下有 ~30 个 checkpoint，时间戳从 2026-04-15 到 2026-04-16，都是最近两天训练出来的）。

### 6.2 评估（`model.py:469-521`）

```
for data in testloader:
    outputs = model(data)
    mu, kappa = outputs['direction_mu'], outputs['direction_kappa']
    errors = angular_error_deg(mu, data.y_direction, antipodal=True)
```

输出：mean error / median error / Acc@5 / Acc@10 / **kappa-accuracy 相关系数**（后者用于检验置信度标定，`model.py:516-519`，`corrcoef([kappa, -error])`）。

> **观察 7**：这个相关系数指标**是本实现的一个亮点**——原论文不提供置信度，自然也没法校准。当前实现不仅输出 kappa，还会衡量 "kappa 大 ⇔ error 小" 的一致程度。

### 6.3 可复现性问题

- 随机种子在数据加载器初始化**之后**才设置（`model.py:300-305`），而训练数据 shuffle 的第一个 epoch 的 RNG 已经消耗；
- `Rotation.random()` 在 `get()` 里调用，没有 seed 传入，每次运行完全不同；
- 这两点加起来，**当前训练无法严格复现**。对研究型代码尚可，产品化前需要修。

---

## 7. 设计评价（逐点）

### 7.1 架构等变 vs 启发式等变（正面）

把 SO(3) 不变 / 等变性直接烧进架构，意味着：
- **不需要 100× 旋转增广**（原论文的主要开销来源）；
- **无论训练集如何采样朝向，测试泛化都严格保持**；
- **不需要 test-time augmentation**；
- 损失和模型输出天然分离为 "方向 (l=1)" 和 "置信度 (l=0)"。

这是一个有教益的设计：对单物体直立估计这个特化任务，等变架构是合理的归纳偏置。

### 7.2 过度设计：l=2 球谐 + S² 激活（负面）

**对"输出一个 l=1 方向"的任务，l=2 通道几乎没有必要**：
- 方向本身就是 l=1 量；
- l=2 通道只能通过 TP(l=1, l=1)→l=2 产生，再在下一层被 TP(l=2, l=1)→l=1 回流；
- 分子力场 / 应力张量这类**需要输出 l≥2 量**的任务才真正吃 l=2 红利；
- 点云分类 / 分割里 l=2 的收益也是边际的。

现实的代价：`train_l2.sh` 里 `128x0e+128x1o+128x2e`, 6 层, bs=4，显存和速度都翻倍以上。

S² 激活（EquiformerV2 借鉴）同理：核心价值是让高阶张量支持非线性混合，但当任务输出只要 l=1 时，**Gate 非线性已经够用**。

### 7.3 MACE 风格 depthwise TP（正面）

这是仓库最扎实的工程优化——把 `FullyConnectedTensorProduct` 拆成 `linear + uvu-TP + linear`，把 per-edge 权重数从 ~3456 压到 ~11（L=2 时）。README 称 ~5-8× 加速，实测 L=1 情况下延迟 ~6ms。这一块是"等变网络可以被做快"的实际例证。

### 7.4 `max_radius` / `radius_graph` 的死参数（轻微负面）

见 §4.7。建议要么切到 `radius_graph`（物理上更合理——半径截断反映局部邻域结构），要么把 `max_radius` 彻底删掉。

### 7.5 归一化与数值稳定（正面）

- `num_neighbors ** 0.5` 归一化 scatter 结果（`:162`、`:449`）防止随 k 漂移；
- vMF 归一化常数用 `log1p(-exp(-2κ))` 稳定（大 κ 时）；
- arccos 前 clamp 到 `1 - 1e-7`；
- kappa `softplus(0.1 * x + bias).clamp(0.1, 500)` 防爆炸。

这些细节都写得到位。

### 7.6 antipodal vMF 的语义（轻微负面）

见 §5.2 的观察。如果训练标签真的有方向（重力朝下），用 antipodal 会**放弃一半信息**，让模型永远不知道"上还是下"。现有 `rotation_matrix_to_upright` 返回 `R[:, 1]`，是有向的。**推荐**：
- 要么训练时关掉 `antipodal`，让模型学到有向 up；
- 要么把 kappa 换成 Watson 分布的 κ，保持语义一致。

### 7.7 fallback 对 GT 的隐式泄漏（负面，原始路径）

`model.py:260` 当支撑点识别完全失败时，直接用 `torch.inverse(rotm[i].cpu())[1]`——这是 GT 旋转矩阵的第二行，等于 GT up 方向本身。这会让"支撑点数 = 0 的样本"的报告准确率虚高。**应该改为固定输出 `[0, 1, 0]` 或 NaN 报错**。

### 7.8 SupportHead 的存在价值（中性）

目前 SupportHead 纯粹是辅助任务，提供 l=0 的监督信号帮助骨干学到"底面 vs 非底面"的语义。但它**与方向 head 的直接耦合为零**——方向头只从全局 l=1 池化得来。

**更深的设计空间**：把 support probability 当作权重，用可微 SVD 算加权协方差的最小特征向量，把该方向作为 l=1 辅助目标（现仓库 `Common/geometric_utils.py:118-168` 已经实现了 `weighted_plane_normal`，但**没人调用它**）。这是一个"原作者显然想做、但没接上"的改造点。

### 7.9 在线随机旋转 vs 在线高斯数据（正面）

等变路径每个 epoch 对同一个点云见到随机朝向 ⇒ 训练集朝向分布近似 SO(3) 均匀。**如果模型严格等变**，这一步甚至都不必要（理论上训练一个朝向就够）；但在线旋转可以帮助缓解 "knn_graph 不严格等变"（边集合在小旋转下会翻转）引起的数值抖动，算是额外鲁棒性来源。

### 7.10 checkpoint 产出过多（轻微运营问题）

`models/` 下 ~30 个 `equivariant_best_*.pth`，都是最近两天的训练试验。每次最优 checkpoint 都新建文件名（`model.py:403`，带时间戳），不覆盖——磁盘会持续膨胀。建议固定文件名 `equivariant_best.pth` + 分开保留 `equivariant_final.pth`。

---

## 8. 替代方案对比（摘要，详见 `upright_direction_literature_review.md`）

| 方案 | 机制 | 严格 SO(3) 等变 | 实现复杂度 | 预期对 "单方向" 的 ROI |
|---|---|---|---|---|
| 原始 UprightNet + RANSAC | 启发式 + 100× 增广 | 否 | 中 | 低（依赖底面） |
| **本仓库 e3nn + MACE + EquiformerV2** | 架构等变，l≤2 | 是 | 高 | 中（功能完备但过剩） |
| **VN-DGCNN**（Deng ICCV 2021） | l=1-only 向量神经元 | 是 | 低 | **高**（对症） |
| **Frame Averaging**（Puny ICLR 2022）+ DGCNN | PCA frame 平均，backbone 非等变 | 是 | 低 | **高**（代码改动最小） |
| **Learned Canonicalization**（Kaba ICML 2023） | 学一个 canonicalization net | 近似（学得到） | 中 | 高（与任务语义一致） |
| **PCA 最小特征向量** | 零参数几何基线 | 是 | 极低 | 取决于类（建筑/家具高，动物低） |
| **DGCNN + 旋转 TTA** | 数据对称化 | 近似 | 低 | 中 |

### 概率头

| 分布 | 适用 | 当前实现 |
|---|---|---|
| vMF（有向 S²） | 重力方向（有向） | **已实现**，但被 antipodal 削弱 |
| Watson / ACG | S² 上对跖对称 | 未实现；推荐 |
| Bingham on S³ | 完整 SO(3) | 未实现 |
| Matrix Fisher | 完整 SO(3) | 未实现 |
| Implicit-PDF | 多峰（对称物体） | 未实现 |

---

## 9. 具体改造建议

按**投入产出比**排序：

### 9.1 零成本必做（1-2 小时）

1. **修 fallback 泄漏**：`model.py:260` 改为 `orientation = torch.tensor([0., 1., 0.], dtype=torch.float32)`。否则原始基线的准确率数字不可信。
2. **增加 PCA 基线**：写一个 10 行的 `pca_upright.py`：对每个点云做协方差特征分解，最小特征向量 + 符号修正 → up 方向。这个基线在"动物 / 不规则形状"类上大概会很差，在"家具 / 交通工具"类上可能已经几度误差。**任何等变网络都必须在这个基线上面**，否则没有存在意义。
3. **增加 DGCNN + 3-vector regression + TTA 基线**：把 `network.py` 的 UprightNet 输出头改为全局 pool + MLP → 3d 向量，TTA 做 4-8 次随机旋转平均；损失直接用 geodesic。这会是"非等变神经网络的最强简单基线"，应该接近当前等变网络的结果。
4. **去掉 `max_radius` 死参数**或切换到 `radius_graph`。
5. **固定文件名**保存 checkpoint，清理 `models/` 下历史 ckpt（30+ 个）。

### 9.2 低成本高收益（半天到一天）

6. **重新评估 antipodal 语义**：
   - 如果业务上关心 "up vs down" 有向，把 `antipodal=False` 重训，损失里的 `|cos|` 改回 `cos`。
   - 如果业务上不关心，把 vMF 换成 **Watson 分布**；公式只差 `κ` 前系数和归一化常数（详见 Kasarapu & Allison 2015）。
7. **调用 `weighted_plane_normal`**：把 `SupportHead` 输出 sigmoid 作为权重，SVD 算加权最小主方向，作为辅助的 l=1 监督目标（与 mu 求 geodesic loss）。这相当于把原论文的 "支撑点 + RANSAC" 思路可微化接回来，代码已经写好，只是没接。
8. **kappa 标定曲线**：训练日志里加一个 reliability diagram —— 按 kappa 分桶，比较预测角误差的分位数，检查置信度真的有没有意义。现有相关系数只看单调性，不看校准。

### 9.3 架构替换实验（2-5 天）

9. **跑 VN-DGCNN 对照**：不依赖 e3nn，纯 PyTorch，几百行。输出用 l=1 全局池化 + vMF/Watson 头。预期速度是当前 L=2 的 1/5，准确率接近。
10. **跑 Frame Averaging 对照**：PCA 算 ≤8 个 frame，DGCNN 前向 8 次取平均。**最有可能彻底超过当前方案的路径**（因为可以直接复用预训练 DGCNN 权重）。
11. **跑 Learned Canonicalization**（Kaba et al. ICML 2023）：输出旋转 R_canon，取 `R_canon[:, 1]` 作为 up；损失用 geodesic。与本任务语义对齐度最高。

### 9.4 任务边界重新审视

12. **分类 vs 回归**：如果下游只需要"站立 vs 翻倒"这类粗分类，当前 vMF 回归是过度设计；如果下游需要**完整 6DoF 姿态**，当前输出又远不够，应升级到 Matrix Fisher 或 Deep Bingham。
13. **多峰支持**：花瓶、哑铃、轮子这类对称物体，任何单峰分布（vMF / Watson / Matrix Fisher）都会在两个合理解之间平均，导致大错。Implicit-PDF（Murphy ICML 2021）可以天然表达。**先跑错误分析**，把 error > 45° 的样本按类别分桶，看多峰是不是真问题。

---

## 关键结论

本仓库是一个**工程完备、数学正确**的 E(3)-等变点云网络实现，作为学习 e3nn / MACE / EquiformerV2 的样板工程非常合格；但对"**单方向估计**"这个任务，架构明显过剩。

真正对结果负责的改造优先级：

1. 先把 PCA 和 DGCNN+TTA 基线跑通。如果等变网络没有显著超越它们，当前堆栈的存在就值得重新评估。
2. 把 antipodal 的语义和业务目标对齐（有向就去掉，无向就换 Watson）。
3. 把 `weighted_plane_normal` 接回训练流程，让"支撑点语义"与"方向输出"真的耦合，而不是两条并行的 head。
4. 把"l=2 + S² 激活"视为消融实验而非主力配置；主力应该是 **VN-DGCNN 或 Frame-Averaged DGCNN**（严格等变、实现 5-10× 简单、性能相当）。
5. 对对称物体先做错误分析，再决定要不要上 Implicit-PDF。

**一句话**：等变架构是对的归纳偏置，但选 e3nn + MACE 这一套重武器去做"一个方向"这个轻任务，工程成本和准确率收益不成正比。VN-DGCNN / Frame Averaging / Learned Canonicalization 任选其一，都更对症。

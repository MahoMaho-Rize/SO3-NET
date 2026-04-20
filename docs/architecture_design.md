# DLIE-UprightNet 架构设计

> Differentiable Lie-algebra Iterative Upright Estimator
> 状态：v0.3（tangent-plane projection 已加入，待重新训练）

---

## 1. 设计目标

### 1.1 任务

从 SO(3) 随机旋转后的点云 P ∈ ℝ^{N×3} 估计其直立方向 **u ∈ S²**。

### 1.2 基线与其缺陷

Pang 2022 (CVPR) 建立的 pipeline：

```
P ──► trunk(DGCNN+SelfAttn) ──► per-point support prob
                                 │
                                 ▼
                              RANSAC 平面拟合 ──► 法向 + 符号消歧 ──► u
```

诊断得到的失败模式：
- **5.74% 样本 180° 翻转**，集中在 lamp/bed/vase（灯罩 vs 灯底的支撑面歧义）
- **11pp test/train 准确率 gap**：trunk 过拟合 1110 训练物体
- 不可微 → 无法端到端优化

### 1.3 核心创新

**"Differentiable Lie-algebra Iterative Refinement"**

1. **RANSAC → 可微加权 SVD**：用 per-point 支撑概率加权协方差的最小特征向量作为粗初始化
2. **Lie 代数迭代细化**：在 so(3) 切空间做 coarse-to-fine 更新，用 exp map 保持 SO(3) 正定
3. **Tangent-plane projection**：ΔR 限制在当前 up 的切空间 S² 上，消除绕 up 轴的 gauge 自由度
4. **Polarity feature 注入**：显式计算等变极性向量 v_polar + skew_vec，给 refine head 提供物体极性信号
5. **严格等变**：相对 frame 迭代机制保证 f(gP) = g·f(P)（g ∈ SO(3)）
6. **端到端可微**：梯度从最终 geodesic loss 回传至 trunk 的 support prediction

---

## 2. 整体数据流

```
                      ┌─────────────────────────────────────────────┐
                      │           Input:   P ∈ ℝ^{B×N×3}            │
                      └──────────────────┬──────────────────────────┘
                                         │
                                         ▼
                       ┌──────────────────────────────────────┐
                       │        ITERATION 0 (Coarse Init)     │
                       │                                      │
                       │   sup_0 = trunk(P)  ∈ [0,1]^{B×N}    │
                       │   u_0, λ = weighted_plane_SVD(P,sup_0)│
                       │   hint   = PCA_largest_axis(P)        │
                       │   R_0    = so3_from_up(u_0, hint)    │
                       └──────────────────┬──────────────────┘
                                          │
                              ┌───────────┴───────────┐
                              │                       │
                              ▼                       ▼
                       R_iters[0] = R_0         delta_omegas = []
                              │
                   ┌──────────┴──────────────────────────────┐
                   │                                         │
                   │         for t = 1 … T (refinement)      │
                   │                                         │
                   │   P_t     = P @ R_t    # R_t^T·p       │
                   │   feat,s  = trunk(P_t)                  │
                   │   polar   = PolarityFeat(P_t, s)  ∈ ℝ⁶ │
                   │   fused   = cat(feat, polar) ∈ ℝ^1030  │
                   │   ω̂      = RefineHead(fused, max_θ_t) │
                   │                                         │
                   │   # tangent projection (NEW):           │
                   │   u_t     = R_t[:, :, 1]  (current up) │
                   │   ω       = ω̂ − (ω̂·u_t)·u_t          │
                   │                                         │
                   │   R_{t+1} = R_t @ exp_so3(ω)           │
                   │   R_iters ← R_{t+1},  delta_omegas ← ω │
                   │                                         │
                   └──────────────────┬──────────────────────┘
                                      │
                                      ▼
                        u = R_T[:, :, 1]  ∈ S²

                   ┌───────────────────────────────────────┐
                   │   Loss (multi-iter geodesic, γ=0.8):  │
                   │   L = Σ_{t=0..T} γ^(T−t) · arccos|u_t·g|│
                   └───────────────────────────────────────┘
```

---

## 3. 各组件详解

### 3.1 Trunk（预训练 + 可选微调）

**`UprightNetTrunk`**（来自 Pang 2022，冻结或用 lr=1e-5 微调）

```
P_t ∈ ℝ^{B×N×3} → transpose → (B, 3, N)
  │
  ├─ 3 × EdgeConv(k=20, ch: 3→32→64→128)
  ├─ 4 × SelfAttention(ch=128)  (stacked, cat → 512)
  ├─ Conv1d(512 → 1024) + AdaptiveMaxPool1d → global_feat ∈ ℝ^{B×1024}
  │
  └─ [x_a, x_b, x_c, global] → Conv1d → Conv1d → Conv1d → sigmoid
                                                        │
                                                        ▼
                                               sup ∈ [0,1]^{B×N}
                                               (per-point support prob)
```

**输出**：`global_feat (B, 1024)` + `sup (B, N)`，二者共享同一前向的特征提取。

**参数量**：2.22M（占模型 87%）。

### 3.2 Coarse Init（可微 SVD）

**`weighted_plane_normal(P, w)`** (Common/geometric_utils.py)

```
centroid   = Σ w_i · p_i  /  Σ w_i
centered   = p_i − centroid
Cov        = centered^T · diag(w) · centered  / Σ w_i     + ε·I
λ, V       = eigh(Cov)                              # ascending
normal     = V[:, :, 0]                             # smallest eigenvalue
sign       = sign(normal · (mass_center − support_center))
up_0       = sign · normal                          # ∈ S², SO(3)-equivariant
```

- **可微性**：`torch.linalg.eigh` 全程可微
- **等变性**：SVD 对 SO(3) 严格等变
- **替代 RANSAC**：所有点参与加权，soft weights 来自 sigmoid（不是硬阈值 inlier）

### 3.3 so3_from_up

从单个 up 向量 + 一个 equivariant in-plane hint 构造完整 `R ∈ SO(3)`：

```
hint'  = hint − (hint·u)·u        # Gram-Schmidt 正交化
x      = normalize(hint')
z      = normalize(u × x)
x      = normalize(z × u)         # 保持 det(R) = +1
R      = [x | u | z] ∈ ℝ^{3×3}    # R[:, :, 1] ≡ u
```

`hint = pca_largest_axis(P)` 保证整个构造严格等变。

### 3.4 Polarity Features

**`PolarityFeatures(P_t, sup)` → ℝ⁶**

两个显式极性向量，相对 frame 下都等变：

**(a) 加权极性向量** `v_polar ∈ ℝ³`

```
w          = sigmoid(sup).unsqueeze(-1)
w_norm     = w / Σw
v_polar    = Σ w_norm_i · (p_i − centroid)
```

物理意义：从支撑区域中心**指向物体整体方向**。对灯就是"从灯底指向灯罩"。

**(b) 偏度向量** `skew_vec ∈ ℝ³`

```
skew_vec   = E[(p − centroid)³] / std³          # per-axis 3rd central moment
```

物理意义：沿各轴分布的不对称性，区分"底平顶尖" vs "顶平底尖"。

**注意**：两者都在 canonicalised `P_t` 坐标系下计算，所以在相对 frame 方案里都是 gauge-invariant 的标量/向量组合。

### 3.5 Refine Head

**`ContinuousRefineHead`**

```
fused (B, 1030) = concat[trunk_global (1024), polarity (6)]
  │
  ├─ Linear 1030 → 256
  ├─ SiLU
  ├─ Linear 256 → 256
  ├─ SiLU
  └─ Linear 256 → 3                     # raw ω̂
      (init: gain=0.01, bias=0)

ω̂_scaled = ω̂ · tanh(‖ω̂‖ / θ_max) / ‖ω̂‖    # soft clip to θ_max
```

每次 iteration 传入不同 `θ_max`（per-iter max angle），实现 coarse-to-fine。

### 3.6 Tangent-plane Projection（本次关键修正）

Refine head 输出 `ω̂ ∈ ℝ³` 后**必须**投影到当前 up 的切平面：

```
u_t    = R_t[:, :, 1]
ω      = ω̂ − (ω̂ · u_t) · u_t
```

**为什么**：

任务 loss 只监督 `R[:, :, 1] = u_t`，绕 u_t 轴的旋转是 **gauge symmetry**——loss 梯度在该方向为 0。不投影时 refine head 可能学习"在无梯度方向乱转"（诊断中观察到 delta 25° 但 error 不变的现象）。

投影后：
- ω ⊥ u_t 严格成立（验证：`|ω · u|` max 4.47e-08）
- 训练梯度信号强度 **6× 提升**（0.11 vs 0.017）
- exp_so3(ω) 的一阶近似是 `u_{t+1} ≈ u_t + ω × u_t`——标准 S² 切空间更新

### 3.7 SO(3) Exponential Map (Rodrigues)

```
K     = skew(ω)               # 3×3 skew-symmetric from R³
θ     = ‖ω‖
exp_so3(ω) = I + (sinθ/θ)·K + ((1−cosθ)/θ²)·K²
```

严格可微，保 SO(3)。

---

## 4. 等变性声明

### 4.1 严格等变的部分

对任意 g ∈ SO(3)，输入 gP 得到 f(gP) = g · f(P)，当且仅当：

1. **Coarse init**: `weighted_plane_normal`、`pca_largest_axis`、`so3_from_up` 都是严格等变（SVD 经 torch.linalg.eigh 实现）
2. **Relative-frame iteration**: 第 t 步 trunk 的输入是 `R_t^T · P`，若 R_t(gP) = g·R_t(P) 则 trunk 输入不变
3. **Tangent projection**: 投影操作等变（u_t 随 R_t 等变，ω̂ 在 trunk 不变输入下不变）
4. **exp_so3**: 元素级函数，等变

### 4.2 近似等变

**Trunk 本身不是结构等变**，但 Pang 2022 用 100× 随机旋转增广训练，在实际数据上数值上 ≈ 等变。Exp 3 诊断显示：
- `trunk(P_rotated)`：support IoU 0.76（训练分布）
- `trunk(P_canonical)`：0.69
- `trunk(P_PCA_frame)`：0.71

差距可接受。整体 pipeline 是 **"near-equivariant"**——对随机旋转的 test 数据能稳定工作。

---

## 5. 损失函数

### 5.1 多迭代 geodesic loss

```
L = Σ_{t=0..T} γ^(T−t) · arccos|⟨R_t[:, :, 1], u_gt⟩|
```

- γ=0.8，T=3 → 权重 [0.512, 0.64, 0.8, 1.0]
- t=0 项监督 coarse init 的可微 SVD 输出 → **让 trunk 的 support prediction 也被 upright loss 直接优化**
- 后续项让 refine head 学到修正方向
- `|·|` 处理 antipodal 歧义：物理上 up 和 -up 是同一个轴的两个方向

### 5.2 训练策略

| 阶段 | 参数 | 说明 |
|---|---|---|
| 当前 | lr_trunk=0, lr_head=1e-3 | 冻结 trunk，只训 refine head（验证架构） |
| 下一步 | lr_trunk=1e-5, lr_head=1e-3 | 解冻 trunk，端到端微调（主力实验） |
| 消融 | trunk=random init | 回应审稿"为什么不从头训" |

---

## 6. 超参数表

| 参数 | 值 | 说明 |
|---|---|---|
| num_iters T | 3 | coarse + 3 次 refinement |
| max_angle_schedule | [π, π/4, π/18] = [180°, 45°, 10°] | coarse-to-fine |
| trunk global dim | 1024 | UprightNet self-attn 后的 pool |
| polarity dim | 6 | v_polar (3) + skew_vec (3) |
| refine head hidden | 256 | 2 层 SiLU MLP |
| γ (loss decay) | 0.8 | 渐进权重 |
| lr_head | 1e-3 | AdamW |
| lr_trunk | 0 → 1e-5 | 两阶段训练 |
| batch size | 48 | RTX5880 48GB |
| num_points | 2048 | 与 Pang 2022 一致 |
| eps_cov (SVD 正则) | 1e-5 | 防退化 |
| grad clip | 5.0 | 保护 eigh 梯度 |

---

## 7. 当前结果

| 模型 | Mean | Median | Acc@10 | Flip | 备注 |
|---|---|---|---|---|---|
| 官方 Pang RANSAC | 11.17° | 0.71° | **93.05%** | 5.72% | Phase 0 baseline |
| DLIE-Refine (zero-shot) | 16.23° | 0.80° | 82.09% | 6.12% | 可微 SVD 版 Pang baseline |
| DLIE-Refine (1 epoch, no proj) | 16.65° | 0.83° | 81.91% | 6.38% | ❌ refine 没学，gauge 乱转 |
| DLIE-Refine (1 epoch, **with proj**) | TBD | TBD | TBD | TBD | 🔬 进行中 |

### 7.1 Oracle 上限（diag_ceilings.py）

- PCA frame oracle (T=0): 70.67% acc@10
- PCA frame + 3-step beam search (T=3): **100% acc@10**, mean 1.97°
- DLIE coarse init 起点 82%，refine oracle 上限理论上 > 100%

---

## 8. 关键文件

| 文件 | 内容 |
|---|---|
| `network_lie.py` | 所有李代数网络（含 4 个变体，推荐用 `DifferentiableLieUprightRefineNet`） |
| `Common/geometric_utils.py` | `weighted_plane_normal` 可微 SVD |
| `scripts/train_dlie_refine.py` | 训练脚本 |
| `scripts/eval_dlie_refine_zeroshot.py` | 零样本评估 |
| `scripts/eval_official_baseline.py` | Pang baseline 复现 |
| `scripts/diag_ceilings.py` | 4 个 oracle 诊断 |

---

## 9. 下一步

1. **重新训练**（带 tangent projection）1 epoch，验证 refine 能降低 error
2. 如果 refine 真的 work：**解冻 trunk**，完整 5 epoch 训练
3. **消融**：no-projection / no-polarity / T=1/2/3/4 / frozen vs unfrozen trunk
4. 泛化评测：UprightNet5（未见类别）、partial 扫描（ScanObjectNN）
5. 等变性数值验证：`scripts/verify_equivariance.py` 式的 test

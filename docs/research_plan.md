# 等变可微 Upright-Net 研究计划

> 论文目标：**"Equivariant Differentiable Upright-Net: Revisiting Pang 2022 with End-to-End Support Plane Regression, Antipodal-Aware Density, and Unimplemented Stability Loss"**
> 三个核心贡献：**C1 可微等变支撑平面回归** / **C2 antipodal-aware 等变分布头** / **C3 等变版 L_Stab（补回原论文漏实现）**
> 指导原则：**每个阶段都有独立 deliverable**，即使后续阶段失败，当前阶段的成果也能独立成文（workshop / short paper）。
> 估计总时长：**12 周**（3 个月）。

---

## 阶段 0 · 基线复现与诊断（第 1 周）

### 目标
在动任何代码之前，建立**所有后续对比的公正基线**，并收集 Fig.5 失败模式的定量证据。

### 任务

- [ ] **0.1** 用 `uprightnet-reference/model/model.pth` 跑 test，记录 per-sample angular error，保存为 `logs/baseline_official.npz`。
- [ ] **0.2** 把 `model.py:201` 的 GT 泄漏 fallback（`torch.inverse(rotm[i])[1]`）改为 `[0, 1, 0]`，重跑 test，记录"真·baseline"数字。**两个版本都要留档**。
- [ ] **0.3** 画 CDF(error) 曲线和 error 直方图：验证 Pang 论文 Fig.4 的 "τ≈180° 陡升" 现象在你的环境下真实存在。
- [ ] **0.4** 挑出 error > 90° 的样本，按类别统计 + 可视化 top-20 失败案例（点云 + 预测方向 + GT 方向）。把它们分类标注为 {"180° 翻转", "干扰平面", "其他"}。
- [ ] **0.5** 跑 fork 里已有的 L=1 等变 checkpoint（`models/equivariant_best_*.pth` 最新一个）做同样的诊断。

### Deliverable
`docs/baseline_diagnostics.md`：一页表格 + 三张图。**这是你接下来 12 周所有创新点的证据基础。**

### Go/No-Go 决策点
- 若 **Fig.5 的 180° 翻转失败不存在**：C2 核心 premise 垮，需要改叙事（转向 C1+C3 + 数据效率路线，C2 降为次要贡献）。
- 若 **GT 泄漏修正后官方数字不变**：说明原论文 0 支撑点样本极少，"修复 baseline 泄漏" 这个 angle 只是诚实披露，不是 contribution。
- 若官方 checkpoint 跑不通：先修依赖，不进入下阶段。

---

## 阶段 1 · 工程基础设施（第 1-2 周，与阶段 0 并行）

### 目标
把 fork 里已有的等变实现收拾干净，让所有消融实验有一个统一、可复现的运行框架。

### 任务

- [ ] **1.1** 修 `EquivariantDataLoader` 的随机种子：保证 `Rotation.random()` 由 seed 控制（`model.py:300-305` 的 seed 设置要移到 DataLoader 创建前）。
- [ ] **1.2** 清理 `models/` 下 30+ 个历史 checkpoint，统一命名为 `best.pth` / `final.pth`（带 `--run_id` 区分不同实验）。
- [ ] **1.3** 删掉或正确实现 `max_radius` 死参数（`network_equivariant.py:724`）—— 建议切换到 `radius_graph` 并做一次对比。
- [ ] **1.4** 增加**统一评估接口** `eval_harness.py`：
  - 输入 checkpoint 路径 + 数据集名
  - 输出 `{mean_error, median, acc@5, acc@10, acc@30, flip_rate, per_category}` JSON
  - 所有后续实验共用这套
- [ ] **1.5** 增加 **per-sample 误差导出** 能力，为后续错误分析做准备。
- [ ] **1.6** 写 `scripts/run_all.sh`：一键跑完所有消融实验的 grid。

### Deliverable
- 统一的 `eval_harness.py`
- `configs/` 下至少 6 个 YAML 配置文件（baseline / +C1 / +C2 / +C3 / +C1C3 / full）
- CI 式的 "跑 smoke test 30 秒出结果" 流程

---

## 阶段 2 · 实现 C3（等变 L_Stab 损失）（第 2-3 周）

> **为什么先做 C3**：代码量最小（~30 行），独立于 C1/C2，能最快产出一个可报告的结果。即使其他贡献失败，"补回原论文漏实现的损失" 本身是一篇 short paper 素材。

### 任务

- [ ] **2.1** 在 `Common/loss_equivariant.py` 新增 `equivariant_stability_loss(mu, pos, support_soft_weight, batch)`：
  - 按 batch 池化：`support_center = Σ w_i · pos_i / Σ w_i`（等变 l=1 量）
  - `mass_center = mean(pos)`（等变 l=1 量）
  - **投影距离** `d = ‖(I - μμᵀ) · (mass_center - support_center)‖₂`（不变 l=0 量，严格等变）
  - 对比论文 Eq.12 的凸包版本：凸包可微实现复杂，用"质心投影到预测 up 轴"是等价的静力学直觉且梯度稳定
- [ ] **2.2** 集成到 `EquivariantLoss`，加 `--gamma_stab` 超参（论文默认 α₂=1，但等变版可能需要调）。
- [ ] **2.3** 单元测试：验证 `L_Stab(mu, Rx) = L_Stab(R⁻¹·mu, x)` 到 1e-6。
- [ ] **2.4** 消融实验：
  - 只加 L_Stab（γ ∈ {0.01, 0.1, 1.0, 10}）
  - 与 L_vMF 联用
  - 报告在 UprightNet15 上的 mean_error 和 flip_rate

### Deliverable
一个 **3 行结果表**：baseline / baseline+L_Stab / 其他 γ 值，以及等变性数值验证的单元测试日志。

### Go/No-Go 决策点
- 若 L_Stab 带来 **>0.3° mean_error 下降**：C3 确立为独立贡献。
- 若 L_Stab 无效或降分：转而作为**消融实验里的"曾经尝试但无效" 章节**，降级但不浪费 —— 因为你已经**证伪了原论文公式 12 的实际价值**，这本身也是有意义的发现。

---

## 阶段 3 · 实现 C1（可微等变支撑平面回归）（第 3-5 周）

### 任务

- [ ] **3.1** 在 `network_equivariant.py` 新增 `DifferentiableSupportPlaneHead`：
  - 输入：`node_features (N, hidden)`
  - 输出：`soft_weight ∈ [0,1]^N`（sigmoid，现成）+ `plane_normal ∈ R³`（由 `weighted_plane_normal(pos, soft_weight)` 产出）
  - 接回现有的 `SupportHead` 作为 warm-start，训练初期给 BCE 监督 → 后期加熵正则让权重自由
- [ ] **3.2** 在 `VMFDirectionHead` 旁边并联第二路方向输出 `mu_plane = plane_normal`。
- [ ] **3.3** 损失组合：
  ```
  L = L_vMF(mu_global) + β·L_vMF(mu_plane) + λ·L_consist(mu_global, mu_plane)
     + γ·L_Stab(mu_final, pos, soft_weight)
     + optional: δ·L_BCE(soft_weight, pid)   # warm-start 用
  ```
- [ ] **3.4** 符号消歧问题：可微 SVD 输出的 normal 有 ±1 不确定性。用"投影到 mass center 一侧"做符号修正（代码已有 `Common/geometric_utils.py:162`）。这个符号修正是**不连续的 sign 函数**，会破坏梯度——需要用可微替代（`tanh(k·⟨n, c_direction⟩)` for 大 k）。
- [ ] **3.5** **关键消融实验 A（C1 的决定性实验）**：
  - 配置 (a)：`δ > 0`（BCE 监督权重） → baseline 可微版
  - 配置 (b)：`δ = 0`（无 BCE 监督，只靠方向损失倒逼 soft weight 涌现）
  - 成功标准：(b) 达到 (a) ≥95% 性能，且可视化的 soft weight 在底面附近确实高亮
- [ ] **3.6** 可视化：把 soft_weight 渲染到点云上，对比 Pang 论文 Fig.3 的 pid 标注。要求产出 **论文级 figure**，横向对比 10 个类别各 2 个样本。

### Deliverable
- 端到端可微等变支撑平面模型 + checkpoint
- 关键消融表：(a) 有 BCE / (b) 无 BCE 的性能对比
- **论文主 figure**：learned soft weights 可视化

### Go/No-Go 决策点
- 若 (b) 配置 mean_error 崩（>50°）：说明"支撑点语义" 必须靠 BCE 监督才能学到。降级叙事为 "可微化 RANSAC"（弱 claim），(b) 作为"我们尝试过的失败方向" 放附录。
- 若 (b) 配置能 work：这是 **全篇论文最强卖点**，把它推到 C1 章节主位。

### 风险与对冲
- **风险 1**：可微 SVD 在点云共面时梯度爆炸。**对冲**：`Common/geometric_utils.py:152` 已加 `eps·I`，观察梯度范数；若仍不稳定，退回用 power iteration 替 eigh。
- **风险 2**：soft weight 塌到全 0 或全 1。**对冲**：加 `entropy_reg = -mean(w log w + (1-w) log(1-w))`，鼓励既不全开也不全关；加 `sparsity_reg = mean(w) - target_ratio`，target_ratio 约等于原论文 pid 里 1 的比例（≈15%）。
- **风险 3**：符号消歧的 tanh 软化在训练早期不稳定。**对冲**：warm-start 前 5 epoch 用硬 sign（detach sign 本身，只让 normal 有梯度）。

---

## 阶段 4 · 实现 C2（Antipodal-aware 等变分布头）（第 5-7 周）

### 任务

- [ ] **4.1** 新增 `AntipodalVMFHead`：
  - **等变轴输出**：`axis ∈ R³ / {±}`（用未归一化的 l=1 向量表示"无向轴"，损失里用 `|cos|`，推理时选 sign）
  - **方向二元 logit**：`flip_logit ∈ R`（l=0 标量，不变）由独立的 MLP 头产生，决定 `up = +axis` 还是 `up = -axis`
  - **置信度**：κ（l=0）保持
- [ ] **4.2** 损失：
  ```
  L_axis = vMF(axis, κ, gt_axis, antipodal=True)       # 学轴
  L_flip = BCE(flip_logit, sign(gt_direction · axis_pred))  # 学方向
  L_C2 = L_axis + μ·L_flip
  ```
- [ ] **4.3** 推理流程：
  ```
  axis = normalize(axis_pred)
  flip = sign(flip_logit - 0)  # 或 sigmoid > 0.5
  up_final = flip · axis
  ```
- [ ] **4.4** **关键消融实验 B（C2 的决定性实验）**：
  - 把预测 `up_pred` 分解为 "轴角误差" `arccos(|up_pred · up_gt|)` 和 "是否翻转" `sign(up_pred · up_gt) < 0` 两个维度
  - 对比基线（vMF）vs 新方法（axis+flip），分别报告两个维度的误差
  - **预期结果**：新方法在"轴角误差"上优于或持平，"翻转率"显著下降 —— 对应 Pang 论文 Fig.4 的 τ=180° 尾巴被砍掉
- [ ] **4.5** 在"对抗干扰平面"子集上额外评测：
  - 从 UprightNet15 人工挑出上下对称/含歧义平面的样本（花瓶、圆桌、吊灯、哑铃类）
  - 或构造合成干扰：给点云加一个人工"天花板平面"
  - 报告对该子集的 flip_rate

### Deliverable
- 分离的 "轴误差 / 翻转率" 双轴评估表
- "干扰平面子集" 专门评测
- 与 Idea 2（L_Stab）和 C1（可微支撑）联合的完整模型结果

### Go/No-Go 决策点
- 若在 ModelNet15 上 C2 对 flip_rate 几乎无影响（阶段 0 诊断显示翻转错误本来就 <5%）：C2 主场地搬到 **OOD 和 partial 数据**（阶段 5），因为 180° 翻转失败在 OOD 上更频繁。
- 若 C2 在干净数据上效果显著：推到主位。

---

## 阶段 5 · OOD 与鲁棒性实验（第 7-9 周）

### 目标
在这里做 **"等变性的真正价值"** —— 不是准确率提升，而是**数据效率 + OOD 鲁棒性**。

### 任务

- [ ] **5.1 数据效率实验**：
  - 固定测试集
  - 训练集取 {1, 5, 10, 100} × 旋转增广的 UprightNet 基线 + 你的等变模型
  - 画"训练数据量 vs mean_error"曲线
  - **预期结果**：等变模型在 1× 和 5× 时与 100× baseline 持平；baseline 在 1× 时崩。这是**等变性唯一不可替代的优势**。
- [ ] **5.2 Partial 扫描评测**：
  - 用 `FPS + random crop` 模拟 partial：保留 {25%, 50%, 75%} 点云
  - 或引入 ScanObjectNN（需要人工标注 per-class up）
  - 对比 baseline vs 你的方法在 partial 下的降级曲线
- [ ] **5.3 Noise 鲁棒性**：
  - 加高斯噪声 σ ∈ {0.01, 0.02, 0.05}（点云归一化后的坐标尺度）
  - 对比两者
- [ ] **5.4 UprightNet5 泛化**：
  - 这是 Pang 论文 Table 2 的泛化集
  - 直接对照他们的数字

### Deliverable
4 张鲁棒性曲线图（数据效率 / partial / noise / cross-dataset）。这批图是"为什么等变性值得做"的**最终证据**。

### Go/No-Go 决策点
- 若等变性在 OOD 上也没有明显优势：核心论点需要调整为"更可解释 + 可微" 而非"更鲁棒"。论文标题要避免 "robust"。

---

## 阶段 6 · 消融、可视化与写作（第 9-12 周）

### 任务

- [ ] **6.1 完整消融表**：
  ```
  Baseline (Pang)
  + 可微 SVD (C1)
  + L_Stab (C3)
  + Antipodal Head (C2)
  + all
  + all + 数据效率 regime (1× rotation)
  ```
- [ ] **6.2 等变性验证**：`scripts/verify_equivariance.py` 跑在最终模型上，报告 `max_error ~ 1e-5`。
- [ ] **6.3 可解释性 figure**：每个类别 1 个样本，展示 `soft_weight heatmap + 预测 up + GT up`。
- [ ] **6.4 kappa 标定图**：按预测 κ 分 10 桶，每桶内画 error 分位数（reliability diagram）。对比 baseline（无 κ）就没法画，突显你方法的 **uncertainty quantification** 是额外卖点。
- [ ] **6.5 失败案例分析**：展示你的方法仍然错的 5-10 个样本，诚实归因（多峰合理姿态？数据标签本身模糊？）。这一节对审稿人信任度提升极大。
- [ ] **6.6 相关工作**：写清楚 VN / Frame Averaging / Learned Canonicalization / Implicit-PDF / Matrix Fisher 的位置，说明为什么没选它们作为主方法（对具体任务过重 or 过轻）。
- [ ] **6.7 写作 + 润色**：摘要、intro、method、experiment、conclusion 四轮迭代。

### Deliverable
投稿级别的论文初稿。

---

## 时间表总览

| 周 | 阶段 | 关键里程碑 |
|---|---|---|
| 1 | 0 + 1 | baseline 数字 + 诊断报告 + 工程基建 |
| 2 | 1 + 2 | 等变 L_Stab 跑通 |
| 3 | 2 + 3 | C3 结果确定 + C1 开写 |
| 4 | 3 | 可微 SVD 前向跑通 |
| 5 | 3 | C1 关键消融 (有/无 BCE) 完成 |
| 6 | 4 | Antipodal 头前向跑通 |
| 7 | 4 + 5 | C2 双轴评估完成 + 数据效率实验开写 |
| 8 | 5 | Partial / noise 实验完成 |
| 9 | 5 + 6 | UprightNet5 泛化实验 + 完整消融 |
| 10 | 6 | 所有 figure 完成 |
| 11 | 6 | 论文一稿 |
| 12 | 6 | 润色 + 投稿 |

## 降级路径（论文能写但不能推主刊时）

- **最差情况**：C1 的 (b) 配置失败 + C2 在干净数据上无提升 + 等变性优势只在极小数据量才显现
- **降级叙事**："我们把 Pang 2022 的三个结构性弱点系统性修正，端到端可微实现与原论文 Eq.12 首次实装，并在 1× 旋转增广 regime 下匹配 100× baseline 性能" —— 这已经是一篇 solid 的 workshop 或 BMVC/3DV 级会议论文。
- **两个底线结果**：
  1. **L_Stab 真的补上了**（不管效果多大）
  2. **数据效率实验**：1× 旋转训练 + 等变架构 vs 100× 旋转训练 + UprightNet，等变方优 —— 这是 0 风险的结果。

## 第一周立刻做的三件事

1. 跑 `uprightnet-reference/model/model.pth` test，拿 per-sample error → **阶段 0.1**
2. 修 GT 泄漏 fallback 重跑 → **阶段 0.2**
3. 画 error 直方图 + 挑 top-20 失败样本 → **阶段 0.3-0.4**

**这三件事 2 天内完成。不出结果不做后续任何代码。**

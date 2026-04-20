# 点云直立方向（Gravity/Upright）估计方法调研

> 面向任务：**从无位姿点云中估计 S² 上的一个方向**（重力方向 / 直立向上方向）。
> 现有实现：Upright-Net（Pang et al., CVPR 2022）的 DGCNN+RANSAC 基线 + 基于 e3nn 的 E(3)-等变图网络（球谐、MACE 风格 depthwise-separable tensor product、vMF 头）。
> 本报告聚焦 "如果今天从头做这个任务应该怎么做"，而不是全面的等变文献综述。

---

## 1. 直立方向 / 重力估计本身

针对这个具体任务，公开文献相对稀疏，主要参考点如下：

- **Upright-Net: Learning Upright Orientation for 3D Point Cloud**（Pang, Li, Ding, Zhong，CVPR 2022）。
  架构是 3 层 EdgeConv + 4 层 self-attention + MLP，逐点做二分类（是否为 "支撑点 / 底面点"），再用 RANSAC 对支撑点集拟合平面并取法向，最后用 "法向需指向远离质心" 修正方向；支撑点少于 3 时退化为 "支撑点质心→整体质心" 的连线方向。
  **问题**：该 pipeline 含两个不可微步骤（硬二分类 + RANSAC），对既不是严格 "底面" 也不是单一平面的类别（球体、动物、薄壳类）很脆弱；数据增广要靠全量旋转，训练开销重；CVPR 2022 之后截至目前公开跟进非常少（Google Scholar 正向引用个位数）。
- **PCA / 最小特征向量基线**：对粗糙地面或建筑物扫描，最小主方向往往近似垂直方向；这是任何方法应超越的下界，常在 "室内场景"、"建筑物扫描" 类数据上已经达到几度误差。**对 ModelNet / ShapeNet 类单物体，PCA 会指向 "最瘦的方向"，和重力方向无关**——这正是需要语义先验的场景。
- **OrienNormNet: Orientation Normalization of 3D Body Models**（Zhao 等，3DBODY.TECH 2022）：直接跟进 Upright-Net 到人体三维模型方向归一化。
- **Canonical Pose Estimation** 相关：没有发现明确命名为 "upright estimation" 的 2023+ 新方法；真正的进展都在 "canonicalization"（第 2 节）。

**相关性判断**：Upright-Net 的二分类 + RANSAC 范式是一个清晰但特化的 pipeline。对 "估一个方向" 这种任务来说，它等价于 "预测一个平面的法向 + 解其二义性"。若目标类别天然存在支撑面（家具、交通工具、建筑物），这种归纳偏置合理；否则不该作为首选。

---

## 2. SO(3)-等变点云骨干

这一层选型决定了 "要不要上 l≥2 球谐"。对于**单方向输出**，存在相当大的过度设计空间。

- **Vector Neurons（VN-PointNet / VN-DGCNN；Deng et al., ICCV 2021）**：把标量神经元换成 3D 向量（本质就是 l=1 不可约表示），线性层 / 非线性 / 池化 / 归一化全部重写；不需要球谐、不需要 Clebsch-Gordan。
  **相对 e3nn：** 实现极简（几百行 PyTorch），速度快，只有 l=1 channels，表达力弱于高阶张量网络，但对 "预测一个方向" **刚好够用**——方向本身就是 l=1 的量。
  **对本任务的定位：这是最对症的 "下一步简化方向"**。
- **Tensor Field Networks（Thomas et al., 2018）**：首个点云上的 SE(3)-等变架构，用球谐作为滤波器，l≥2，非 message-passing；计算昂贵，后续被 SE(3)-Transformer / e3nn 框架取代。
- **SE(3)-Transformer（Fuchs et al., NeurIPS 2020）**：在 TFN 基础上加等变自注意力；对 QM9、N-body 等任务有效；对一个方向的回归属于 overkill。
- **EPN / E2PN（Chen et al., 2021/2022）**：离散化 SO(3)（商群 S² × SO(2)）的卷积式等变网络；工程复杂度高，主要用于形状配准和类级位姿估计。
- **Equivariant Message Passing（Brandstetter et al., ICLR 2022, "Geometric and Physical Quantities Improve E(3) Equivariant Message Passing"）**：和 MACE 一脉相承，在 OC20 / QM9 上有效；用于点云分类是 overkill。
- **MACE（Batatia et al., NeurIPS 2022）**：高阶（4-body）消息 + 只需 2 层 message passing；现有实现借鉴其 **depthwise-separable tensor product**，这是 MACE 的工程精髓（核心其实就是 "channel-wise TP + atomic basis"），放到点云问题里能显著降低 TP 的通道耦合开销。
- **EquiformerV2（Liao et al., ICLR 2024）**：把 SO(3) convolution 替换为 **eSCN convolution**，引入 **separable S² activation**（把 l≥2 张量投射到球面网格再做逐点非线性），使高阶等变 Transformer 可扩展；其 S² 激活目前也是代码借鉴来源。对 "一个方向" 的任务，真正受益的只有 l=1 通道，高阶 S² 激活的价值主要在内部中间表示。
- **Frame Averaging（Puny et al., ICLR 2022）**：**非架构等变**——选一小组 "frame"（如点云的 PCA frame，最多 8 个反射歧义），对每个 frame 做前向并平均，就能得到严格等变 / 不变。在点云法向估计任务上已达到 SOTA。
  **对本任务的定位：强竞争者**。用 PCA frame + 非等变骨干（PointNet++/DGCNN），做 4~8 次前向平均，即可得到严格 SO(3) 等变的方向输出。实现成本极低，性能在大多数基准上与 e3nn 持平。
- **Canonical Capsules（Sun et al., NeurIPS 2021）**：自监督方式在成对随机旋转点云间学一致的胶囊分解，产出 canonical pose；**它就是一个 direction/pose 回归器的前置模块**。
- **Compass（Spezialetti et al., ECCV 2020）**：自监督学 SHOT-like LRF（local reference frame），可用作基线。
- **Equivariance via Learned Canonicalization（Kaba et al., ICML 2023, "Equivariance with Learned Canonicalization Functions"）**：显式学一个 "canonicalization net"（输出一个旋转），把输入归一化后喂给普通网络。**这和本任务的输出目标几乎完全一致**：只是把 "canonicalize" 的旋转的第三行当作直立方向即可。

**相关性判断**：
- **Vector Neurons / Frame Averaging / Learned Canonicalization 是三条最对症的替代路线**；TFN / SE(3)-Transformer / EPN / Equiformer 对本任务是 overkill。
- 目前 e3nn + MACE + EquiformerV2 借鉴的架构堆栈，在 "l=2 球谐 + S² 激活" 这一块给任务加了显著复杂度，但最终输出只取 l=1 通道，**高阶张量的信息收益在分类 / 分割类任务上才显现**。

---

## 3. 概率方向 / 旋转分布

对 "方向 + 置信度" 的建模，关键问题是：**S²（无符号）** vs **RP²（带对跖对称）** vs **SO(3)（完整旋转）**。

- **Deep Directional Statistics（Prokudin, Gall, Leibe, ECCV 2018）**：用神经网络预测 von Mises-Fisher (vMF) 参数做方向回归；训练目标是对 vMF 分布的 NLL。对 **S² 上无对跖对称** 的方向，vMF 是正解。
- **Deep Bingham / A Smooth Representation of Belief over SO(3)**（Peretroukhin et al., RSS 2020）：用对称矩阵参数化 Bingham 分布（单位四元数上的 antipodal-symmetric 分布），天然处理 q ≡ -q 的对称性；用于 SO(3) 旋转回归时平滑性比 quaternion 直接回归更好。
- **Implicit-PDF（Murphy et al., ICML 2021）**：**非参数**方式建模 SO(3) 上的 PDF——把候选旋转 R 和 query feature 送进 MLP 输出 log p(R|x)，训练只需 ground-truth R，推理靠网格采样 / 梯度上升。优势：能表达多模态（对称物体），无分布族假设。
- **Matrix Fisher（Mohlin, Sullivan, Bianchi, NeurIPS 2020）**：SO(3) 上的另一种自然分布族，用 3×3 矩阵 F 作参数；比 Bingham 更易实现，对 "单峰 + 单位置信度" 够用。
- **SVD orthogonalization（Levinson et al., CVPR 2020）**：深度学习里预测 SO(3) 时推荐直接回归 3×3 矩阵并做 SVD 投影，收敛性优于 quaternion / Euler。

**关于当前实现中 vMF 的选择**：
- 严格来说 "直立方向" 在物理上是**有向**的（重力指向地面，不是指天花板），所以 vMF（无对跖对称）是正确的；Upright-Net 最后那步 "指向远离质心" 正是在消这个对跖歧义。
- 但如果训练标签定义不清（有些数据集把 "up" 和 "down" 混着标），或者类别天然具有 180° 旋转对称（如哑铃、花瓶、车轮），就该考虑：
  - **Watson 分布 / Angular Central Gaussian (ACG)**：对 antipodal-symmetric 情形的 S² 自然分布；
  - **Bingham on S²** = ACG 的高维推广；
  - **Implicit-PDF 的 S² 版本**（MLP 输入方向 v + feature，输出 log p(v)，Möbius/marginal 下对对称物体自然多峰）。

**相关性判断**：当前 vMF 头对 "干净的单峰重力方向" 是合理默认；对 antipodally-symmetric 类别需要换 Watson/ACG；对 "多种合理直立姿态" 的类别（如一根圆柱可以上下任意倒置）需要 Implicit-PDF 风格的非参数分布。

---

## 4. 更简更便宜的替代方案

对 "预测一个方向" 的任务，以下任一方案都极可能匹配甚至超越当前实现：

1. **PCA / 协方差最小特征向量基线**（零参数）：在扫描场景 / 有明显地面的数据上，这是一个不该被超越就报警的下界。
2. **PointNet++ / DGCNN + 3 维向量直接回归 + TTA 旋转平均**：
   - 预测 μ ∈ R³，归一化后作为方向；
   - 推理时对输入做 K 个随机旋转 R_k，把每次预测 R_k⁻¹ μ_k 平均。
   - 成本：标量骨干 + K× 推理 + 均值；工程复杂度远低于 e3nn。
3. **Frame Averaging + 标准非等变骨干**：PCA frame（最多 8 个反射歧义）+ DGCNN；**严格等变**，无需等变层，无需数据增广。
4. **Learned Canonicalization**（Kaba et al. ICML 2023）+ 任何回归头：学一个 canonicalization net 输出 R，训练时用不变损失。
5. **Vector Neurons DGCNN**：如果必须架构等变，这是最简方案；l=1 足够表达方向，不上球谐。

**为什么 l=2 球谐对 "一个方向" 几乎没必要**：
- 输出目标是 l=1 的一个向量；
- 在网络内部，l=2 通道只能通过 TP(l=1, l=1) → l=2 形成，再在下一层通过 TP(l=2, l=1) → l=1 回流到输出；
- 对于只想从 "点云整体形状" 映射到 "一个方向" 的任务，2~3 层 l=1-only 消息（即 VN 风格）的通用近似能力已经够用；高阶张量的主要价值在于 "需要输出高阶量"（应力张量、极化张量）或 "需要精细区分几何" 的分类 / 分割。

---

## 5. 近期（2023–2025）相关工作

- **EquiformerV2**（Liao et al., ICLR 2024, arXiv:2306.12059）：可扩展高阶等变 Transformer；分子力场主场，点云单方向任务 overkill，但 S² 激活思路可复用。
- **Equivariance with Learned Canonicalization Functions**（Kaba et al., ICML 2023, arXiv:2211.06489）：显式学 canonicalization，**是本任务的最直接现代方案之一**。
- **Equivariant Adaptation of Large Pretrained Models**（Mondal et al., NeurIPS 2023, arXiv:2310.01647）：在 Kaba 思路上加 "dataset-dependent priors"，让 canonicalization 输出与预训练分布对齐；对 "用大预训练骨干 + 一个小 canonicalization 头" 的做法提供了现成范式。
- **Banana: Banach Fixed-Point Network for Pointcloud Segmentation with Inter-Part Equivariance**（arXiv:2305.16314, NeurIPS 2023）：对多部件、铰接物体的等变处理；对 "每个对象一个 up 方向" 不直接相关，但对铰接体数据需要时可参考。
- **Point Cloud Canonicalization via self-supervision**：2023-2024 有若干小工作尝试在无标注下学 canonical frame，基本延续 Canonical Capsules + Compass 的思路，未看到明显超越。
- **GigaPose**（arXiv:2311.14155, CVPR 2024）：新物体 6D pose 估计，思路是 "模板 + patch correspondence"，对 "已知 CAD 的 up 估计" 可以退化使用但 overkill。
- **SO(3) / rotation regression 方向**：近两年主流工作聚焦于 6D pose（含平移）和多模态分布（Implicit-PDF 系），单方向 / 单旋转回归本身已经相对稳定，没出现颠覆性新方法。

---

## 总结与建议 —— "今天从头做应该怎么做"

按 **性价比 + 工程简洁度** 排序，推荐如下几种可落地方案：

### A. 首选：Vector Neurons + vMF 头（简洁、严格等变、对症）
- 骨干：**VN-DGCNN**（Deng et al., ICCV 2021）。只有 l=1，代码量 ~几百行 PyTorch，不依赖 e3nn。
- 头：vMF（μ, κ），若类别含 antipodal 对称则换 Watson / Angular Central Gaussian。
- 理由：输出就是 l=1 的向量；高阶球谐对本任务无可证明收益；训练显存和延迟比当前实现小一个数量级。

### B. 强替代：Frame Averaging + 任意非等变骨干
- 骨干：**DGCNN / PointNet++ / Point Transformer**（任选，可直接用预训练权重）。
- 等变机制：**Frame Averaging（Puny ICLR 2022）**，用 PCA 主轴构造 ≤8 元 frame，前向平均。
- 头：3 维向量回归 + vMF 标定（后验估 κ）。
- 理由：严格 SO(3) 等变，但 "只改 wrapper，不改 backbone"；非常容易复现，且可以无损用预训练骨干的工业特性。

### C. 保留 e3nn 高阶网络的唯一充分理由
- 若需要同时输出多个几何量（方向 + 局部 frame + 法向场 + ...），或者数据分布本身有丰富的各向异性结构（蛋白质、晶体、复杂铰接物体），l=2 球谐 + MACE/EquiformerV2 风格的堆叠才有投入产出比。
- 对 "只估一个方向" 的纯任务，建议作为 **research 基线** 保留，但不作为主模型推上生产 / 实验对比。

### D. 概率头的选型（独立于骨干）
- 单峰、**有向**方向（重力、法向"outward side 已定"）：**vMF**。
- 单峰、**无向 / antipodal 对称**（车轮、桌腿朝上或朝下）：**Watson / ACG**，或对称化的 Bingham。
- 多峰 / 语义歧义（花瓶可能任意立起）：**Implicit-PDF（Murphy ICML 2021）** 在 S² 上的变种；这是目前唯一能自然表达多模态的方案。
- 完整 SO(3) 姿态：**Matrix Fisher（Mohlin NeurIPS 2020）** 或 **Deep Bingham（Peretroukhin RSS 2020）**，或直接回归 3×3 + SVD（Levinson CVPR 2020）。

### E. 必跑的基线（任何方法都要超越它们）
1. **纯 PCA**（最小 / 最大特征向量，带符号修正）。
2. **DGCNN + 3-vector 回归 + 旋转 TTA**（K=4～8 次旋转平均）。
3. **类别先验**：如果测试数据类别分布已知，"每类固定 up 方向" 的类先验基线往往惊人地难超越。

### 关键结论
**当前基于 e3nn + MACE + EquiformerV2 的实现，对 "单方向估计" 任务是架构过剩的。** 如果目标是更好的准确率 / 置信度标定，真正 ROI 高的改造是：
1. 把骨干换为 VN-DGCNN 或 Frame-Averaged DGCNN（A 或 B），保留 vMF/Watson 头；
2. 对模糊类别引入 Implicit-PDF 风格的非参数分布；
3. 强化基线对比——至少把 PCA 和 DGCNN+TTA 两条基线跑到底，再谈 "等变网络值不值得"。

### 参考文档指引
- **e3nn**：官方文档 https://docs.e3nn.org ，核心对象是 `Irreps`、`TensorProduct`、`SphericalHarmonics`。
- **MACE**：https://github.com/ACEsuit/mace ，关键模块是 `InteractionBlock` 中的 channel-wise TP。
- **VN-DGCNN**：https://github.com/FlyingGiraffe/vnn-pc ，推荐作为本任务骨干替换的起点。
- **Frame Averaging**：https://github.com/omri1348/Frame-Averaging ，点云法向估计分支直接可用。
- （注：仓库内 context7 配额本次调用已耗尽，未获取到 e3nn/MACE/VN 的结构化文档条目；以上链接为公开官方仓库。）

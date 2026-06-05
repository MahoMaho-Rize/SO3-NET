# 残缺点云层级分割训练流程

## 核心定义

当前主线不是候选方向二分类，而是点级层级分割：

```text
partial point cloud
  -> per-point bottom-to-top hierarchy labels
  -> per-point hierarchy logits
  -> upright direction from predicted hierarchy ordering
```

原始 UprightNet 的输出是 `(B, 1, N)` 支撑点概率。这里扩展为：

```text
level_logits: (B, K, N)
```

`K` 是层级数，默认 `5`。标签只定义在可见点上；不可见底面不会被强行补点。

## 层级标注

脚本：

```bash
python3 scripts/build_hierarchy_npz.py \
  --input-root datasets/uprightnet15_partial_camera \
  --full-root datasets/uprightnet15 \
  --out-dir datasets/upright_hierarchy_npz \
  --num-points 2048 \
  --num-levels 5 \
  --source-up-axis z
```

对每个 partial OFF：

1. 读取可见点。
2. 找到对应完整 OFF mesh。
3. 用完整物体在 canonical up 轴上的 bbox 高度归一化每个可见点。
4. 将归一化高度量化为 `0..K-1` 的层级标签。

这样即使底部完全缺失，最低可见点也不会被错误标成真实底层。

## 训练

```bash
python3 scripts/train_hierarchical_uprightnet.py \
  --train-npz datasets/upright_hierarchy_npz/train.npz \
  --test-npz datasets/upright_hierarchy_npz/test.npz \
  --out-dir models/hierarchical_uprightnet \
  --epochs 80 \
  --batch-size 128 \
  --lr 1e-3 \
  --num-workers 8 \
  --device cuda \
  --class-balance
```

训练时对点云做随机 SO(3) 旋转，点级层级标签不变，GT upright 同步旋转。主损失是：

```text
CrossEntropy(level_logits, level_labels)
```

## 从层级预测恢复方向

模型输出 `level_logits` 后先做 softmax，得到每个点属于各层级的概率。每个点的预测层级期望为：

```text
s_i = sum_k p(level=k | point_i) * k / (K - 1)
```

如果层级预测正确，`s_i` 会沿物体 upright 方向单调增加。因此方向由点坐标和层级分数的一阶协方差给出：

```text
u = normalize(sum_i (s_i - mean(s)) * (x_i - mean(x)))
```

这个估计不要求真实底面可见。底层点可见时，它接近“底到顶”的方向；底层缺失时，它仍然可以利用可见下部、中部、上部的相对排序恢复方向。

## 一键流程

本地有 Blender 时：

```bash
DEVICE=cuda EPOCHS=80 BATCH_SIZE=128 \
  ./scripts/run_modelnet15_hierarchy_pipeline.sh
```

远端没有 Blender、已有 partial OFF 时：

```bash
SKIP_DOWNLOAD=1 SKIP_PARTIAL=1 DEVICE=cuda \
  ./scripts/run_modelnet15_hierarchy_pipeline.sh
```

输出：

```text
datasets/upright_hierarchy_npz/train.npz
datasets/upright_hierarchy_npz/test.npz
models/hierarchical_uprightnet/best.pth
models/hierarchical_uprightnet/final.pth
models/hierarchical_uprightnet/train_log.csv
```

评估日志同时报告点级分割质量和最终直立方向质量：

```text
point_acc, mIoU, mean_err, median_err, acc@5, acc@10, acc@30, flip>90
```

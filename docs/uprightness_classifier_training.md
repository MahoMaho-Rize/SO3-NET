# Candidate Uprightness Classifier 训练说明

## 1. 目标

这个最小版本实现的不是方向回归，也不是球面方向离散化分类。它训练的是一个真正的二分类器：

```text
C(P, h) -> h 是否是点云 P 的有效 upright hypothesis
```

其中：

```text
P: partial point cloud
h: 候选 upright direction
label = 1: h 是真实直立解释
label = 0: h 是错误解释
```

训练时会把点云旋到候选坐标系：

```text
P_h = Align(h -> +Y) @ P
```

模型看到的是 `P_h`，任务是判断“在这个候选姿态下，物体看起来是否是直立的”。

## 2. 文件

新增文件：

```text
scripts/build_uprightness_npz.py
scripts/train_uprightness_classifier.py
docs/uprightness_classifier_training.md
docs/partial_uprightnet_true_classification_loss.md
```

## 3. 生成 partial 点云

一步到位方式：

```bash
DEVICE=cuda EPOCHS=80 BATCH_SIZE=128 \
  ./scripts/run_modelnet15_uprightness_pipeline.sh
```

这个脚本会自动：

```text
下载官方 ModelNet40 OFF 数据
抽取 bed / bench / bottle / bowl / car / chair / cone / cup / lamp / monitor / sofa / stool / table / toilet / vase
修复 inline OFF header
生成单视角 partial OFF
打包 train/test NPZ
启动 uprightness 二分类训练
```

常用覆盖参数：

```bash
MODELNET40_URL=http://modelnet.cs.princeton.edu/ModelNet40.zip
PYTHON=python
BLENDER=blender
VIEWS_PER_MODEL=8
EPOCHS=80
BATCH_SIZE=128
SAMPLES_PER_CLOUD=20
DEVICE=cuda
NUM_WORKERS=8
```

只准备数据不训练：

```bash
SKIP_TRAIN=1 ./scripts/run_modelnet15_uprightness_pipeline.sh
```

下面是分步方式。

如果远程服务器上还没有 partial OFF 数据，先用 Blender 从完整 UprightNet15 OFF mesh 生成：

```bash
blender --background --python scripts/blender_partial_uprightnet15.py -- \
  --input-root datasets/uprightnet15 \
  --output-root datasets/uprightnet15_partial_camera \
  --views-per-model 8 \
  --output-count 2048 \
  --depth-width 128 \
  --depth-height 128 \
  --max-depth-size 512
```

输出结构应类似：

```text
datasets/uprightnet15_partial_camera/<category>/<train|test>/<name>_viewXX.off
```

## 4. 打包为训练 NPZ

```bash
python scripts/build_uprightness_npz.py \
  --input-root datasets/uprightnet15_partial_camera \
  --out-dir datasets/uprightness_partial_npz \
  --num-points 2048 \
  --source-up-axis z
```

生成：

```text
datasets/uprightness_partial_npz/train.npz
datasets/uprightness_partial_npz/test.npz
```

NPZ 只保存 source partial clouds 和元数据，不展开候选正负样本。候选会在训练时在线生成。

## 5. 开始训练

CPU/GPU 自动选择：

```bash
python scripts/train_uprightness_classifier.py \
  --train-npz datasets/uprightness_partial_npz/train.npz \
  --test-npz datasets/uprightness_partial_npz/test.npz \
  --out-dir models/uprightness_classifier \
  --epochs 80 \
  --batch-size 128 \
  --samples-per-cloud 20 \
  --lr 1e-3 \
  --num-workers 8
```

指定 GPU：

```bash
python scripts/train_uprightness_classifier.py \
  --train-npz datasets/uprightness_partial_npz/train.npz \
  --test-npz datasets/uprightness_partial_npz/test.npz \
  --out-dir models/uprightness_classifier \
  --epochs 80 \
  --batch-size 128 \
  --samples-per-cloud 20 \
  --lr 1e-3 \
  --num-workers 8 \
  --device cuda
```

输出：

```text
models/uprightness_classifier/best.pth
models/uprightness_classifier/final.pth
models/uprightness_classifier/train_log.csv
```

## 6. 在线候选构造

当前最小版本每个 source partial cloud 在线构造以下候选：

| Candidate | Label | 目的 |
|---|---:|---|
| `pos` | 1 | GT upright 加小扰动 |
| `flip` | 0 | `-GT`，专门打击 180° 翻转 |
| `tilt` | 0 | 大角度倾斜错误 |
| `random` | 0 | 普通随机错误方向 |
| `pca` | 0 | PCA 诱导的错误轴 |

主损失：

```text
L = BCEWithLogits(upright_logit, valid_label)
```

辅助项：

```text
L_vis = CE(visibility_logits, support_visible_state)
```

总损失：

```text
L_total = L + vis_weight * L_vis
```

默认：

```text
pos_weight = 4.0
vis_weight = 0.2
```

## 7. 这个版本还没有做的事

这是最小可训练版本，先验证“真正分类”路线能否压低 partial 场景的 flip 错误。它还没有实现：

```text
原 UprightNet/RANSAC failed proposal 作为 hard negative
可见局部平面法向作为 hard negative
pairwise BCE ranking loss
完整 inference candidate search
```

下一步应该优先补 hard negative proposal 生成，因为当前 partial 失败主要来自错误可见平面，而不是纯随机方向。

# Hierarchical DGCNN Partial UprightNet Result

## Experiment

This run trains the hierarchical point-wise segmentation model on camera-style
partial UprightNet15 point clouds.

Checkpoint inspected locally:

```text
final.pth
sha256=df229817946f6d49455790fa00d9a3c89151c7fd74b5c8037c13f48843eb9dd5
```

Checkpoint contents:

```text
keys: model, args, num_levels
arch: dgcnn
num_levels: 5
num_points: 2048
batch_size: 64
epochs: 80
class_balance: true
train_npz: datasets/upright_hierarchy_npz/train.npz
test_npz: datasets/upright_hierarchy_npz/test.npz
```

The final layer is `conv7.weight` with shape `(5, 64, 1)`, confirming that the
model outputs 5 point-wise hierarchy classes.

## Final Epoch Metrics

From `train_hierarchical_dgcnn.log`, epoch 80:

| Model | Data | Point Acc | mIoU | Mean | Median | Acc@10 | Flip |
|---|---|---:|---:|---:|---:|---:|---:|
| Hierarchical DGCNN | partial test | 89.83% | 81.28% | 5.97 deg | 3.07 deg | 88.46% | 0.59% |

## Baseline Comparison

| Method | Test Data | Mean | Median | Acc@10 | Flip |
|---|---|---:|---:|---:|---:|
| Original UprightNet | full UprightNet15 | 11.17 deg | 0.70 deg | 93.03% | 5.74% |
| Original UprightNet | partial, with fallback | 86.05 deg | 90.19 deg | 19.03% | 51.96% |
| Original UprightNet | partial, no fallback | 98.04 deg | 91.24 deg | 7.75% | 59.20% |
| Hierarchical DGCNN | partial | 5.97 deg | 3.07 deg | 88.46% | 0.59% |

## Interpretation

The hierarchical model is trained and evaluated on partial point clouds.  The
full UprightNet15 meshes are only used during dataset construction to compute
canonical object height ranges for visible-point hierarchy labels.

Compared with original UprightNet on the partial test set, the hierarchical
DGCNN reduces flip rate from 51.96% to 0.59% and improves Acc@10 from 19.03% to
88.46%.  Remaining error is primarily small-to-medium angular precision rather
than upright polarity flips.

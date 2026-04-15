# E(3)-Equivariant Upright Orientation Estimation for 3D Point Clouds

An E(3)-equivariant approach to upright orientation estimation, replacing the traditional classification + RANSAC pipeline with an end-to-end differentiable framework built on spherical harmonic features and probabilistic directional modeling.

## Key Ideas

**Problem**: Estimate the upright (gravity) direction of a 3D point cloud.

**Prior approach** (Upright-Net, CVPR 2022): DGCNN classifies per-point support labels, then RANSAC fits a plane, and the plane normal gives the upright direction. This pipeline is non-differentiable, requires heavy rotation augmentation (100x per object), and provides no uncertainty estimate.

**Our approach**: An E(3)-equivariant graph neural network with spherical harmonic edge features directly outputs the upright direction as an *l*=1 equivariant vector, paired with a von Mises-Fisher (vMF) probabilistic head for calibrated uncertainty. The architecture guarantees rotational consistency by construction, eliminating the need for rotation augmentation.

### Core contributions

1. **Equivariant-probabilistic coupling**: The upright direction *mu* comes from *l*=1 (vector) features that co-rotate with the input; the confidence *kappa* comes from *l*=0 (scalar) features that are rotation-invariant. This separation arises naturally from the irreducible representation structure of E(3).

2. **End-to-end differentiable**: No RANSAC, no post-processing. Gradients flow from the geodesic/vMF loss all the way back through the geometric reasoning layers.

3. **MACE-style depthwise separable tensor products**: Decouples channel mixing (shared Linear layers) from angular coupling (lightweight "uvu"-mode CG tensor products), reducing per-edge computation by ~18x while preserving exact equivariance.

## Architecture

```
Input: Point cloud (N, 3)
  |
  v
[radius_graph]  -->  edge_vec (E, 3)
  |                      |
  |              [spherical_harmonics Y^l]  -->  edge_sh (E, 1+3+5)
  |                      |
  v                      v
[Scatter init]  -->  node_features (N, hidden_dim)
  |
  v
[EfficientConvLayer x num_layers]
  |   Linear_up (channel mixing, shared)
  |   DepthwiseTP (CG coupling, "uvu" mode, ~192 weights/edge for L=2)
  |   Scatter aggregation
  |   Linear_down (channel mixing, shared)
  |   Gate nonlinearity + skip connection
  |
  v
node_features (N, hidden_dim)
  |
  +---> [VMFDirectionHead]  -->  mu (B, 3): upright direction (equivariant)
  |         global pool l=1        kappa (B, 1): confidence (invariant)
  |
  +---> [SupportHead]  -->  support_prob (N,): per-point support probability
            extract l=0              (invariant, auxiliary task)
```

**Irreducible representations used**:
- *l*=0 (scalars, dim=1): density, confidence, support probability
- *l*=1 (vectors, dim=3): directions, normals, the upright direction itself
- *l*=2 (rank-2 tensors, dim=5): curvature, planarity, anisotropy (optional)

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- e3nn >= 0.5.0
- torch-geometric >= 2.3
- torch-cluster, torch-scatter
- numpy, scikit-learn, tqdm

Install dependencies:
```bash
pip install -r requirements.txt
```

For torch-cluster and torch-scatter, match your PyTorch + CUDA version:
```bash
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html
```

## Dataset

**UprightNet15**: 15 object categories with upright standing pose, derived from [ModelNet40](https://modelnet.cs.princeton.edu/). Points with y < 0.05 are annotated as support points. Each model is rotated 100 times for the original method; for the equivariant model, rotation augmentation is optional (equivariance guarantees generalization).

Place `.npy` data files in `./datasets/uprightnet15/`.

## Usage

### Training the equivariant model

**L=1 configuration** (fastest, ~6 ms/inference):
```bash
python train.py --network equivariant --conv_type depthwise \
    --irreps_hidden "32x0e+16x1o" --lmax 1 \
    --loss_type vmf --use_rotation_aug False \
    --epoch 15 --batch_size 8 --learning_rate 0.001
```

**L=2 configuration** (richer angular features, ~51 ms/inference):
```bash
python train.py --network equivariant --conv_type depthwise \
    --irreps_hidden "32x0e+16x1o+8x2e" --lmax 2 \
    --loss_type vmf --use_rotation_aug False \
    --epoch 15 --batch_size 8 --learning_rate 0.001
```

### Testing

```bash
python test.py --network equivariant --conv_type depthwise \
    --irreps_hidden "32x0e+16x1o" --lmax 1 \
    --model_file equivariant_best_YYYYMMDD-HHMM.pth
```

### Training the original UprightNet (baseline)

```bash
python train.py --network uprightnet
```

### Verifying equivariance

```bash
python scripts/verify_equivariance.py --network equivariant --conv_type depthwise
```

This tests f(Rx) = Rf(x) for random rotations R. Expected error: ~1e-5 (floating-point precision).

## Key Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--network` | `uprightnet` | `uprightnet` (original) or `equivariant` (ours) |
| `--conv_type` | `depthwise` | `depthwise` (MACE-style, efficient) or `fctp` (full tensor product) |
| `--lmax` | `2` | Max spherical harmonic degree (1 or 2) |
| `--irreps_hidden` | `32x0e+16x1o+8x2e` | Hidden layer irreducible representations |
| `--equi_layers` | `4` | Number of equivariant message passing layers |
| `--loss_type` | `vmf` | `vmf` (von Mises-Fisher NLL) or `geodesic` (arc-length on S^2) |
| `--use_rotation_aug` | `True` | Rotation augmentation; set `False` for equivariant model |
| `--max_radius` | `0.5` | Radius for graph construction |
| `--beta` | `0.1` | Weight of auxiliary support point BCE loss |

## Performance

Single-sample inference latency (2048 points, single GPU):

| Method | Latency | FPS |
|--------|---------|-----|
| Original UprightNet | 5.6 ms | 178 |
| **L=1 depthwise (ours)** | **6.0 ms** | **166** |
| L=2 depthwise (ours) | 50.7 ms | 20 |

Training throughput with equivariance advantage (no rotation augmentation needed):

| Method | Data per epoch | Throughput | Estimated total training |
|--------|---------------|------------|------------------------|
| Original UprightNet | N x 100 rotations | 161 samp/s | ~86 hours |
| L=1 depthwise (ours) | N (no augmentation) | 61 samp/s | ~41 min |
| L=2 depthwise (ours) | N (no augmentation) | 10.8 samp/s | ~3.9 hours |

## Project Structure

```
uprightnet/
├── config.py                          # All configuration (original + equivariant)
├── train.py                           # Training entry point
├── test.py                            # Testing entry point
├── model.py                           # Model class (train/test logic for both architectures)
├── network.py                         # Original UprightNet (DGCNN + Attention)
├── network_equivariant.py             # E(3)-equivariant network (e3nn-based)
│   ├── EquivariantConvLayer           #   Full CG tensor product convolution (for ablation)
│   ├── EfficientConvLayer             #   MACE-style depthwise separable convolution
│   ├── VMFDirectionHead               #   von Mises-Fisher probabilistic direction output
│   ├── SupportHead                    #   Per-point support classification (auxiliary)
│   └── EquivariantUprightNet          #   Main network assembling all components
├── Common/
│   ├── EquivariantDataLoader.py       # PyG-format data loader
│   ├── loss_equivariant.py            # Geodesic loss, vMF NLL, auxiliary BCE
│   ├── geometric_utils.py             # Angular error, vMF utilities, differentiable SVD
│   ├── RobustPointSetDataLoader.py    # Original data loader
│   ├── loss_utils.py                  # Original BCE + FR loss
│   ├── data_utils.py                  # Point cloud augmentations
│   └── point_operation.py             # Point cloud utilities
├── scripts/
│   └── verify_equivariance.py         # Equivariance numerical verification
├── requirements.txt                   # Dependencies
├── model/                             # Saved model weights
└── datasets/uprightnet15/             # Dataset directory
```

## Acknowledgments

This project builds upon:
- [Upright-Net](https://openaccess.thecvf.com/content/CVPR2022/html/Pang_Upright-Net_Learning_Upright_Orientation_for_3D_Point_Cloud_CVPR_2022_paper.html) (Pang et al., CVPR 2022) for the problem formulation and dataset
- [e3nn](https://e3nn.org/) for E(3)-equivariant neural network primitives
- [MACE](https://github.com/ACEsuit/mace) for the depthwise separable tensor product architecture

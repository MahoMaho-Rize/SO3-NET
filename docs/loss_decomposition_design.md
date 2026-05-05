# Loss Decomposition Design: Axis Regression + Polarity Classification

> Status: Design Draft (2026-04-30)
> Branch: `shs-style-signed-direction`
> Predecessor: `network_shs.py` (SHS-style signed direction baseline)

---

## 1. Motivation

### 1.1 Current Problem

The current `SHSUprightNet` uses a single signed geodesic loss to regress the
signed upright direction end-to-end.  While this already beats the Pang 2022
baseline on flip rate (4.05% vs 5.72%) and antipodal Acc@10 (93.8% vs 93.05%),
two issues remain:

1. **Median error regression** (1.46 vs 0.71): the network is less precise
   on easy samples than RANSAC.
2. **lamp/bed flip rate still ~16%**: the hardest categories haven't improved.

### 1.2 Root Cause Analysis

The signed geodesic loss `arccos(cos)` asks the network to simultaneously learn:
- (a) the correct axis (geometric, local)
- (b) the correct polarity / sign (semantic, global)

These two sub-tasks have conflicting gradient dynamics:
- Near 0 error (easy samples): gradient is strong, axis branch dominates.
- Near 180 error (flip cases): `arccos` gradient vanishes at `cos = -1`,
  so the sign-correction signal is weakest exactly where it's most needed.

### 1.3 Evidence from SHS-Net Ablation

SHS-Net (CVPR 2023) showed that decomposing oriented normal estimation into
`sin loss` (axis accuracy) + `sign BCE` (orientation classification) improved
oriented RMSE by 40% (33.29 -> 19.79) on the PCPNet benchmark.  Removing this
decomposition was the **second largest source of degradation** in their
ablation, after removing the shape encoder entirely.

---

## 2. Proposed Loss Design

### 2.1 Network Output Decomposition

```
SHSUprightNet outputs:
  |
  +-- axis_branch  -->  n in S^2     (unsigned axis, 3D unit vector)
  |                     trained with antipodal geodesic loss
  |
  +-- sign_branch  -->  p in (0,1)   (polarity probability)
  |                     trained with binary cross-entropy
  |
  +-- support_branch -> w in [0,1]^N (per-point support probability)
  |                     trained with BCE (existing)
  |
  Final signed direction:
      up = n * sign(p)              (inference)
      up = n * (2p - 1)             (training, soft version for gradient flow)
```

### 2.2 Loss Components

#### L_axis: Antipodal Geodesic Loss (Axis Accuracy)

```
L_axis = arccos(|<n, y_up>|)
```

- Uses **absolute** cosine similarity (antipodal identification).
- The axis branch only needs to find the correct line through the origin;
  it is explicitly freed from the polarity burden.
- This should recover the baseline's low median error on easy samples,
  since axis estimation is a purely geometric task that local features
  handle well.

#### L_sign: Binary Cross-Entropy (Polarity Classification)

```
n_bar = n.detach()                          # stop gradient
sign_gt = (<n_bar, y_up> > 0).float()      # 1 = keep, 0 = flip
L_sign = BCE(p, sign_gt)
```

Key design choices:

1. **Detach n before computing sign GT.**  Without detach, the sign loss
   gradient would flow back into the axis branch, causing the axis to
   distort itself to make sign classification easier.  This is the same
   decoupling principle as SHS-Net's separate sin/sgn losses.

2. **GT derived at runtime, not pre-annotated.**  The sign label is
   computed from the relationship between the predicted (detached) axis
   and the GT direction.  No additional annotation required.

3. **BCE gradient is strongest at p = 0.5** --- precisely the uncertain
   (flip-prone) samples.  This directly addresses the vanishing gradient
   problem of signed geodesic loss at 180 error.

#### L_sup: Support Point BCE (Auxiliary, Existing)

```
L_sup = BCE(w, y_support)
```

Unchanged from the current implementation.  Provides geometric grounding
for the trunk features.

#### L_stab: Physical Stability Prior (Self-Supervised)

```
c_mass = mean(points)                       # (B, 3) mass center
c_sup  = sum(w * points) / sum(w)           # (B, 3) support center
gap    = <up, c_mass - c_sup>               # scalar: height of mass center
                                            #   above support center along up
L_stab = relu(-gap)                         # penalize when mass center is
                                            #   BELOW support center
```

Physical interpretation:
- An object standing upright has its mass center **above** its support
  region (along the gravity/up direction).
- If the predicted `up` direction places the mass center below the
  support center, the object would be "floating" --- physically unstable.
- This directly penalizes the lamp-shade-as-base failure mode: when
  the network flips a lamp, the heavy base ends up "above" the thin
  shade, violating stability.

Key properties:
- **Self-supervised**: requires no additional annotation beyond the
  support point labels already in the dataset.
- **Equivariant-compatible**: the stability criterion is rotation-
  invariant (dot product of two co-rotating vectors).
- **Sparse activation**: only fires on incorrect flips; zero loss
  when the prediction is physically plausible.

### 2.3 Total Loss

```
L = lambda_1 * L_axis + lambda_2 * L_sign + lambda_3 * L_sup + lambda_4 * L_stab
```

Recommended initial hyperparameters:

| Parameter  | Value | Rationale |
|------------|-------|-----------|
| lambda_1   | 1.0   | Primary objective (axis accuracy) |
| lambda_2   | 0.5   | Sign classification, secondary |
| lambda_3   | 0.1   | Auxiliary (same as current) |
| lambda_4   | 0.2   | Physics prior, soft regularization |

---

## 3. Supervision Signal Summary

| Signal | Source | Annotation Cost | Used By |
|--------|--------|----------------|---------|
| y_up (GT upright direction) | Dataset (existing) | Zero | L_axis, L_sign |
| y_support (support points) | Dataset (existing) | Zero | L_sup, L_stab |
| sign GT | Computed from n.detach() and y_up | Zero (runtime) | L_sign |
| Stability gap | Computed from points, w, up | Zero (self-supervised) | L_stab |

**No additional annotation is required.**

---

## 4. Architecture Changes

### 4.1 Network Output Modification

Current `SHSUprightNet` produces a single `up` vector.  We modify the
`SignedDirectionHead` to produce two outputs:

```python
class DecomposedDirectionHead(nn.Module):
    """Predicts unsigned axis + polarity sign separately."""

    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        # Axis branch: predicts 3D unit vector (unsigned)
        self.axis_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3),
        )
        # Sign branch: predicts flip probability (scalar)
        self.sign_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, feat):
        h = self.shared(feat)
        axis = F.normalize(self.axis_head(h), dim=-1)   # (B, 3)
        sign_logit = self.sign_head(h).squeeze(-1)       # (B,)
        return axis, sign_logit
```

### 4.2 Inference

```python
axis, sign_logit = model.direction_head(fused)
sign = (torch.sigmoid(sign_logit) > 0.5).float() * 2 - 1  # +1 or -1
up = axis * sign.unsqueeze(-1)
```

### 4.3 Training Forward (Soft Sign for Gradient Flow)

```python
axis, sign_logit = model.direction_head(fused)
p = torch.sigmoid(sign_logit)
# Straight-through: forward uses hard sign, backward flows through p
hard_sign = (p > 0.5).float() * 2 - 1
soft_sign = hard_sign + (2 * p - 1) - (2 * p - 1).detach()  # STE
up = axis * soft_sign.unsqueeze(-1)
```

---

## 5. Training Strategy

### Phase 1: Frozen Trunk, Train Heads (5 epochs)

- Freeze DGCNN trunk (use Pang 2022 pretrained weights).
- Train: context module + axis head + sign head.
- Purpose: verify the decomposed loss works before end-to-end tuning.
- Expected: L_sign should converge quickly (binary classification),
  L_axis should match or beat current antipodal performance.

### Phase 2: End-to-End Fine-Tuning (50 epochs)

- Unfreeze trunk with small LR (5e-5).
- Train all parameters.
- 10x rotations per object per epoch.
- Expected: median error should drop toward 0.7 (matching baseline),
  flip rate on hard categories should decrease further.

### Phase 3: Ablation (for paper)

| Experiment | Config | What It Shows |
|-----------|--------|---------------|
| A1 | L_axis only (current antipodal loss) | Axis-only baseline |
| A2 | L_axis + L_sign | Effect of loss decomposition |
| A3 | L_axis + L_sign + L_sup | Effect of support auxiliary |
| A4 | L_axis + L_sign + L_sup + L_stab | Full model |
| A5 | Signed geodesic (current) | Single-loss baseline |
| A6 | Vary lambda_2 in {0.1, 0.2, 0.5, 1.0} | Sign loss weight sensitivity |
| A7 | With/without n.detach() | Gradient decoupling necessity |

---

## 6. Evaluation Protocol

### 6.1 Fair Comparison (100x Rotations)

Use the pre-generated 100x rotation test set (37,000 samples) for all
comparisons with the Pang 2022 baseline.  The current 1x rotation
evaluation (370 samples) is only suitable for development iteration.

### 6.2 Metrics

| Metric | Definition | Purpose |
|--------|-----------|---------|
| Mean error (signed) | mean(arccos(cos)) | Overall with polarity |
| Mean error (antipodal) | mean(arccos(\|cos\|)) | Axis accuracy only |
| Median error | median of above | Precision on easy samples |
| Acc@5, Acc@10 | % within threshold | Standard benchmarks |
| Flip rate | % with error > 90 | Direct measure of polarity failures |
| Per-category breakdown | Above metrics per class | Identify hard categories |

### 6.3 Key Success Criteria

| Criterion | Target | Rationale |
|-----------|--------|-----------|
| Antipodal Acc@10 | >= 93.05% | Match or beat Pang baseline |
| Signed mean error | < 11.17 | Beat Pang baseline |
| Flip rate (overall) | < 4% | Improve on current 4.05% |
| Flip rate (lamp) | < 10% | Down from 16%, the hardest case |
| Median error | < 1.0 | Close to baseline's 0.71 |

---

## 7. Why This Is Not Just "Apply SHS-Net"

The loss decomposition is inspired by SHS-Net, but the application context
introduces fundamentally different challenges:

| Aspect | SHS-Net (Normal Orientation) | Ours (Upright Polarity) |
|--------|------------------------------|------------------------|
| Output | Per-point oriented normal | Single global direction |
| Sign semantics | Geometric (inside/outside) | Semantic (up/down) |
| Local information | Sufficient for most points | Never sufficient |
| Symmetry | Rare (most surfaces have clear in/out) | Common (vase, cup, cylinder) |
| Physical prior | Not applicable | Stability constraint (L_stab) |

The physical stability loss (L_stab) is a novel contribution specific to
the upright estimation problem --- it has no counterpart in SHS-Net or any
prior normal estimation work.  It encodes the domain-specific insight that
"objects stand on their support, not float above it" as a differentiable,
self-supervised loss term.

---

## 8. File Plan

| File | Action | Content |
|------|--------|---------|
| `Common/loss_shs.py` | Modify | Add `DecomposedLoss` with L_axis + L_sign + L_sup + L_stab |
| `network_shs.py` | Modify | Replace `SignedDirectionHead` with `DecomposedDirectionHead` |
| `scripts/train_shs.py` | Modify | Update training loop for decomposed outputs |
| `scripts/eval_100x.py` | New | Fair evaluation with 100x rotation test set |

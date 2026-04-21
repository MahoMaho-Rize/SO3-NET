# DifferentiableLieUprightRefineNet — Experiment Plan

Status snapshot (2026-04-21): coarse-init support-plane baseline reaches
acc@10 ≈ 82%, median 0.8° on UprightNet15 test; ~6% catastrophic 180° flips
concentrated in bed/lamp/vase. The iterative refine head now runs as a fully
continuous Lie-algebra solver (no per-iter `max_angle_schedule`) but does not
yet beat the coarse baseline on the held-out set.

## Architecture recap

`DifferentiableLieUprightRefineNet` (`network_lie.py`):

```
R_0 = so3_from_up(weighted_plane_normal(trunk(P)), pca_largest_axis(P))
for t in 1..T:
    P_t      = R_t^T @ P
    feat     = trunk(P_t)            # scalar global feature
    polar    = PolarityFeatures(P_t) # 3rd moment + weighted polar vec
    omega_t  = refine_head(feat ⊕ polar)    # (B, 3), tanh-soft-clipped at π-ε
    omega_t  = omega_t · (1, 0, 1)          # gauge: kill local y-component
    R_{t+1}  = R_t @ exp_so3(omega_t)
up = R_T @ e_y
```

Training loss (`compute_multi_iter_loss`) is the γ-weighted sum of per-iter
geodesics, with the t=0 term excluded (no gradient to the head when trunk
is frozen).

## Known facts

- **Head capacity ✓** — overfit on 16 hard samples with trainable trunk
  hits `geo_final < 1°` in train mode.
- **Trunk is the ceiling when frozen** — same overfit with frozen trunk
  plateaus at ~22°, because Pang 2022's BN running stats don't distinguish
  180°-flipped inputs well enough.
- **BN handling is now correct** — trunk in `.eval()` during training when
  frozen; `--freeze_bn` flag freezes running stats + γ/β even when trunk
  convs are fine-tuning; `--recalibrate_bn` re-estimates running stats after
  training (equivalent to `torch.optim.swa_utils.update_bn`).
- **OOM at bs=48 with `lr_trunk>0`** — BPTT through 3 trunk calls blows up
  DGCNN activations. bs=16 is the safe cap on a 48 GB card without gradient
  checkpointing.

## Phases

### Phase A — Validate fine-tune recipe (small split, fast)

Datasets: `--max_train_samples 7500 --max_test_samples 3000` (class-stratified).
Runtime: ~5 min per epoch at bs=16.

| ID  | Config                                                    | Success criterion                               |
| --- | --------------------------------------------------------- | ----------------------------------------------- |
| A1  | `lr_trunk=0 --freeze_bn` (baseline)                       | `[final]` mean < `[init]` mean by ≥0.5°         |
| A2  | `lr_trunk=1e-4 --freeze_bn --recalibrate_bn`              | `[final]` mean drops ≥2° over 5 epochs          |
| A3  | `lr_trunk ∈ {1e-5, 1e-4, 3e-4} --freeze_bn`               | Pick lr with best eval-set final at epoch 5     |

### Phase B — Locate hard-case ceiling

| ID  | Config                                                                         | What it tells us                                         |
| --- | ------------------------------------------------------------------------------ | -------------------------------------------------------- |
| B1  | Phase-A best ckpt → full 37k test, per-category breakdown                      | Whether trunk fine-tune helps bed/lamp/vase specifically |
| B2  | `overfit_dlie_refine.py --hard_only --labels 0 8 14 --num_iters 5 --lr_trunk 1e-4` | Whether extra iters give the head enough budget      |

### Phase C — Full training

Only run if A2 and B1 validate the recipe.

| ID  | Config                                                                         | Expected outcome                                |
| --- | ------------------------------------------------------------------------------ | ----------------------------------------------- |
| C1  | Full 111k/37k, `--epochs 10 --lr_trunk 1e-4 --freeze_bn --recalibrate_bn`      | acc@10 ≥ 85%, mean < 12°                        |
| C2  | Control: `--lr_trunk 0 --epochs 10`                                            | Lower bound (how much the head alone can learn) |

### Phase D — Escape hatches (only if C1 stalls)

1. **Gradient checkpointing** on trunk → recover bs=48, more stable optim.
2. **Swap trunk for VN-DGCNN** (strictly SO(3)-equivariant). Gives up Pang's
   pretraining but fixes the root cause of approximate equivariance.
3. **Neural ODE refine** (`torchdiffeq`) — adaptive step count at test time
   for the hard-case tail.
4. **Hard-sample mining / reweighted loss** so the ~5% 180°-flip cases are
   not drowned out by the median-0.8° majority.

## Execution order

```
A1 → A2 → B1
      └── A2 gives final mean < 10° and recalibration helps
          → launch C1 overnight
      └── A2 fails (final ~= 16°)
          → run B2, then pick from Phase D
```

## Metrics we track per run

- `[init]` / `[final]` mean, median, acc@5, acc@10, flip-rate (>90°)
- Per-category breakdown (especially bed, lamp, vase, toilet)
- Loss curve (`train loss`, `per-iter geo in deg`)
- Train vs eval consistency check (after recalibration, they should match)

All checkpoints are saved under `models/dlie_refine_{best,final}.pth`.

"""
Evaluate the official Upright-Net baseline checkpoint (model.pth from the
reference repo) on the full UprightNet15 test set.

Exports:
  logs/baseline_official.npz   - per-sample errors with BOTH fallback variants:
      * error_leak_deg : (N,)  using the original GT-leaking fallback at line 201
      * error_fix_deg  : (N,)  using [0,1,0] fallback (honest baseline)
      * num_support    : (N,)  #points predicted as supporting
      * label          : (N,)  category id
      * is_leak_triggered : (N,) bool  true when num_support == 0

This is Phase 0 of the research plan.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn import linear_model

REPO = Path(__file__).resolve().parents[1]  # uprightnet/
REF_REPO = REPO.parent / "uprightnet-reference"

# Import official network from reference repo
sys.path.insert(0, str(REF_REPO))
from network import UprightNet  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(REF_REPO / "model" / "model.pth"))
    p.add_argument("--data_dir", default=str(REPO / "datasets" / "uprightnet15"))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--out", default=str(REPO / "logs" / "baseline_official.npz"))
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def upright_from_pred(
    pred_mask: torch.Tensor,  # (B, N) bool
    points: torch.Tensor,  # (B, N, 3) original (unrotated) points
    rotm: torch.Tensor,  # (B, 3, 3) rotation applied to rotated input
    use_leak_fallback: bool,
):
    """Reproduce UprightOriEst but operate in a batched loop and return per-sample
    info about whether the fallback was triggered."""
    B = pred_mask.shape[0]
    orientations = torch.empty((B, 3), dtype=torch.float32)
    num_support = torch.zeros(B, dtype=torch.int32)
    fallback_triggered = torch.zeros(B, dtype=torch.bool)

    ransac = linear_model.RANSACRegressor(residual_threshold=0.03)
    for i in range(B):
        p = points[i]
        m = pred_mask[i]
        mcenter = torch.mean(p, dim=0).cpu()
        sp = p[m].cpu()
        n = sp.shape[0]
        num_support[i] = n
        if n >= 3:
            ransac.fit(sp[:, [0, 2]].numpy(), sp[:, 1].numpy())
            a, c = ransac.estimator_.coef_
            d = float(ransac.estimator_.intercept_)
            v = torch.tensor([a, -1.0, c], dtype=torch.float32)
            v = v / torch.norm(v, p=2)
            sign = torch.sign(a * mcenter[0] - 1.0 * mcenter[1] + c * mcenter[2] + d)
            orientations[i] = v * sign
        elif n > 0:
            scenter = torch.mean(sp, dim=0)
            v = mcenter - scenter
            orientations[i] = v / torch.norm(v, p=2)
        else:
            fallback_triggered[i] = True
            if use_leak_fallback:
                # GT leak: R^{-1}[:,1]
                orientations[i] = torch.inverse(rotm[i].cpu())[1].float()
            else:
                orientations[i] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    return orientations, num_support, fallback_triggered


def angular_error_deg(pred_o, rotm):
    """Angular error (in degrees) between predicted upright and GT upright.

    Reference eval uses: angle = arccos(orientation[:,1]) because the
    prediction is rotated back into canonical frame by construction: the
    pipeline operates on `original` points (not rotated ones) so the upright
    axis should be aligned with world y=[0,1,0]. error = arccos(y-component).
    We use the direct |cos| with GT = [0,1,0] to get signed errors.
    """
    # Reference's formula: angle = arccos(pred[:, 1])
    # which gives 0 when perfectly pointing up and 180 when inverted.
    cos = pred_o[:, 1].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    err_signed = torch.acos(cos) * (180.0 / torch.pi)  # 0..180
    return err_signed


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda")

    # Load data
    data_dir = Path(args.data_dir)
    original = np.load(data_dir / "test_original.npy").astype(np.float32)
    rotation = np.load(data_dir / "test_rotation.npy").astype(np.float32)
    rotm = np.load(data_dir / "rotm_test.npy").astype(np.float32)
    labels = np.load(data_dir / "labels_test.npy").astype(np.int32)
    N = original.shape[0]
    print(f"Loaded {N} test samples (N_points={original.shape[1]})")

    # Build model and load weights
    net = UprightNet().to(device)
    # Reference trained with DataParallel -> keys are module.xxx
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    is_dp = any(k.startswith("module.") for k in ckpt.keys())
    if is_dp:
        net = nn.DataParallel(net)
    net.load_state_dict(ckpt)
    net.eval()
    print(f"Loaded checkpoint from {args.ckpt} (DataParallel={is_dp})")

    # Run inference
    BS = args.batch_size
    all_err_leak = np.zeros(N, dtype=np.float32)
    all_err_fix = np.zeros(N, dtype=np.float32)
    all_num_support = np.zeros(N, dtype=np.int32)
    all_fallback = np.zeros(N, dtype=bool)

    with torch.no_grad():
        for start in tqdm(range(0, N, BS)):
            end = min(N, start + BS)
            batch_orig = torch.from_numpy(original[start:end]).to(device)
            batch_rot = torch.from_numpy(rotation[start:end]).to(device)
            batch_rotm = torch.from_numpy(rotm[start:end]).to(device)

            # Reference uses rotated input as network input, then evaluates
            # the pipeline on the ORIGINAL (canonical) points
            x = batch_rot.transpose(2, 1).contiguous()  # (B, 3, N)
            logits = net(x).squeeze(1)  # (B, N)
            pred_mask = logits > 0.5

            o_leak, nsup, fb = upright_from_pred(
                pred_mask, batch_orig, batch_rotm, use_leak_fallback=True
            )
            o_fix, _, _ = upright_from_pred(
                pred_mask, batch_orig, batch_rotm, use_leak_fallback=False
            )

            err_leak = angular_error_deg(o_leak, batch_rotm.cpu())
            err_fix = angular_error_deg(o_fix, batch_rotm.cpu())

            all_err_leak[start:end] = err_leak.numpy()
            all_err_fix[start:end] = err_fix.numpy()
            all_num_support[start:end] = nsup.numpy()
            all_fallback[start:end] = fb.numpy()

    # Summary
    def summarize(name, err):
        mean = err.mean()
        med = np.median(err)
        acc5 = (err < 5).mean() * 100
        acc10 = (err < 10).mean() * 100
        acc30 = (err < 30).mean() * 100
        flip = (err > 90).mean() * 100
        print(
            f"[{name:10s}] mean={mean:6.2f}  median={med:6.2f}  "
            f"acc@5={acc5:5.2f}%  acc@10={acc10:5.2f}%  acc@30={acc30:5.2f}%  "
            f"flip(>90°)={flip:.2f}%"
        )

    print()
    print("=" * 72)
    summarize("leak", all_err_leak)
    summarize("fix ", all_err_fix)
    print(f"Fallback triggered on {all_fallback.sum()} / {N} samples "
          f"({all_fallback.mean() * 100:.2f}%)")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        error_leak_deg=all_err_leak,
        error_fix_deg=all_err_fix,
        num_support=all_num_support,
        is_leak_triggered=all_fallback,
        label=labels,
    )
    print(f"Saved per-sample errors to {out_path}")


if __name__ == "__main__":
    main()

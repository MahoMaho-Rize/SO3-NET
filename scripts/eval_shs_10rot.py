"""Fixed 10-rotation evaluation for SHS decomposed-head network.

Uses datasets/uprightnet15/test_10rot_rotation.npy (3700 samples = 370 objects
x 10 fixed random SO(3) rotations) so results are reproducible and comparable
to baseline_per_category.csv.

Reports:
  - Overall mean/median/acc@5/10/30/flip
  - Per-category breakdown
  - Decomposition: axis error (antipodal) vs sign error rate
    → tells you whether failures come from bad axis or wrong polarity
  - Lamp-specific failure dump: top-K worst lamp predictions
"""

from pathlib import Path
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from network_shs import SHSUprightNet
from network_hybrid import HybridUprightNet

CATEGORY_NAMES = [
    "bed", "bench", "bottle", "bowl", "car", "chair", "cone", "cup",
    "lamp", "monitor", "sofa", "stool", "table", "toilet", "vase",
]


def angular_error_deg(pred, gt, antipodal=False):
    pred = F.normalize(pred, dim=-1)
    gt = F.normalize(gt, dim=-1)
    cos = (pred * gt).sum(-1)
    if antipodal:
        cos = cos.abs()
    cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos) * (180.0 / torch.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/shs_best.pth")
    ap.add_argument("--data_dir", default="datasets/uprightnet15")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--topk_fail", type=int, default=10,
                    help="dump K hardest lamp samples")
    ap.add_argument("--split", default="10rot", choices=["10rot", "full"],
                    help="'10rot' = 3700 samples; 'full' = 37000 samples "
                         "(matches baseline_official.npz)")
    ap.add_argument("--arch", default="shs", choices=["shs", "hybrid"])
    args = ap.parse_args()

    device = torch.device("cuda")
    data_dir = Path(args.data_dir)

    if args.split == "full":
        rot_f = "test_rotation.npy"
        rotm_f = "rotm_test.npy"
        lab_f = "labels_test.npy"
    else:
        rot_f = "test_10rot_rotation.npy"
        rotm_f = "rotm_test_10rot.npy"
        lab_f = "labels_test_10rot.npy"
    rotation = np.load(data_dir / rot_f).astype(np.float32)
    rotm = np.load(data_dir / rotm_f).astype(np.float32)
    labels = np.load(data_dir / lab_f).astype(np.int64).flatten()
    N = rotation.shape[0]
    gt_up = rotm[:, :, 1]  # rotated y-axis = GT upright
    print(f"Loaded {N} samples ({N//10} objects × 10 rotations)")

    if args.arch == "hybrid":
        model = HybridUprightNet(freeze_trunk=False).to(device)
    else:
        model = SHSUprightNet(
            freeze_trunk=False, head_type="decomposed",
        ).to(device)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded {args.ckpt} — missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    err_signed = np.zeros(N, dtype=np.float32)
    err_anti = np.zeros(N, dtype=np.float32)
    sign_correct = np.zeros(N, dtype=np.float32)
    sign_prob = np.zeros(N, dtype=np.float32)

    with torch.no_grad():
        for s in tqdm(range(0, N, args.batch_size)):
            e = min(N, s + args.batch_size)
            P = torch.from_numpy(rotation[s:e]).to(device)
            gt = torch.from_numpy(gt_up[s:e]).to(device)

            out = model(P)
            up = out["up"]
            axis = out["axis"]
            sign_logit = out["sign_logit"]

            err_signed[s:e] = angular_error_deg(up, gt).cpu().numpy()
            err_anti[s:e] = angular_error_deg(axis, gt, antipodal=True).cpu().numpy()

            # sign correctness: does predicted sign match what would align
            # the predicted axis with the GT?
            axis_n = F.normalize(axis, dim=-1)
            gt_n = F.normalize(gt, dim=-1)
            sign_gt = ((axis_n * gt_n).sum(-1) > 0).float()
            sign_pred = (torch.sigmoid(sign_logit) > 0.5).float()
            sign_correct[s:e] = (sign_pred == sign_gt).float().cpu().numpy()
            sign_prob[s:e] = torch.sigmoid(sign_logit).cpu().numpy()

    # ---- Overall ----
    print()
    print("=" * 66)
    print("  Overall (3700 samples)")
    print("=" * 66)
    print(f"  Signed  — mean: {err_signed.mean():6.2f}°  "
          f"median: {np.median(err_signed):5.2f}°  "
          f"acc@5: {(err_signed<5).mean()*100:5.1f}%  "
          f"acc@10: {(err_signed<10).mean()*100:5.1f}%  "
          f"acc@30: {(err_signed<30).mean()*100:5.1f}%")
    print(f"  Antipod — mean: {err_anti.mean():6.2f}°  "
          f"median: {np.median(err_anti):5.2f}°  "
          f"acc@5: {(err_anti<5).mean()*100:5.1f}%  "
          f"acc@10: {(err_anti<10).mean()*100:5.1f}%")
    print(f"  Flip rate (>90°): {(err_signed>90).mean()*100:5.2f}%")
    print(f"  Sign-branch accuracy: {sign_correct.mean()*100:5.2f}%")
    print(f"  Sign confidence: mean prob = {sign_prob.mean():.3f}, "
          f"std = {sign_prob.std():.3f}")

    # ---- Failure-mode decomposition ----
    # A sample can fail in 3 ways:
    #   (a) axis wrong  (antipodal err > 10°)
    #   (b) axis right but sign wrong  (antipod < 10° but flip happened)
    #   (c) both
    axis_bad = err_anti > 10.0
    flip = err_signed > 90.0
    sign_bad_only = (~axis_bad) & flip  # axis OK, sign flipped
    both_bad = axis_bad & flip
    axis_bad_only = axis_bad & (~flip)
    all_good = (~axis_bad) & (~flip)

    print()
    print("  Failure decomposition:")
    print(f"    all good (axis<10° AND no flip):       {all_good.mean()*100:5.2f}%")
    print(f"    axis bad only (>10°, sign OK):         {axis_bad_only.mean()*100:5.2f}%")
    print(f"    sign flip only (axis<10°, flipped):    {sign_bad_only.mean()*100:5.2f}%  ← pure sign failure")
    print(f"    both bad:                              {both_bad.mean()*100:5.2f}%")

    # ---- Per-category ----
    print()
    print("=" * 78)
    print("  Per-category (mean_s = mean signed,  anti = axis mean,  sign% = sign acc)")
    print("=" * 78)
    print(f"  {'cat':<9} {'N':>5} {'mean_s':>7} {'med_s':>7} "
          f"{'anti':>7} {'acc@10':>7} {'flip%':>6} {'sign%':>6} "
          f"{'sign_only_fail%':>16}")
    print("  " + "-" * 76)
    by_cat_rows = []
    for c in range(15):
        mask = labels == c
        if mask.sum() == 0:
            continue
        row = {
            "cat": CATEGORY_NAMES[c],
            "N": int(mask.sum()),
            "mean_s": err_signed[mask].mean(),
            "median_s": np.median(err_signed[mask]),
            "anti": err_anti[mask].mean(),
            "acc10": (err_signed[mask] < 10).mean() * 100,
            "flip": (err_signed[mask] > 90).mean() * 100,
            "sign_acc": sign_correct[mask].mean() * 100,
            "sign_only_fail": sign_bad_only[mask].mean() * 100,
        }
        by_cat_rows.append(row)
        print(f"  {row['cat']:<9} {row['N']:>5} {row['mean_s']:>6.2f}° "
              f"{row['median_s']:>6.2f}° {row['anti']:>6.2f}° "
              f"{row['acc10']:>6.1f}% {row['flip']:>5.1f}% "
              f"{row['sign_acc']:>5.1f}% {row['sign_only_fail']:>15.2f}%")

    # ---- Lamp failure dump ----
    lamp_idx = np.where(labels == 8)[0]
    lamp_errs = err_signed[lamp_idx]
    worst_order = lamp_idx[np.argsort(-lamp_errs)[: args.topk_fail]]
    print()
    print(f"=== Top-{args.topk_fail} hardest lamp samples ===")
    print(f"  {'idx':>5} {'obj':>4} {'rot':>4} {'err_s':>7} {'err_a':>7} "
          f"{'sign_p':>7} {'sign_ok':>7}")
    for i in worst_order:
        obj = i // 10
        rot = i % 10
        print(f"  {i:>5} {obj:>4} {rot:>4} "
              f"{err_signed[i]:>6.2f}° {err_anti[i]:>6.2f}° "
              f"{sign_prob[i]:>7.3f} {int(sign_correct[i]):>7}")

    # ---- Save ----
    out_path = REPO / "logs" / "shs_10rot_eval.npz"
    np.savez(
        out_path,
        err_signed=err_signed,
        err_anti=err_anti,
        sign_correct=sign_correct,
        sign_prob=sign_prob,
        labels=labels,
    )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

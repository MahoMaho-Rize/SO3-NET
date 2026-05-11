"""Overfit sanity check for HybridUprightNet.

Verifies that on a fixed 16-sample batch, the full stack (trunk → support →
weighted-PCA → sign head) can drive axis and sign errors to near zero.

Expected: after ~300 steps, mean angular error < 2°, sign_acc = 100%.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from network_hybrid import HybridUprightNet
from Common.loss_shs import DecomposedLoss
from Common.geometric_utils import angular_error_deg
from scripts.train_shs import UprightDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(REPO / "datasets" / "uprightnet15"))
    ap.add_argument("--trunk_ckpt", default=str(
        REPO.parent / "uprightnet-reference" / "model" / "model.pth"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_trunk", type=float, default=1e-4)
    ap.add_argument("--lambda_axis", type=float, default=1.0)
    ap.add_argument("--lambda_sign", type=float, default=0.5)
    ap.add_argument("--lambda_sup", type=float, default=0.5)
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=str, default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    ds = UprightDataset(args.data_dir, "train", 2048, rotations_per_object=1)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(ds.num_objects, size=args.batch_size, replace=False)
    pos_list, yd_list, sup_list, lab_list = [], [], [], []
    for i in idx.tolist():
        p, y, s, lb = ds[i]
        pos_list.append(p); yd_list.append(y); sup_list.append(s); lab_list.append(lb)
    pos = torch.stack(pos_list).to(device)
    y_dir = torch.stack(yd_list).to(device)
    support = torch.stack(sup_list).to(device)
    labels = torch.tensor(lab_list)
    print(f"Batch: pos={tuple(pos.shape)}  labels={np.bincount(labels.numpy(), minlength=15).tolist()}")

    model = HybridUprightNet(freeze_trunk=False).to(device)
    model.load_trunk_from_ckpt(args.trunk_ckpt)

    groups = [
        {"params": list(model.trunk.parameters()), "lr": args.lr_trunk, "name": "trunk"},
        {"params": [p for n, p in model.named_parameters()
                    if not n.startswith("trunk.")], "lr": args.lr_head, "name": "head"},
    ]
    optim = torch.optim.AdamW(groups, weight_decay=0.0)
    crit = DecomposedLoss(
        lambda_axis=args.lambda_axis, lambda_sign=args.lambda_sign,
        lambda_sup=args.lambda_sup, lambda_stab=0.0,
    )

    print()
    print(f"  {'step':>5} {'total':>7} {'axis':>6} {'sign':>6} {'sup':>6} | "
          f"{'ang':>7} {'sign%':>6} {'flip%':>6}")
    print("  " + "-" * 60)

    model.train()
    best_ang = float("inf")
    for step in range(1, args.steps + 1):
        out = model(pos)
        ld = crit(out, {"y_direction": y_dir, "y_support": support, "points": pos})
        optim.zero_grad(set_to_none=True)
        ld["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()

        with torch.no_grad():
            ang = angular_error_deg(out["up"], y_dir, antipodal=False)
            sign_pred = (torch.sigmoid(out["sign_logit"]) > 0.5).float()
            axis_n = F.normalize(out["axis"], dim=-1)
            gt_n = F.normalize(y_dir, dim=-1)
            sign_gt = ((axis_n * gt_n).sum(-1) > 0).float()
            sign_acc = (sign_pred == sign_gt).float().mean().item() * 100
            flip_rate = (ang > 90.0).float().mean().item() * 100
            ang_mean = ang.mean().item()

        if ang_mean < best_ang:
            best_ang = ang_mean

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(f"  {step:5d} {ld['total'].item():7.4f} "
                  f"{ld['axis'].item():6.3f} {ld['sign'].item():6.3f} "
                  f"{ld['support'].item():6.3f} | "
                  f"{ang_mean:6.2f}° {sign_acc:5.1f}% {flip_rate:5.1f}%")

    model.eval()
    with torch.no_grad():
        out = model(pos)
        ang = angular_error_deg(out["up"], y_dir, antipodal=False)
        ang_a = angular_error_deg(out["axis"], y_dir, antipodal=True)
        sign_pred = (torch.sigmoid(out["sign_logit"]) > 0.5).float()
        axis_n = F.normalize(out["axis"], dim=-1)
        gt_n = F.normalize(y_dir, dim=-1)
        sign_gt = ((axis_n * gt_n).sum(-1) > 0).float()
        sign_acc = (sign_pred == sign_gt).float().mean().item() * 100
    print()
    print(f"eval-mode:  mean_signed={ang.mean():.3f}°  mean_anti={ang_a.mean():.3f}°  "
          f"sign_acc={sign_acc:.1f}%  flip%={(ang>90).float().mean()*100:.1f}")
    print(f"best during: ang={best_ang:.3f}°")


if __name__ == "__main__":
    main()

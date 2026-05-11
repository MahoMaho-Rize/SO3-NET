"""Sanity-check: overfit SHSUprightNet (decomposed head) on a single fixed batch.

If the decomposed head, auxiliary losses, and gradient flow are healthy,
a few hundred AdamW steps on the same batch should drive both:
  - axis_err (antipodal) well under 1°
  - sign_acc to 100%
  - stab_gap positive on all samples

If any of these plateau, the corresponding branch is mis-wired.

Typical usage:
    # frozen trunk, fit head+context only
    python3 scripts/overfit_shs.py --steps 500 --batch_size 16

    # unfreeze trunk with tiny LR (simulate phase 2)
    python3 scripts/overfit_shs.py --steps 500 --batch_size 16 --lr_trunk 1e-5
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

from network_shs import SHSUprightNet
from Common.loss_shs import DecomposedLoss
from Common.geometric_utils import angular_error_deg
from scripts.train_shs import UprightDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(REPO / "datasets" / "uprightnet15"))
    ap.add_argument("--trunk_ckpt", default=str(
        REPO.parent / "uprightnet-reference" / "model" / "model.pth"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_points", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_trunk", type=float, default=0.0,
                    help=">0 unfreezes trunk with this LR (simulates phase 2)")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--lambda_axis", type=float, default=1.0)
    ap.add_argument("--lambda_sign", type=float, default=0.5)
    ap.add_argument("--lambda_sup", type=float, default=0.1)
    ap.add_argument("--lambda_stab", type=float, default=0.2)
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip_grad", type=float, default=5.0)
    ap.add_argument("--labels", type=int, nargs="+", default=None,
                    help="restrict to these class labels (e.g. 0 8 14 for bed/lamp/vase)")
    ap.add_argument("--gpu", type=str, default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    print(f"Overfit SHS: bs={args.batch_size} steps={args.steps} "
          f"lr_head={args.lr_head} lr_trunk={args.lr_trunk}")
    print(f"λ: axis={args.lambda_axis} sign={args.lambda_sign} "
          f"sup={args.lambda_sup} stab={args.lambda_stab}")

    # ---- Fixed batch ----
    ds = UprightDataset(args.data_dir, "train", args.num_points,
                        rotations_per_object=1)
    rng = np.random.default_rng(args.seed)
    label_arr = np.asarray(ds.labels).flatten()
    if args.labels is not None:
        mask = np.isin(label_arr.astype(int), args.labels)
        cand = np.where(mask)[0]
        print(f"Filtered to labels {args.labels}: {len(cand)} candidates")
    else:
        cand = np.arange(ds.num_objects)
    idx = rng.choice(cand, size=args.batch_size, replace=False)

    pos_list, ydir_list, sup_list, lab_list = [], [], [], []
    for i in idx.tolist():
        # Draw one rotated view and freeze it for the whole run.
        p, yd, s, lb = ds[i]
        pos_list.append(p)
        ydir_list.append(yd)
        sup_list.append(s)
        lab_list.append(lb)
    pos = torch.stack(pos_list).to(device)
    y_dir = torch.stack(ydir_list).to(device)
    support = torch.stack(sup_list).to(device)
    labels = torch.tensor(lab_list)
    print(f"Fixed batch: pos={tuple(pos.shape)} y_dir={tuple(y_dir.shape)} "
          f"label hist={np.bincount(labels.numpy(), minlength=15).tolist()}")

    # ---- Model ----
    trunk_frozen = args.lr_trunk <= 0
    model = SHSUprightNet(
        freeze_trunk=trunk_frozen,
        head_type="decomposed",
    ).to(device)
    missing, unexpected = model.load_trunk_from_ckpt(
        args.trunk_ckpt, strict=False)
    print(f"Loaded trunk: missing={len(missing or [])} "
          f"unexpected={len(unexpected or [])}")

    # ---- Optim ----
    trunk_params = list(model.trunk.parameters())
    other = [p for n, p in model.named_parameters()
             if not n.startswith("trunk.") and p.requires_grad]
    groups = [{"params": other, "lr": args.lr_head, "name": "head"}]
    if not trunk_frozen:
        for p in trunk_params:
            p.requires_grad = True
        groups.append({"params": trunk_params, "lr": args.lr_trunk, "name": "trunk"})
    else:
        for p in trunk_params:
            p.requires_grad = False
        print("Trunk frozen — BN in eval mode during overfit")
    optim = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    criterion = DecomposedLoss(
        lambda_axis=args.lambda_axis,
        lambda_sign=args.lambda_sign,
        lambda_sup=args.lambda_sup,
        lambda_stab=args.lambda_stab,
    )

    # ---- Loop ----
    model.train()
    if trunk_frozen:
        model.trunk.eval()

    print()
    header = (f"  {'step':>5} {'total':>7} {'axis':>6} {'sign':>6} {'sup':>6} "
              f"{'stab':>6} | {'ang_err':>8} {'sign_acc':>9} {'flip%':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    best_ang = float("inf")
    best_sign_acc = 0.0
    for step in range(1, args.steps + 1):
        out = model(pos)
        loss_dict = criterion(
            out,
            {"y_direction": y_dir, "y_support": support, "points": pos},
        )
        loss = loss_dict["total"]

        optim.zero_grad(set_to_none=True)
        loss.backward()

        if step == 1:
            # Report grad norms once to confirm all branches are live.
            def gn(params):
                s = 0.0
                for p in params:
                    if p.grad is not None:
                        s += p.grad.detach().pow(2).sum().item()
                return s ** 0.5
            h = model.direction_head
            print(f"  [step 1] grad norms — axis_head={gn(h.axis_head.parameters()):.3e} "
                  f"sign_head={gn(h.sign_head.parameters()):.3e} "
                  f"shared={gn(h.shared.parameters()):.3e} "
                  f"context={gn(model.context.parameters()):.3e}")

        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=args.clip_grad)
        optim.step()

        # Diagnostics on this fixed batch
        with torch.no_grad():
            ang = angular_error_deg(out["up"], y_dir, antipodal=False)
            ang_mean = ang.mean().item()
            flip_rate = (ang > 90.0).float().mean().item() * 100
            sign_pred = (torch.sigmoid(out["sign_logit"]) > 0.5).float()
            axis_d = F.normalize(out["axis"].detach(), dim=-1)
            gt_d = F.normalize(y_dir, dim=-1)
            sign_gt = ((axis_d * gt_d).sum(dim=-1) > 0).float()
            sign_acc = (sign_pred == sign_gt).float().mean().item() * 100

        if ang_mean < best_ang:
            best_ang = ang_mean
        if sign_acc > best_sign_acc:
            best_sign_acc = sign_acc

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(f"  {step:5d} {loss_dict['total'].item():7.4f} "
                  f"{loss_dict['axis'].item():6.3f} "
                  f"{loss_dict['sign'].item():6.3f} "
                  f"{loss_dict['support'].item():6.3f} "
                  f"{loss_dict['stab'].item():6.3f} | "
                  f"{ang_mean:7.2f}° {sign_acc:8.1f}% {flip_rate:5.1f}%")

    # ---- Final eval snapshot ----
    model.eval()
    with torch.no_grad():
        out = model(pos)
        ang = angular_error_deg(out["up"], y_dir, antipodal=False)
        ang_anti = angular_error_deg(out["up"], y_dir, antipodal=True)
        sign_pred = (torch.sigmoid(out["sign_logit"]) > 0.5).float()
        axis_d = F.normalize(out["axis"], dim=-1)
        gt_d = F.normalize(y_dir, dim=-1)
        sign_gt = ((axis_d * gt_d).sum(dim=-1) > 0).float()
        sign_acc = (sign_pred == sign_gt).float().mean().item() * 100

    print()
    print("Per-sample final (eval-mode) errors:")
    for i in range(len(labels)):
        print(f"    [{i:2d}] lab={labels[i].item():2d}  "
              f"signed={ang[i].item():6.2f}°  anti={ang_anti[i].item():6.2f}°  "
              f"sign_logit={out['sign_logit'][i].item():+.2f}  "
              f"sign_gt={int(sign_gt[i].item())}")
    print()
    print(f"Eval-mode: mean_signed={ang.mean():.3f}°  mean_anti={ang_anti.mean():.3f}°  "
          f"sign_acc={sign_acc:.1f}%  flip%={(ang>90).float().mean()*100:.1f}")
    print(f"Best during training: ang={best_ang:.3f}°  sign_acc={best_sign_acc:.1f}%")


if __name__ == "__main__":
    main()

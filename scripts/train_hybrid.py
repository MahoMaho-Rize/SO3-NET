#!/usr/bin/env python3
"""Training for HybridUprightNet = weighted-PCA axis + learned sign.

The axis comes from geometry (differentiable eigendecomposition of the
support-weighted covariance), so the heavy lifting is done by the
Pang-style trunk that predicts support probability. The only free part
the loss drives end-to-end is the sign_head (+ context).

Usage (first run):
    python3 scripts/train_hybrid.py --epochs 40 --batch_size 48 \
        --lr 1e-3 --lr_trunk 1e-4

Resume:
    python3 scripts/train_hybrid.py --epochs 40 --batch_size 48 \
        --lr 3e-4 --lr_trunk 3e-5 --resume models/hybrid_final_<stamp>.pth
"""

import argparse
import os
import sys
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_hybrid import HybridUprightNet
from Common.loss_shs import DecomposedLoss
from scripts.train_shs import UprightDataset, evaluate, print_metrics


def main():
    parser = argparse.ArgumentParser("Hybrid upright training")
    parser.add_argument("--data_dir", default="./datasets/uprightnet15/")
    parser.add_argument("--model_dir", default="./models/")
    parser.add_argument("--trunk_ckpt", default=None,
                        help="Pang trunk ckpt. If unset, tries "
                             "~/uprightnet-reference/model/model.pth")
    parser.add_argument("--resume", default=None)

    # Architecture
    parser.add_argument("--context_layers", type=int, default=2)
    parser.add_argument("--context_heads", type=int, default=4)
    parser.add_argument("--sign_hidden", type=int, default=256)

    # Loss (reuses DecomposedLoss — stab off by default)
    parser.add_argument("--lambda_axis", type=float, default=1.0)
    parser.add_argument("--lambda_sign", type=float, default=0.5)
    parser.add_argument("--lambda_sup", type=float, default=0.5,
                        help="bumped vs SHS default (0.1) because support "
                             "is the direct supervisor of the axis now")
    parser.add_argument("--lambda_stab", type=float, default=0.0,
                        help="Physical stability prior; default OFF for hybrid")

    # Training
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_trunk", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--rotations_per_object", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="0")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Data ----
    train_ds = UprightDataset(args.data_dir, "train", args.num_points,
                              rotations_per_object=args.rotations_per_object)
    test_ds = UprightDataset(args.data_dir, "test", args.num_points)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} objects  Test: {len(test_ds)} objects")

    # ---- Model ----
    model = HybridUprightNet(
        context_layers=args.context_layers,
        context_heads=args.context_heads,
        sign_hidden=args.sign_hidden,
        freeze_trunk=False,  # always train trunk end-to-end for hybrid
    ).to(device)

    trunk_ckpt = args.trunk_ckpt or os.path.expanduser(
        "~/uprightnet-reference/model/model.pth"
    )
    if os.path.exists(trunk_ckpt):
        missing, unexpected = model.load_trunk_from_ckpt(trunk_ckpt)
        print(f"Loaded Pang trunk from {trunk_ckpt}  "
              f"missing={len(missing or [])} unexpected={len(unexpected or [])}")
    else:
        print(f"WARNING: trunk checkpoint not found at {trunk_ckpt} "
              "— training from scratch")

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Resumed from {args.resume}  "
              f"missing={len(missing)} unexpected={len(unexpected)}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total:,} total")

    # ---- Optimizer (per-group LR) ----
    trunk_params = list(model.trunk.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith("trunk.") and p.requires_grad]
    param_groups = [
        {"params": trunk_params, "lr": args.lr_trunk, "name": "trunk"},
        {"params": other_params, "lr": args.lr, "name": "head"},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    criterion = DecomposedLoss(
        lambda_axis=args.lambda_axis,
        lambda_sign=args.lambda_sign,
        lambda_sup=args.lambda_sup,
        lambda_stab=args.lambda_stab,
    )
    print(f"Loss: DecomposedLoss "
          f"(λ_axis={args.lambda_axis} λ_sign={args.lambda_sign} "
          f"λ_sup={args.lambda_sup} λ_stab={args.lambda_stab})")

    # ---- Initial eval ----
    print("\n--- Zero-shot evaluation (trunk only) ---")
    metrics_init = evaluate(model, test_dl, device)
    print_metrics(metrics_init, "Zero-shot")

    # ---- Train ----
    os.makedirs(args.model_dir, exist_ok=True)
    best_mean = float("inf")
    best_flip = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_sums = {}
        num_batches = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}")
        for pos, y_dir, support, labels in pbar:
            pos = pos.to(device)
            y_dir = y_dir.to(device)
            support = support.to(device)

            out = model(pos)
            targets = {"y_direction": y_dir, "y_support": support, "points": pos}
            loss_dict = criterion(out, targets)
            loss = loss_dict["total"]

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            for k, v in loss_dict.items():
                epoch_sums[k] = epoch_sums.get(k, 0.0) + v.item()
            num_batches += 1

            pbar.set_postfix(
                loss=f"{loss_dict['total'].item():.4f}",
                axis=f"{loss_dict['axis'].item():.3f}",
                sign=f"{loss_dict['sign'].item():.3f}",
            )

        scheduler.step()

        avgs = {k: v / num_batches for k, v in epoch_sums.items()}
        cur_lr = optimizer.param_groups[-1]["lr"]
        comp_str = ", ".join(f"{k}: {v:.4f}" for k, v in avgs.items()
                             if k != "total")
        print(f"  Loss: {avgs['total']:.4f} ({comp_str}), lr: {cur_lr:.6f}")

        if epoch % args.eval_every == 0 or epoch == args.epochs or epoch == 1:
            metrics = evaluate(model, test_dl, device)
            print_metrics(metrics, f"Epoch {epoch}")

            if metrics["mean_signed"] < best_mean:
                best_mean = metrics["mean_signed"]
                save_path = os.path.join(args.model_dir, "hybrid_best.pth")
                torch.save(model.state_dict(), save_path)
                print(f"  ** New best signed mean: {best_mean:.2f}° -> {save_path}")

            if metrics["flip_rate"] < best_flip:
                best_flip = metrics["flip_rate"]
                save_path = os.path.join(args.model_dir, "hybrid_best_flip.pth")
                torch.save(model.state_dict(), save_path)
                print(f"  ** New best flip rate: {best_flip:.2f}% -> {save_path}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    final_path = os.path.join(args.model_dir, f"hybrid_final_{timestamp}.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nDone. Best mean={best_mean:.2f}°  best flip={best_flip:.2f}%  "
          f"final={final_path}")


if __name__ == "__main__":
    main()

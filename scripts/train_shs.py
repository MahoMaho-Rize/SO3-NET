#!/usr/bin/env python3
"""
Training script for SHS-style signed direction upright estimator.

Usage:
    # Phase 1: frozen trunk, train context + head only
    python scripts/train_shs.py --freeze_trunk --epochs 30 --lr 1e-3 --batch_size 48

    # Phase 2: unfreeze trunk, fine-tune end-to-end
    python scripts/train_shs.py --lr_trunk 1e-5 --lr 5e-4 --epochs 20 --batch_size 16 \
        --resume models/shs_best.pth

    # Quick sanity check
    python scripts/train_shs.py --freeze_trunk --epochs 3 --batch_size 16
"""

import argparse
import os
import sys
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_shs import SHSUprightNet
from Common.loss_shs import SHSLoss
from Common.geometric_utils import angular_error_deg


# ---- Dataset (plain PyTorch, no PyG dependency) ----

class UprightDataset(Dataset):
    """Point cloud dataset with on-the-fly random SO(3) rotation.

    Returns plain tensors (no PyG Data objects needed).

    Args:
        rotations_per_object: number of virtual copies per object per epoch.
            len(dataset) = num_objects * rotations_per_object.
            Each access generates a fresh random rotation, so even with
            rotations_per_object=1 every epoch sees different rotations.
            Higher values increase effective epoch size (more gradient steps).
    """

    def __init__(self, data_dir, partition="train", num_points=2048,
                 rotations_per_object=1):
        self.num_points = num_points
        self.rotations_per_object = rotations_per_object
        self.data_original = np.load(
            os.path.join(data_dir, f"{partition}_noaug_original.npy")
        ).astype(np.float32)
        self.labels = np.load(
            os.path.join(data_dir, f"labels_{partition}_noaug.npy")
        ).astype(np.float32)
        self.pid = np.load(
            os.path.join(data_dir, f"pid_{partition}_noaug.npy")
        ).astype(np.float32)
        self.num_objects = len(self.data_original)

    def __len__(self):
        return self.num_objects * self.rotations_per_object

    def __getitem__(self, idx):
        obj_idx = idx % self.num_objects
        original = self.data_original[obj_idx][:self.num_points].copy()
        support = self.pid[obj_idx][:self.num_points].copy()
        label = int(self.labels[obj_idx].flat[0])

        original = torch.from_numpy(original)
        support = torch.from_numpy(support)

        # Random SO(3) rotation
        from scipy.spatial.transform import Rotation
        R = torch.from_numpy(Rotation.random().as_matrix().astype(np.float32))
        pos = (R @ original.T).T  # (N, 3)
        y_direction = R[:, 1]  # (3,) — GT upright is rotated y-axis

        return pos, y_direction, support, label


# ---- Evaluation ----

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_errors = []
    all_labels = []

    for pos, y_dir, support, labels in dataloader:
        pos = pos.to(device)
        y_dir = y_dir.to(device)

        out = model(pos)
        pred_up = out["up"]

        # Signed angular error (no antipodal)
        errors_signed = angular_error_deg(pred_up, y_dir, antipodal=False)
        # Also compute antipodal error for comparison
        errors_anti = angular_error_deg(pred_up, y_dir, antipodal=True)

        all_errors.append(torch.stack([errors_signed, errors_anti], dim=-1).cpu())
        all_labels.append(labels)

    all_errors = torch.cat(all_errors)  # (total, 2)
    all_labels = torch.cat(all_labels)

    err_signed = all_errors[:, 0]
    err_anti = all_errors[:, 1]

    flip_rate = (err_signed > 90.0).float().mean().item() * 100

    metrics = {
        "mean_signed": err_signed.mean().item(),
        "median_signed": err_signed.median().item(),
        "mean_anti": err_anti.mean().item(),
        "median_anti": err_anti.median().item(),
        "acc5_signed": (err_signed < 5.0).float().mean().item() * 100,
        "acc10_signed": (err_signed < 10.0).float().mean().item() * 100,
        "acc5_anti": (err_anti < 5.0).float().mean().item() * 100,
        "acc10_anti": (err_anti < 10.0).float().mean().item() * 100,
        "flip_rate": flip_rate,
    }

    # Per-category breakdown
    category_names = [
        "bed", "bench", "bottle", "bowl", "car", "chair", "cone", "cup",
        "lamp", "monitor", "sofa", "stool", "table", "toilet", "vase",
    ]
    per_cat = {}
    for cat_idx in range(15):
        mask = all_labels == cat_idx
        if mask.sum() == 0:
            continue
        cat_err_s = err_signed[mask]
        cat_err_a = err_anti[mask]
        per_cat[category_names[cat_idx]] = {
            "mean_signed": cat_err_s.mean().item(),
            "acc10_signed": (cat_err_s < 10.0).float().mean().item() * 100,
            "flip_rate": (cat_err_s > 90.0).float().mean().item() * 100,
            "mean_anti": cat_err_a.mean().item(),
        }
    metrics["per_category"] = per_cat

    return metrics


def print_metrics(metrics, header=""):
    if header:
        print(f"\n{'='*60}")
        print(f"  {header}")
        print(f"{'='*60}")
    print(f"  Signed  — Mean: {metrics['mean_signed']:.2f}°, "
          f"Median: {metrics['median_signed']:.2f}°, "
          f"Acc@5: {metrics['acc5_signed']:.1f}%, "
          f"Acc@10: {metrics['acc10_signed']:.1f}%")
    print(f"  Antipod — Mean: {metrics['mean_anti']:.2f}°, "
          f"Median: {metrics['median_anti']:.2f}°, "
          f"Acc@5: {metrics['acc5_anti']:.1f}%, "
          f"Acc@10: {metrics['acc10_anti']:.1f}%")
    print(f"  Flip rate (>90°): {metrics['flip_rate']:.2f}%")

    # Hard categories
    per_cat = metrics.get("per_category", {})
    hard_cats = ["bed", "lamp", "vase", "cup"]
    if per_cat:
        print(f"\n  {'Category':<10} {'Mean(S)':>8} {'Acc@10(S)':>10} {'Flip%':>7} {'Mean(A)':>8}")
        print(f"  {'-'*46}")
        for cat in hard_cats:
            if cat in per_cat:
                c = per_cat[cat]
                print(f"  {cat:<10} {c['mean_signed']:>7.2f}° {c['acc10_signed']:>9.1f}% "
                      f"{c['flip_rate']:>6.1f}% {c['mean_anti']:>7.2f}°")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser("SHS-style upright training")
    parser.add_argument("--data_dir", default="./datasets/uprightnet15/")
    parser.add_argument("--model_dir", default="./models/")
    parser.add_argument("--trunk_ckpt", default=None,
                        help="Path to Pang 2022 trunk checkpoint")
    parser.add_argument("--resume", default=None,
                        help="Resume from SHS checkpoint")

    # Architecture
    parser.add_argument("--context_layers", type=int, default=2)
    parser.add_argument("--context_heads", type=int, default=4)
    parser.add_argument("--head_hidden", type=int, default=512)
    parser.add_argument("--freeze_trunk", action="store_true")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_trunk", type=float, default=0.0,
                        help="Separate LR for trunk (0 = frozen)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Auxiliary support loss weight")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--rotations_per_object", type=int, default=1,
                        help="Virtual copies per object per epoch (more = more data)")
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
    print(f"Train: {len(train_ds)} objects, Test: {len(test_ds)} objects")

    # ---- Model ----
    model = SHSUprightNet(
        context_layers=args.context_layers,
        context_heads=args.context_heads,
        head_hidden=args.head_hidden,
        freeze_trunk=args.freeze_trunk or args.lr_trunk == 0,
    ).to(device)

    # Load trunk
    trunk_ckpt = args.trunk_ckpt
    if trunk_ckpt is None:
        default_ckpt = os.path.expanduser(
            "~/uprightnet-reference/model/model.pth"
        )
        if os.path.exists(default_ckpt):
            trunk_ckpt = default_ckpt
    if trunk_ckpt:
        missing, unexpected = model.load_trunk_from_ckpt(trunk_ckpt)
        print(f"Loaded trunk from {trunk_ckpt}")
        if missing:
            print(f"  Missing keys: {missing}")

    # Resume full model
    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=False)
        print(f"Resumed from {args.resume}")

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # ---- Optimizer ----
    param_groups = []
    trunk_params = list(model.trunk.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith("trunk.") and p.requires_grad]

    if args.lr_trunk > 0 and not args.freeze_trunk:
        for p in trunk_params:
            p.requires_grad = True
        param_groups.append({"params": trunk_params, "lr": args.lr_trunk})

    param_groups.append({"params": other_params, "lr": args.lr})

    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    criterion = SHSLoss(beta=args.beta)

    # ---- Initial eval ----
    print("\n--- Zero-shot evaluation (before training) ---")
    metrics_init = evaluate(model, test_dl, device)
    print_metrics(metrics_init, "Zero-shot")

    # ---- Training loop ----
    os.makedirs(args.model_dir, exist_ok=True)
    best_mean = float("inf")
    best_flip = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_trunk or args.lr_trunk == 0:
            model.trunk.eval()  # keep BN in eval mode

        epoch_loss_dir = 0.0
        epoch_loss_sup = 0.0
        epoch_loss_total = 0.0
        num_batches = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}")
        for pos, y_dir, support, labels in pbar:
            pos = pos.to(device)
            y_dir = y_dir.to(device)
            support = support.to(device)

            out = model(pos)
            targets = {"y_direction": y_dir, "y_support": support}
            loss_dict = criterion(out, targets)
            loss = loss_dict["total"]

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss_total += loss_dict["total"].item()
            epoch_loss_dir += loss_dict["direction"].item()
            epoch_loss_sup += loss_dict["support"].item()
            num_batches += 1

            pbar.set_postfix(
                loss=f"{loss_dict['total'].item():.4f}",
                dir=f"{loss_dict['direction'].item():.4f}",
            )

        scheduler.step()

        avg_total = epoch_loss_total / num_batches
        avg_dir = epoch_loss_dir / num_batches
        avg_sup = epoch_loss_sup / num_batches
        cur_lr = optimizer.param_groups[-1]["lr"]
        print(f"  Loss: {avg_total:.4f} (dir: {avg_dir:.4f}, sup: {avg_sup:.4f}), lr: {cur_lr:.6f}")

        # ---- Evaluate ----
        if epoch % args.eval_every == 0 or epoch == args.epochs or epoch == 1:
            metrics = evaluate(model, test_dl, device)
            print_metrics(metrics, f"Epoch {epoch}")

            # Save best by signed mean error
            if metrics["mean_signed"] < best_mean:
                best_mean = metrics["mean_signed"]
                save_path = os.path.join(args.model_dir, "shs_best.pth")
                torch.save(model.state_dict(), save_path)
                print(f"  ** New best signed mean: {best_mean:.2f}° -> {save_path}")

            if metrics["flip_rate"] < best_flip:
                best_flip = metrics["flip_rate"]
                save_path = os.path.join(args.model_dir, "shs_best_flip.pth")
                torch.save(model.state_dict(), save_path)
                print(f"  ** New best flip rate: {best_flip:.2f}% -> {save_path}")

    # Save final
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    torch.save(
        model.state_dict(),
        os.path.join(args.model_dir, f"shs_final_{timestamp}.pth"),
    )
    print(f"\nTraining complete. Best signed mean: {best_mean:.2f}°, Best flip: {best_flip:.2f}%")


if __name__ == "__main__":
    main()

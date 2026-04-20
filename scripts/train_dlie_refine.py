"""Train DifferentiableLieUprightRefineNet on UprightNet15.

Loss (per batch):
    L = Σ_{t=0..T} γ^(T-t) · geodesic(R_t @ e_y, gt_up)

where:
    - R_iters[0] is the differentiable coarse-init rotation (trunk + SVD).
      Supervising it lets the trunk fine-tune end-to-end.
    - R_iters[1..T] are the iterative refinement rotations.
    - γ=0.8 by default → weights [0.512, 0.64, 0.8, 1.0] for T=3.

Training details:
    - Trunk (pretrained from Pang 2022) uses a 10x smaller LR than the head
      so we do not wash out its support-classification prior.
    - Each epoch samples the 111k rotated training set in random order.
    - Full 37k test set evaluation runs at epoch end (coarse eval every
      epoch, per-category breakdown every 5 epochs).
    - Best-by-test-acc@10 ckpt is saved under models/dlie_refine_best.pth.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from network_lie import DifferentiableLieUprightRefineNet


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


class UprightNet15RotatedDataset(Dataset):
    """Memory-mapped loader for the 100x-rotated UprightNet15 split.

    Returns:
        points: (N, 3) rotated point cloud
        gt_up:  (3,) ground-truth upright direction = rotm[:, 1]
        label:  () int category label
    """

    def __init__(self, data_dir, partition="train", num_points=2048):
        self.num_points = num_points
        d = Path(data_dir)
        self.points = np.load(d / f"{partition}_rotation.npy", mmap_mode="r")
        self.rotm = np.load(d / f"rotm_{partition}.npy", mmap_mode="r")
        self.labels = np.load(d / f"labels_{partition}.npy", mmap_mode="r")

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        P = np.asarray(self.points[idx][: self.num_points], dtype=np.float32).copy()
        R = np.asarray(self.rotm[idx], dtype=np.float32).copy()
        up = R[:, 1]  # gt upright direction in world frame
        label = int(self.labels[idx].item())
        return (
            torch.from_numpy(P),
            torch.from_numpy(up),
            torch.tensor(label, dtype=torch.long),
        )


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------


def geodesic_from_rotation_up_column(R, gt_up, eps=1e-7):
    """Geodesic loss (radians) between the second column of R and gt_up."""
    up_pred = R[:, :, 1]
    cos = (up_pred * gt_up).sum(-1).clamp(-1 + eps, 1 - eps)
    return torch.acos(cos)  # (B,) in radians


def compute_multi_iter_loss(R_iters, gt_up, gamma=0.8):
    """Compute Σ γ^(T-t) · geodesic(R_t, gt_up). R_iters has len T+1."""
    T = len(R_iters) - 1  # num_iters
    total = 0.0
    per_iter = []
    # Weights: coarse init (t=0) has weight γ^T; final (t=T) has weight 1.
    for t, R in enumerate(R_iters):
        w = gamma ** (T - t)
        l = geodesic_from_rotation_up_column(R, gt_up).mean()
        total = total + w * l
        per_iter.append(l.item())
    return total, per_iter


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, device, verbose=False):
    """Return per-sample errors (degrees) for the final prediction."""
    model.eval()
    errs = []
    labels = []
    errs_init = []
    for P, gt_up, label in loader:
        P = P.to(device, non_blocking=True)
        gt_up = gt_up.to(device, non_blocking=True)
        out = model(P)
        cos_f = (out["up"] * gt_up).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
        err_f = torch.acos(cos_f) * (180.0 / math.pi)
        cos_0 = (out["up_0"] * gt_up).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
        err_0 = torch.acos(cos_0) * (180.0 / math.pi)
        errs.append(err_f.cpu())
        errs_init.append(err_0.cpu())
        labels.append(label)
    errs = torch.cat(errs).numpy()
    errs_init = torch.cat(errs_init).numpy()
    labels = torch.cat(labels).numpy()
    model.train()
    return errs, errs_init, labels


def summarise(name, err):
    mean = err.mean()
    med = float(np.median(err))
    acc5 = float((err < 5).mean() * 100)
    acc10 = float((err < 10).mean() * 100)
    flip = float((err > 90).mean() * 100)
    print(f"  [{name}]  mean={mean:6.2f}°  median={med:5.2f}°  "
          f"acc@5={acc5:5.2f}%  acc@10={acc10:5.2f}%  flip={flip:5.2f}%")
    return mean, acc10, flip


def per_category(err, labels):
    CATS = ["bed", "bench", "bottle", "bowl", "car", "chair", "cone", "cup",
            "lamp", "monitor", "sofa", "stool", "table", "toilet", "vase"]
    lines = []
    for lab in range(15):
        m = labels == lab
        if m.sum() == 0:
            continue
        e = err[m]
        lines.append(f"    {CATS[lab]:>8}  N={int(m.sum()):>4}  "
                     f"mean={e.mean():6.2f}  acc@10={(e<10).mean()*100:5.2f}%  "
                     f"flip={(e>90).mean()*100:5.2f}%")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(REPO / "datasets" / "uprightnet15"))
    ap.add_argument("--ref_ckpt", default=str(
        REPO.parent / "uprightnet-reference" / "model" / "model.pth"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--num_points", type=int, default=2048)
    ap.add_argument("--num_iters", type=int, default=3)
    ap.add_argument("--max_angle_schedule", type=float, nargs="+",
                    default=[math.pi, math.pi / 4, math.pi / 18])
    ap.add_argument("--gamma", type=float, default=0.8,
                    help="loss weighting: Σ γ^(T-t) · geo(R_t, gt)")
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_trunk", type=float, default=1e-4,
                    help="set 0 to freeze trunk")
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_interval", type=int, default=100,
                    help="print training loss every N batches")
    ap.add_argument("--save_name", default="dlie_refine")
    ap.add_argument("--eval_at_start", action="store_true",
                    help="evaluate the untrained model first (baseline)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda")
    print(f"Schedule: {[f'{math.degrees(a):.0f}°' for a in args.max_angle_schedule]}")
    print(f"Gamma={args.gamma}  lr_head={args.lr_head}  lr_trunk={args.lr_trunk}")

    # ---- Data ----
    train_ds = UprightNet15RotatedDataset(args.data_dir, "train", args.num_points)
    test_ds = UprightNet15RotatedDataset(args.data_dir, "test", args.num_points)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} samples  Test: {len(test_ds)} samples  "
          f"bs_train={args.batch_size} bs_eval={args.eval_batch_size}")

    # ---- Model ----
    model = DifferentiableLieUprightRefineNet(
        num_iters=args.num_iters,
        max_angle_schedule=args.max_angle_schedule,
    ).to(device)
    model.load_backbone_from_ckpt(args.ref_ckpt)
    print(f"Loaded reference trunk from {args.ref_ckpt}")
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params total={n_params:,} trainable={n_trainable:,}")

    # ---- Optimiser: separate LR for trunk vs head ----
    trunk_params = list(model.trunk.parameters())
    head_params = list(model.refine_head.parameters())
    param_groups = []
    if args.lr_trunk > 0:
        param_groups.append({"params": trunk_params, "lr": args.lr_trunk,
                             "name": "trunk"})
    else:
        for p in trunk_params:
            p.requires_grad = False
        print("Trunk frozen (lr_trunk=0)")
    param_groups.append({"params": head_params, "lr": args.lr_head,
                         "name": "head"})
    optim = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay,
                              betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * len(train_loader),
        eta_min=min(args.lr_head, max(args.lr_trunk, 1e-6)) * 0.01,
    )

    # ---- Eval at start (baseline) ----
    ckpt_dir = REPO / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc10 = -1.0
    best_mean = float("inf")
    if args.eval_at_start:
        print()
        print("=== Eval at start (no training yet) ===")
        errs, errs_init, labels_np = evaluate(model, test_loader, device)
        summarise("init ", errs_init)
        summarise("final", errs)
        print(per_category(errs, labels_np))

    # ---- Training loop ----
    train_log = []
    for epoch in range(1, args.epochs + 1):
        print()
        print(f"=== Epoch {epoch}/{args.epochs} ===")
        model.train()
        t_start = time.time()
        running_total = 0.0
        running_per_iter = None
        n_samples = 0

        pbar = tqdm(train_loader, total=len(train_loader))
        for bi, (P, gt_up, _lab) in enumerate(pbar):
            P = P.to(device, non_blocking=True)
            gt_up = gt_up.to(device, non_blocking=True)

            out = model(P)
            loss, per_iter = compute_multi_iter_loss(
                out["R_iters"], gt_up, gamma=args.gamma,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            # Clip grads to stabilise SVD eigh gradients at near-degenerate covs
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=5.0,
            )
            optim.step()
            scheduler.step()

            bs = P.shape[0]
            running_total += loss.item() * bs
            if running_per_iter is None:
                running_per_iter = [v * bs for v in per_iter]
            else:
                for i, v in enumerate(per_iter):
                    running_per_iter[i] += v * bs
            n_samples += bs

            if (bi + 1) % args.log_interval == 0:
                avg = running_total / n_samples
                per = [v / n_samples for v in running_per_iter]
                pbar.set_postfix({
                    "loss": f"{avg:.4f}",
                    "geo_init": f"{math.degrees(per[0]):.2f}°",
                    "geo_final": f"{math.degrees(per[-1]):.2f}°",
                })

        avg = running_total / n_samples
        per = [v / n_samples for v in running_per_iter]
        dt = time.time() - t_start
        print(f"  train loss={avg:.4f}  "
              f"per-iter (geo in deg)={[f'{math.degrees(p):.2f}' for p in per]}  "
              f"time={dt/60:.1f} min")

        # ---- Eval ----
        errs, errs_init, labels_np = evaluate(model, test_loader, device)
        print("  Test:")
        m_init, a10_init, flip_init = summarise("init ", errs_init)
        m_final, a10_final, flip_final = summarise("final", errs)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(per_category(errs, labels_np))

        # ---- Save best ----
        if a10_final > best_acc10:
            best_acc10 = a10_final
            best_mean = m_final
            ck = ckpt_dir / f"{args.save_name}_best.pth"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "acc10": a10_final,
                "mean": m_final,
                "flip": flip_final,
                "args": vars(args),
            }, ck)
            print(f"  [Saved] {ck}  (acc@10={a10_final:.2f}% mean={m_final:.2f}°)")

        train_log.append({
            "epoch": epoch, "train_loss": avg,
            "test_mean_init": m_init, "test_mean_final": m_final,
            "test_acc10_init": a10_init, "test_acc10_final": a10_final,
            "test_flip_init": flip_init, "test_flip_final": flip_final,
        })

    # ---- Final save ----
    ck_final = ckpt_dir / f"{args.save_name}_final.pth"
    torch.save({
        "epoch": args.epochs,
        "model": model.state_dict(),
        "args": vars(args),
    }, ck_final)
    print(f"\nSaved {ck_final}")

    print()
    print("=" * 60)
    print("Training summary:")
    for row in train_log:
        print(f"  ep{row['epoch']:>2}  "
              f"loss={row['train_loss']:.4f}  "
              f"test_mean={row['test_mean_final']:6.2f}°  "
              f"acc@10={row['test_acc10_final']:5.2f}%  "
              f"flip={row['test_flip_final']:5.2f}%")
    print(f"  Best: acc@10={best_acc10:.2f}%  mean={best_mean:.2f}°")


if __name__ == "__main__":
    main()

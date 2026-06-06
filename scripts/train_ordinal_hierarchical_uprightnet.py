#!/usr/bin/env python3
"""Train ordinal point-wise hierarchy classification for partial uprightness."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_hierarchical_uprightnet import (
    DGCNNHierarchyNet,
    HierarchyDataset,
    PointNetHierarchyNet,
    _weighted_ls_direction,
    angular_error_deg,
    miou_from_confusion,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train ordinal hierarchical partial UprightNet")
    p.add_argument("--train-npz", default="datasets/upright_hierarchy_npz/train.npz")
    p.add_argument("--test-npz", default="datasets/upright_hierarchy_npz/test.npz")
    p.add_argument("--out-dir", default="models/ordinal_hierarchical_uprightnet_dgcnn")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--arch", default="dgcnn", choices=("dgcnn", "pointnet"))
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--threshold-balance", action="store_true")
    p.add_argument("--data-parallel", action="store_true")
    p.add_argument("--log-csv", default="")
    p.add_argument("--direction-method", default="ls", choices=("ls", "weighted_ls"))
    p.add_argument("--confidence-power", type=float, default=1.0)
    p.add_argument("--min-confidence-weight", type=float, default=0.05)
    return p.parse_args()


def ordinal_targets(labels: torch.Tensor, num_levels: int) -> torch.Tensor:
    thresholds = torch.arange(num_levels - 1, device=labels.device).view(1, -1, 1)
    return (labels.unsqueeze(1) > thresholds).float()


def labels_from_ordinal_logits(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) >= 0.5).long().sum(dim=1)


def score_from_ordinal_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits).sum(dim=1) / max(logits.shape[1], 1)


def confidence_from_ordinal_logits(
    logits: torch.Tensor,
    confidence_power: float,
    min_confidence_weight: float,
) -> torch.Tensor:
    probs = torch.sigmoid(logits).clamp(1e-8, 1.0 - 1e-8)
    entropy = -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())
    entropy = entropy.mean(dim=1) / np.log(2.0)
    confidence = (1.0 - entropy).clamp(0.0, 1.0)
    if confidence_power != 1.0:
        confidence = confidence.pow(confidence_power)
    return confidence.clamp_min(min_confidence_weight)


def direction_from_ordinal_logits(
    points: torch.Tensor,
    logits: torch.Tensor,
    method: str,
    confidence_power: float,
    min_confidence_weight: float,
) -> torch.Tensor:
    score = score_from_ordinal_logits(logits)
    if method == "weighted_ls":
        weights = confidence_from_ordinal_logits(
            logits, confidence_power, min_confidence_weight
        )
    else:
        weights = torch.ones_like(score)
    direction, _point_center, _score_center = _weighted_ls_direction(points, score, weights)
    return F.normalize(direction, dim=1, eps=1e-6)


def threshold_pos_weight(hist: np.ndarray, device: torch.device) -> torch.Tensor:
    hist = hist.astype(np.float64)
    weights = []
    for threshold in range(len(hist) - 1):
        neg = hist[: threshold + 1].sum()
        pos = hist[threshold + 1 :].sum()
        weights.append(neg / max(pos, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device).view(1, -1, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_levels: int,
    pos_weight: torch.Tensor | None,
    direction_method: str,
    confidence_power: float,
    min_confidence_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_points = 0
    correct = 0
    errors = []
    conf = np.zeros((num_levels, num_levels), dtype=np.int64)

    for points, labels, gt_up, _cat in loader:
        points = points.to(device)
        labels = labels.to(device)
        gt_up = gt_up.to(device)
        targets = ordinal_targets(labels, num_levels)
        logits = model(points)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )
        pred = labels_from_ordinal_logits(logits)

        total_loss += float(loss.detach().cpu()) * labels.numel()
        total_points += labels.numel()
        correct += int((pred == labels).sum().item())

        pred_up = direction_from_ordinal_logits(
            points,
            logits,
            direction_method,
            confidence_power,
            min_confidence_weight,
        )
        errors.append(angular_error_deg(pred_up, gt_up).detach().cpu())

        y = labels.detach().cpu().numpy().reshape(-1)
        p = pred.detach().cpu().numpy().reshape(-1)
        np.add.at(conf, (y, p), 1)

    err = torch.cat(errors).numpy() if errors else np.asarray([], dtype=np.float32)
    return {
        "loss": total_loss / max(total_points, 1),
        "point_acc": correct / max(total_points, 1),
        "miou": miou_from_confusion(conf),
        "mean_err": float(err.mean()) if len(err) else float("nan"),
        "median_err": float(np.median(err)) if len(err) else float("nan"),
        "acc5": float((err < 5).mean()) if len(err) else 0.0,
        "acc10": float((err < 10).mean()) if len(err) else 0.0,
        "acc30": float((err < 30).mean()) if len(err) else 0.0,
        "flip": float((err > 90).mean()) if len(err) else 0.0,
    }


def write_log(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_ds = HierarchyDataset(args.train_npz, args.num_points, args.seed, augment=True)
    test_ds = HierarchyDataset(args.test_npz, args.num_points, args.seed + 100000, augment=False)
    if train_ds.num_levels != test_ds.num_levels:
        raise ValueError("train/test num_levels mismatch")
    num_levels = train_ds.num_levels
    num_thresholds = num_levels - 1

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if args.arch == "dgcnn":
        model = DGCNNHierarchyNet(num_thresholds, args.dropout).to(device)
    else:
        model = PointNetHierarchyNet(num_thresholds, args.hidden, args.dropout).to(device)
    if args.data_parallel:
        if device.type != "cuda":
            raise ValueError("--data-parallel requires CUDA")
        if torch.cuda.device_count() < 2:
            raise ValueError("--data-parallel requested but fewer than 2 visible GPUs")
        model = nn.DataParallel(model)
        print(f"data_parallel=True visible_cuda_devices={torch.cuda.device_count()}")

    hist = np.load(args.train_npz, allow_pickle=False)["level_histogram"]
    pos_weight = threshold_pos_weight(hist, device) if args.threshold_balance else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv) if args.log_csv else out_dir / "train_log.csv"
    best_acc10 = -1.0

    print(f"device={device}")
    print(f"arch={args.arch} ordinal_thresholds={num_thresholds}")
    print(
        f"direction_method={args.direction_method} confidence_power={args.confidence_power} "
        f"min_confidence_weight={args.min_confidence_weight}"
    )
    print(f"train_clouds={len(train_ds)} test_clouds={len(test_ds)} levels={num_levels}")
    if pos_weight is not None:
        flat = pos_weight.detach().cpu().view(-1).tolist()
        print(f"threshold_pos_weight={[round(float(x), 4) for x in flat]}")

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        running = 0.0
        count = 0
        for points, labels, _gt_up, _cat in train_loader:
            points = points.to(device)
            labels = labels.to(device)
            targets = ordinal_targets(labels, num_levels)
            optimizer.zero_grad(set_to_none=True)
            logits = model(points)
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * labels.numel()
            count += labels.numel()

        scheduler.step()
        metrics = evaluate(
            model,
            test_loader,
            device,
            num_levels,
            pos_weight,
            args.direction_method,
            args.confidence_power,
            args.min_confidence_weight,
        )
        train_loss = running / max(count, 1)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        write_log(log_csv, row)

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} point_acc={metrics['point_acc']*100:.2f}% "
            f"miou={metrics['miou']*100:.2f}% mean={metrics['mean_err']:.2f} "
            f"median={metrics['median_err']:.2f} acc10={metrics['acc10']*100:.2f}% "
            f"flip={metrics['flip']*100:.2f}%",
            flush=True,
        )

        if metrics["acc10"] >= best_acc10:
            best_acc10 = metrics["acc10"]
            torch.save(
                {
                    "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    "args": vars(args),
                    "num_levels": num_levels,
                    "ordinal": True,
                    "metrics": metrics,
                },
                out_dir / "best.pth",
            )
            print(f"[save] {out_dir / 'best.pth'} acc10={best_acc10*100:.2f}%")

    torch.save(
        {
            "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "args": vars(args),
            "num_levels": num_levels,
            "ordinal": True,
        },
        out_dir / "final.pth",
    )
    print(f"[done] wrote {out_dir / 'final.pth'}")


if __name__ == "__main__":
    main()

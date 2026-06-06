#!/usr/bin/env python3
"""Evaluate an ensemble of nominal hierarchical uprightness checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from train_hierarchical_uprightnet import (
    DGCNNHierarchyNet,
    HierarchyDataset,
    PointNetHierarchyNet,
    angular_error_deg,
    direction_from_level_logits,
    miou_from_confusion,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate hierarchical uprightness ensemble")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--test-npz", default="")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--direction-method", default="ls", choices=("ls", "weighted_ls"))
    p.add_argument("--confidence-power", type=float, default=1.0)
    p.add_argument("--min-confidence-weight", type=float, default=0.05)
    return p.parse_args()


def build_model(ckpt: dict) -> torch.nn.Module:
    ckpt_args = ckpt.get("args", {})
    num_levels = int(ckpt.get("num_levels", 5))
    arch = ckpt_args.get("arch", "dgcnn")
    if ckpt.get("ordinal", False):
        raise ValueError("ordinal checkpoints are not supported by this ensemble script")
    if arch == "dgcnn":
        model = DGCNNHierarchyNet(num_levels, float(ckpt_args.get("dropout", 0.2)))
    elif arch == "pointnet":
        model = PointNetHierarchyNet(
            num_levels,
            int(ckpt_args.get("hidden", 512)),
            float(ckpt_args.get("dropout", 0.2)),
        )
    else:
        raise ValueError(f"unsupported checkpoint arch: {arch}")
    model.load_state_dict(ckpt["model"])
    return model


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoints = [torch.load(path, map_location="cpu") for path in args.checkpoints]
    num_levels = int(checkpoints[0].get("num_levels", 5))
    for path, ckpt in zip(args.checkpoints, checkpoints):
        if int(ckpt.get("num_levels", num_levels)) != num_levels:
            raise ValueError(f"{path}: num_levels mismatch")

    models = [build_model(ckpt).to(device).eval() for ckpt in checkpoints]
    ckpt_args = checkpoints[0].get("args", {})
    test_npz = args.test_npz or ckpt_args.get(
        "test_npz", "datasets/upright_hierarchy_npz/test.npz"
    )
    seed = int(ckpt_args.get("seed", 2026)) + 100000
    ds = HierarchyDataset(test_npz, args.num_points, seed, augment=False)
    if args.limit > 0:
        ds = Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    total_loss = 0.0
    total_points = 0
    correct = 0
    errors = []
    conf = np.zeros((num_levels, num_levels), dtype=np.int64)

    for points, labels, gt_up, _cat in loader:
        points = points.to(device)
        labels = labels.to(device)
        gt_up = gt_up.to(device)
        logits_sum = None
        for model in models:
            logits = model(points)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits = logits_sum / float(len(models))
        loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(dim=1)

        total_loss += float(loss.detach().cpu()) * labels.numel()
        total_points += labels.numel()
        correct += int((pred == labels).sum().item())

        pred_up = direction_from_level_logits(
            points,
            logits,
            method=args.direction_method,
            confidence_power=args.confidence_power,
            min_confidence_weight=args.min_confidence_weight,
        )
        errors.append(angular_error_deg(pred_up, gt_up).detach().cpu())

        y = labels.detach().cpu().numpy().reshape(-1)
        p = pred.detach().cpu().numpy().reshape(-1)
        np.add.at(conf, (y, p), 1)

    err = torch.cat(errors).numpy() if errors else np.asarray([], dtype=np.float32)
    print(f"checkpoints={len(args.checkpoints)}")
    for path in args.checkpoints:
        print(f"  {Path(path)}")
    print(f"device={device} levels={num_levels} test_clouds={len(ds)}")
    print(
        f"method=ensemble_{args.direction_method} "
        f"loss={total_loss / max(total_points, 1):.4f} "
        f"point_acc={correct / max(total_points, 1) * 100:.2f}% "
        f"miou={miou_from_confusion(conf) * 100:.2f}% "
        f"mean={float(err.mean()):.2f} median={float(np.median(err)):.2f} "
        f"acc5={(err < 5).mean() * 100:.2f}% "
        f"acc10={(err < 10).mean() * 100:.2f}% "
        f"acc30={(err < 30).mean() * 100:.2f}% "
        f"flip={(err > 90).mean() * 100:.2f}%"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a trained hierarchical uprightness checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from train_hierarchical_uprightnet import (
    DGCNNHierarchyNet,
    HierarchyDataset,
    PointNetHierarchyNet,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate hierarchical partial UprightNet")
    p.add_argument("--checkpoint", default="models/hierarchical_uprightnet_dgcnn/best.pth")
    p.add_argument("--test-npz", default="")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="0 means the full test set")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument(
        "--methods",
        nargs="+",
        default=["ls", "weighted_ls", "trimmed_ls"],
        choices=("ls", "weighted_ls", "trimmed_ls"),
    )
    p.add_argument("--trim-fraction", type=float, default=0.10)
    p.add_argument("--confidence-power", type=float, default=1.0)
    p.add_argument("--min-confidence-weight", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    num_levels = int(ckpt.get("num_levels", 5))
    arch = ckpt_args.get("arch", "dgcnn")
    test_npz = args.test_npz or ckpt_args.get(
        "test_npz", "datasets/upright_hierarchy_npz/test.npz"
    )

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
    model.to(device)

    ds = HierarchyDataset(
        test_npz,
        num_points=args.num_points,
        seed=int(ckpt_args.get("seed", 2026)) + 100000,
        augment=False,
    )
    if args.limit > 0:
        ds = Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"checkpoint={Path(args.checkpoint)}")
    print(f"device={device} arch={arch} levels={num_levels} test_clouds={len(ds)}")
    print(f"test_npz={test_npz}")

    for method in args.methods:
        metrics = evaluate(
            model,
            loader,
            device,
            num_levels,
            loss_weight=None,
            direction_method=method,
            trim_fraction=args.trim_fraction,
            confidence_power=args.confidence_power,
            min_confidence_weight=args.min_confidence_weight,
        )
        print(
            f"method={method} loss={metrics['loss']:.4f} "
            f"point_acc={metrics['point_acc']*100:.2f}% "
            f"miou={metrics['miou']*100:.2f}% mean={metrics['mean_err']:.2f} "
            f"median={metrics['median_err']:.2f} acc5={metrics['acc5']*100:.2f}% "
            f"acc10={metrics['acc10']*100:.2f}% acc30={metrics['acc30']*100:.2f}% "
            f"flip={metrics['flip']*100:.2f}% "
            f"oracle_mean={metrics['oracle_mean_err']:.2f} "
            f"oracle_acc10={metrics['oracle_acc10']*100:.2f}% "
            f"oracle_gap={metrics['oracle_gap_mean']:.2f}"
        )


if __name__ == "__main__":
    main()

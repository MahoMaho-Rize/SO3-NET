#!/usr/bin/env python3
"""Evaluate label-oracle upright recovery upper bounds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from train_hierarchical_uprightnet import (
    HierarchyDataset,
    angular_error_deg,
    direction_from_level_labels,
)
from train_pairwise_uprightnet import (
    PairwiseUprightDataset,
    deterministic_pairs,
    direction_from_pair_targets,
    pair_targets,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate hierarchy and pairwise label oracles")
    p.add_argument("--npz", default="datasets/upright_hierarchy_npz/test.npz")
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--same-threshold-ratio", type=float, default=0.03)
    p.add_argument("--skip-pairwise", action="store_true")
    p.add_argument(
        "--pair-counts",
        type=int,
        nargs="+",
        default=[4096, 8192, 32768],
        help="Number of deterministic point pairs per cloud for pairwise oracle.",
    )
    return p.parse_args()


def summarize(name: str, errors: np.ndarray) -> None:
    print(
        f"{name}: mean={float(errors.mean()):.2f} "
        f"median={float(np.median(errors)):.2f} "
        f"acc5={(errors < 5).mean() * 100:.2f}% "
        f"acc10={(errors < 10).mean() * 100:.2f}% "
        f"acc30={(errors < 30).mean() * 100:.2f}% "
        f"flip={(errors > 90).mean() * 100:.2f}%"
    )


@torch.no_grad()
def eval_hierarchy_oracle(
    npz_path: str,
    num_points: int,
    batch_size: int,
    num_workers: int,
    limit: int,
    seed: int,
    device: torch.device,
) -> None:
    data = np.load(npz_path, allow_pickle=False)
    if "level_labels" not in data.files:
        print("hierarchy_ls: skipped, npz has no level_labels")
        return
    ds: Dataset = HierarchyDataset(npz_path, num_points, seed + 100000, augment=False)
    num_levels = int(data["num_levels"])
    if limit > 0:
        ds = Subset(ds, range(min(limit, len(ds))))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    errors = []
    for points, labels, gt_up, _cat in loader:
        points = points.to(device)
        labels = labels.to(device)
        gt_up = gt_up.to(device)
        pred_up = direction_from_level_labels(points, labels, num_levels)
        errors.append(angular_error_deg(pred_up, gt_up).detach().cpu())
    summarize("hierarchy_ls_oracle", torch.cat(errors).numpy())


@torch.no_grad()
def eval_pairwise_oracle(
    npz_path: str,
    num_points: int,
    batch_size: int,
    num_workers: int,
    limit: int,
    seed: int,
    device: torch.device,
    same_threshold_ratio: float,
    pair_count: int,
) -> None:
    ds: Dataset = PairwiseUprightDataset(npz_path, num_points, seed + 100000, augment=False)
    if limit > 0:
        ds = Subset(ds, range(min(limit, len(ds))))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    errors = []
    for points, gt_up, _cat in loader:
        points = points.to(device)
        gt_up = gt_up.to(device)
        pair_i, pair_j = deterministic_pairs(
            points.shape[0], points.shape[1], pair_count, device
        )
        target = pair_targets(points, gt_up, pair_i, pair_j, same_threshold_ratio)
        pred_up = direction_from_pair_targets(points, pair_i, pair_j, target)
        errors.append(angular_error_deg(pred_up, gt_up).detach().cpu())
    summarize(f"pairwise_ls_oracle pairs={pair_count}", torch.cat(errors).numpy())


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"npz={Path(args.npz)}")
    print(
        f"device={device} num_points={args.num_points} limit={args.limit} "
        f"same_threshold_ratio={args.same_threshold_ratio}"
    )
    eval_hierarchy_oracle(
        args.npz,
        args.num_points,
        args.batch_size,
        args.num_workers,
        args.limit,
        args.seed,
        device,
    )
    if not args.skip_pairwise:
        for pair_count in args.pair_counts:
            eval_pairwise_oracle(
                args.npz,
                args.num_points,
                args.batch_size,
                args.num_workers,
                args.limit,
                args.seed,
                device,
                args.same_threshold_ratio,
                pair_count,
            )


if __name__ == "__main__":
    main()

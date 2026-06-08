#!/usr/bin/env python3
"""Repack hierarchy NPZ files for pairwise upright-order experiments.

The pairwise experiment does not need absolute hierarchy labels.  It only needs
the observed partial points and the ground-truth upright vector so pair labels
can be sampled online during training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build compact pairwise upright-order NPZ files")
    p.add_argument("--source-dir", default="datasets/upright_hierarchy_npz")
    p.add_argument("--out-dir", default="datasets/upright_pairwise_npz")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--limit", type=int, default=0, help="0 means all clouds")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--compressed",
        action="store_true",
        help="Use np.savez_compressed. Smaller but much slower for large train splits.",
    )
    return p.parse_args()


def normalize_up(up: np.ndarray) -> np.ndarray:
    out = up.astype(np.float32, copy=True)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-12)


def repack_split(
    source_path: Path,
    out_path: Path,
    num_points: int,
    limit: int,
    seed: int,
    compressed: bool,
) -> None:
    data = np.load(source_path, allow_pickle=False)
    points = data["points"].astype(np.float32)
    gt_up = normalize_up(data["gt_up"])
    category_id = data["category_id"].astype(np.int64)

    if limit > 0:
        keep = min(limit, points.shape[0])
        points = points[:keep]
        gt_up = gt_up[:keep]
        category_id = category_id[:keep]

    if points.shape[1] > num_points:
        rng = np.random.default_rng(seed)
        choice = rng.choice(points.shape[1], size=num_points, replace=False)
        points = points[:, choice, :]
    elif points.shape[1] < num_points:
        raise ValueError(
            f"{source_path}: has {points.shape[1]} points, fewer than --num-points={num_points}"
        )

    payload = {
        "points": points,
        "gt_up": gt_up,
        "category_id": category_id,
        "num_points": np.asarray(points.shape[1], dtype=np.int64),
    }
    for key in (
        "category_names",
        "rel_path",
        "source_path",
        "source_up_axis",
        "bottom_band_retained_ratio",
    ):
        if key in data.files:
            value = data[key]
            if limit > 0 and getattr(value, "shape", ())[:1] == (data["points"].shape[0],):
                value = value[: points.shape[0]]
            payload[key] = value

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compressed else np.savez
    writer(out_path, **payload)
    print(
        f"[ok] {source_path} -> {out_path} "
        f"clouds={points.shape[0]} points={points.shape[1]} compressed={compressed}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    for split_id, split in enumerate(args.splits):
        source_path = source_dir / f"{split}.npz"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        repack_split(
            source_path,
            out_dir / f"{split}.npz",
            args.num_points,
            args.limit,
            args.seed + split_id * 1000003,
            args.compressed,
        )


if __name__ == "__main__":
    main()

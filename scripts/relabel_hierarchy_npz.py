#!/usr/bin/env python3
"""Re-quantize existing hierarchy NPZ files with a different level count."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from build_hierarchy_npz import hierarchy_labels, read_off_vertices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Relabel existing hierarchy NPZ files")
    p.add_argument("--source-dir", default="datasets/upright_hierarchy_npz")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--full-root", default="datasets/uprightnet15")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--num-levels", type=int, required=True)
    p.add_argument("--source-up-axis", choices=("x", "y", "z"), default="z")
    p.add_argument("--compress", action="store_true")
    return p.parse_args()


def resolve_source_path(raw_source: str, rel_path: str, full_root: Path) -> Path:
    source_path = Path(raw_source)
    if source_path.exists():
        return source_path

    rel = Path(rel_path)
    if len(rel.parts) >= 3:
        category, split, filename = rel.parts[0], rel.parts[1], rel.parts[-1]
        stem = re.sub(r"_view\d+$", "", Path(filename).stem)
        fallback = full_root / category / split / f"{stem}.off"
        if fallback.exists():
            return fallback

    return source_path


def relabel_split(
    source_path: Path,
    out_path: Path,
    full_root: Path,
    num_levels: int,
    axis_idx: int,
    compress: bool,
) -> None:
    data = np.load(source_path, allow_pickle=False)
    points = data["points"].astype(np.float32)
    rel_paths = data["rel_path"].astype(str)
    source_paths = data["source_path"].astype(str)
    labels = np.empty(points.shape[:2], dtype=np.int64)
    bbox_cache: dict[Path, tuple[float, float]] = {}

    for i in range(points.shape[0]):
        full_path = resolve_source_path(source_paths[i], rel_paths[i], full_root)
        if full_path not in bbox_cache:
            vertices = read_off_vertices(full_path)
            bbox_cache[full_path] = (
                float(vertices[:, axis_idx].min()),
                float(vertices[:, axis_idx].max()),
            )
        bbox_min, bbox_max = bbox_cache[full_path]
        labels[i] = hierarchy_labels(points[i], bbox_min, bbox_max, axis_idx, num_levels)
        if (i + 1) % 500 == 0:
            print(f"[label] {source_path.name} {i + 1}/{points.shape[0]}", flush=True)

    payload = {key: data[key] for key in data.files if key != "level_labels"}
    payload["level_labels"] = labels
    payload["num_levels"] = np.asarray(num_levels, dtype=np.int64)
    payload["level_histogram"] = np.bincount(
        labels.reshape(-1), minlength=num_levels
    ).astype(np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(out_path, **payload)
    print(
        f"[ok] {source_path} -> {out_path} clouds={points.shape[0]} "
        f"points={points.shape[1]} levels={num_levels} compressed={compress}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.num_levels < 2:
        raise SystemExit("--num-levels must be >= 2")

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    full_root = Path(args.full_root).resolve()
    axis_idx = "xyz".index(args.source_up_axis)

    for split in args.splits:
        source_path = source_dir / f"{split}.npz"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        relabel_split(
            source_path,
            out_dir / f"{split}.npz",
            full_root,
            args.num_levels,
            axis_idx,
            args.compress,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build compact NPZ files for candidate-conditioned uprightness training.

The script reads point-only OFF files produced by
scripts/blender_partial_uprightnet15.py and stores fixed-size source clouds.
Candidate positive/negative hypotheses are generated online by
scripts/train_uprightness_classifier.py, so this preprocessing step does not
duplicate every cloud for every candidate.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


CATEGORY_NAMES = [
    "bed",
    "bench",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "lamp",
    "monitor",
    "sofa",
    "stool",
    "table",
    "toilet",
    "vase",
]
CATEGORY_TO_ID = {name: idx for idx, name in enumerate(CATEGORY_NAMES)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pack partial UprightNet OFF point clouds into train/test NPZ files."
    )
    p.add_argument("--input-root", default="datasets/uprightnet15_partial_camera")
    p.add_argument("--glob", default="*/*/*.off")
    p.add_argument("--out-dir", default="datasets/uprightness_partial_npz")
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--limit", type=int, default=0, help="0 means all matched files")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--source-up-axis",
        choices=("y", "z"),
        default="z",
        help="Up axis of the source partial OFF files.",
    )
    return p.parse_args()


def parse_value(text: str):
    text = text.strip()
    if "," in text:
        return tuple(parse_value(part) for part in text.split(","))
    try:
        if re.match(r"^-?\d+$", text):
            return int(text)
        return float(text)
    except ValueError:
        return text


def read_point_off(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    metadata: dict[str, object] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        magic = handle.readline().strip()
        if magic != "OFF":
            raise ValueError(f"{path}: expected point-only OFF header, got {magic!r}")

        counts = None
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                item = line[1:].strip()
                if "=" in item:
                    key, value = item.split("=", 1)
                    metadata[key.strip()] = parse_value(value)
                continue
            counts = line.split()
            break

        if counts is None or len(counts) < 3:
            raise ValueError(f"{path}: missing OFF counts line")
        num_vertices, num_faces, _ = [int(x) for x in counts[:3]]
        if num_faces != 0:
            raise ValueError(f"{path}: expected point-only OFF, got {num_faces} faces")

        points = []
        for _ in range(num_vertices):
            parts = handle.readline().split()
            if len(parts) >= 3:
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))

    if not points:
        raise ValueError(f"{path}: no vertices found")
    return np.asarray(points, dtype=np.float32), metadata


def resample_points(
    points: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    if len(points) == count:
        return points.astype(np.float32, copy=True)
    replace = len(points) < count
    idx = rng.choice(len(points), size=count, replace=replace)
    return points[idx].astype(np.float32, copy=True)


def infer_category_and_split(path: Path, root: Path) -> tuple[str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) < 3:
        raise ValueError(f"{path}: expected <category>/<split>/<file>.off")
    category, split = parts[0], parts[1]
    if split not in {"train", "test"}:
        raise ValueError(f"{path}: split must be train or test, got {split!r}")
    return category, split


def write_npz(
    out_path: Path,
    rows: list[tuple[Path, str, str, np.ndarray, dict[str, object]]],
    root: Path,
    num_points: int,
    seed: int,
    source_up_axis: str,
) -> None:
    rng = np.random.default_rng(seed)
    points = np.empty((len(rows), num_points, 3), dtype=np.float32)
    category_id = np.empty((len(rows),), dtype=np.int64)
    bottom_ratio = np.full((len(rows),), np.nan, dtype=np.float32)
    view_index = np.full((len(rows),), -1, dtype=np.int64)
    rel_paths: list[str] = []
    categories: list[str] = []

    for i, (path, category, _split, cloud, meta) in enumerate(rows):
        points[i] = resample_points(cloud, num_points, rng)
        category_id[i] = CATEGORY_TO_ID.get(category, -1)
        rel_paths.append(str(path.relative_to(root)))
        categories.append(category)

        br = meta.get("bottom_band_retained_ratio", np.nan)
        try:
            bottom_ratio[i] = float(br)
        except (TypeError, ValueError):
            bottom_ratio[i] = np.nan
        try:
            view_index[i] = int(meta.get("view_index", -1))
        except (TypeError, ValueError):
            view_index[i] = -1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        points=points,
        category_id=category_id,
        category=np.asarray(categories),
        rel_path=np.asarray(rel_paths),
        bottom_band_retained_ratio=bottom_ratio,
        view_index=view_index,
        source_up_axis=np.asarray(source_up_axis),
        category_names=np.asarray(CATEGORY_NAMES),
    )
    print(f"[write] {out_path} clouds={len(rows)} points={num_points}")


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    paths = sorted(root.glob(args.glob))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No OFF files matched {root / args.glob}")

    rows_by_split: dict[str, list] = {"train": [], "test": []}
    for index, path in enumerate(paths):
        category, split = infer_category_and_split(path, root)
        points, meta = read_point_off(path)
        rows_by_split[split].append((path, category, split, points, meta))
        if (index + 1) % 500 == 0:
            print(f"[read] {index + 1}/{len(paths)}")

    out_dir = Path(args.out_dir)
    for split, rows in rows_by_split.items():
        if not rows:
            print(f"[skip] split={split} has no files")
            continue
        write_npz(
            out_dir / f"{split}.npz",
            rows,
            root,
            args.num_points,
            args.seed + (0 if split == "train" else 1),
            args.source_up_axis,
        )


if __name__ == "__main__":
    main()

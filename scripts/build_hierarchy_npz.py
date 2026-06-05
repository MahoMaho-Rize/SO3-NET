#!/usr/bin/env python3
"""Pack partial UprightNet15 OFF files with point-wise height hierarchy labels.

The labels are defined on visible points only.  For each partial point, its
canonical coordinate along the source up axis is normalized by the full object
bbox when available, then quantized into ordered bottom-to-top levels.
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
        description="Build train/test NPZ files for hierarchical partial uprightness."
    )
    p.add_argument("--input-root", default="datasets/uprightnet15_partial_camera")
    p.add_argument("--full-root", default="datasets/uprightnet15")
    p.add_argument("--glob", default="*/*/*.off")
    p.add_argument("--out-dir", default="datasets/upright_hierarchy_npz")
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--num-levels", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="0 means all matched files")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--source-up-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Up axis of the canonical OFF files.",
    )
    p.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed. Smaller but much slower on large datasets.",
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


def clean_data_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                yield line


def read_off_vertices(path: Path) -> np.ndarray:
    lines = clean_data_lines(path)
    try:
        first = next(lines)
    except StopIteration as exc:
        raise ValueError(f"{path}: empty OFF file") from exc

    if first == "OFF":
        counts = next(lines).split()
    elif first.startswith("OFF"):
        counts = first[3:].strip().split()
    else:
        raise ValueError(f"{path}: unsupported OFF header {first!r}")

    if len(counts) < 3:
        raise ValueError(f"{path}: malformed OFF counts line")
    num_vertices = int(counts[0])
    vertices = []
    for _ in range(num_vertices):
        parts = next(lines).split()
        if len(parts) < 3:
            raise ValueError(f"{path}: malformed vertex line")
        vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not vertices:
        raise ValueError(f"{path}: no vertices found")
    return np.asarray(vertices, dtype=np.float32)


def read_point_off(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    metadata: dict[str, object] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        magic = handle.readline().strip()
        if magic != "OFF":
            raise ValueError(f"{path}: expected OFF header, got {magic!r}")

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


def infer_category_and_split(path: Path, root: Path) -> tuple[str, str]:
    rel = path.relative_to(root)
    if len(rel.parts) < 3:
        raise ValueError(f"{path}: expected <category>/<split>/<file>.off")
    category, split = rel.parts[0], rel.parts[1]
    if split not in {"train", "test"}:
        raise ValueError(f"{path}: split must be train or test, got {split!r}")
    return category, split


def source_path_for(path: Path, input_root: Path, full_root: Path, meta: dict[str, object]) -> Path:
    source = meta.get("source")
    if isinstance(source, str):
        return full_root / source

    rel = path.relative_to(input_root)
    category, split, filename = rel.parts[0], rel.parts[1], rel.parts[-1]
    stem = Path(filename).stem
    stem = re.sub(r"_view\d+$", "", stem)
    return full_root / category / split / f"{stem}.off"


def resample_with_labels(
    points: np.ndarray,
    labels: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == count:
        return points.astype(np.float32, copy=True), labels.astype(np.int64, copy=True)
    replace = len(points) < count
    idx = rng.choice(len(points), size=count, replace=replace)
    return points[idx].astype(np.float32, copy=True), labels[idx].astype(np.int64, copy=True)


def hierarchy_labels(
    points: np.ndarray,
    bbox_min: float,
    bbox_max: float,
    axis_idx: int,
    num_levels: int,
) -> np.ndarray:
    height = max(float(bbox_max - bbox_min), 1e-8)
    t = (points[:, axis_idx] - float(bbox_min)) / height
    labels = np.floor(np.clip(t, 0.0, 1.0 - 1e-7) * num_levels).astype(np.int64)
    return np.clip(labels, 0, num_levels - 1)


def write_split(
    out_path: Path,
    rows: list[tuple[Path, str, np.ndarray, np.ndarray, dict[str, object], Path, tuple[float, float]]],
    root: Path,
    num_points: int,
    num_levels: int,
    source_up_axis: str,
    seed: int,
    compress: bool,
) -> None:
    rng = np.random.default_rng(seed)
    points = np.empty((len(rows), num_points, 3), dtype=np.float32)
    level_labels = np.empty((len(rows), num_points), dtype=np.int64)
    category_id = np.empty((len(rows),), dtype=np.int64)
    bottom_ratio = np.full((len(rows),), np.nan, dtype=np.float32)
    rel_paths: list[str] = []
    source_paths: list[str] = []

    for i, (path, category, cloud, labels, meta, source_path, bbox_range) in enumerate(rows):
        points[i], level_labels[i] = resample_with_labels(cloud, labels, num_points, rng)
        category_id[i] = CATEGORY_TO_ID.get(category, -1)
        rel_paths.append(str(path.relative_to(root)))
        source_paths.append(str(source_path))

        try:
            bottom_ratio[i] = float(meta.get("bottom_band_retained_ratio", np.nan))
        except (TypeError, ValueError):
            bottom_ratio[i] = np.nan

    hist = np.bincount(level_labels.reshape(-1), minlength=num_levels).astype(np.int64)
    up = np.zeros(3, dtype=np.float32)
    up["xyz".index(source_up_axis)] = 1.0
    gt_up = np.repeat(up[None, :], len(rows), axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if compress else np.savez
    print(
        f"[save] {out_path} mode={'compressed' if compress else 'uncompressed'} "
        f"clouds={len(rows)} points={num_points} levels={num_levels}",
        flush=True,
    )
    save_fn(
        out_path,
        points=points,
        level_labels=level_labels,
        gt_up=gt_up,
        category_id=category_id,
        category_names=np.asarray(CATEGORY_NAMES),
        rel_path=np.asarray(rel_paths),
        source_path=np.asarray(source_paths),
        source_up_axis=np.asarray(source_up_axis),
        num_levels=np.asarray(num_levels, dtype=np.int64),
        level_histogram=hist,
        bottom_band_retained_ratio=bottom_ratio,
    )
    print(f"[write] {out_path} level_histogram={hist.tolist()}")


def main() -> None:
    args = parse_args()
    if args.num_levels < 2:
        raise SystemExit("--num-levels must be >= 2")

    input_root = Path(args.input_root).resolve()
    full_root = Path(args.full_root).resolve()
    paths = sorted(input_root.glob(args.glob))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No OFF files matched {input_root / args.glob}")

    axis_idx = "xyz".index(args.source_up_axis)
    bbox_cache: dict[Path, tuple[float, float]] = {}
    rows_by_split: dict[str, list] = {"train": [], "test": []}
    fallback_count = 0

    for index, path in enumerate(paths):
        category, split = infer_category_and_split(path, input_root)
        cloud, meta = read_point_off(path)
        source_path = source_path_for(path, input_root, full_root, meta)

        if source_path.exists():
            if source_path not in bbox_cache:
                source_vertices = read_off_vertices(source_path)
                bbox_cache[source_path] = (
                    float(source_vertices[:, axis_idx].min()),
                    float(source_vertices[:, axis_idx].max()),
                )
            bbox_range = bbox_cache[source_path]
        else:
            fallback_count += 1
            bbox_range = (float(cloud[:, axis_idx].min()), float(cloud[:, axis_idx].max()))

        labels = hierarchy_labels(
            cloud, bbox_range[0], bbox_range[1], axis_idx, args.num_levels
        )
        rows_by_split[split].append((path, category, cloud, labels, meta, source_path, bbox_range))

        if (index + 1) % 500 == 0:
            print(f"[read] {index + 1}/{len(paths)}", flush=True)

    if fallback_count:
        print(
            f"[warn] full source OFF missing for {fallback_count} partial files; "
            "used partial bbox for those labels",
            flush=True,
        )

    out_dir = Path(args.out_dir)
    for split, rows in rows_by_split.items():
        if not rows:
            print(f"[skip] split={split} has no files")
            continue
        write_split(
            out_dir / f"{split}.npz",
            rows,
            input_root,
            args.num_points,
            args.num_levels,
            args.source_up_axis,
            args.seed + (0 if split == "train" else 1),
            args.compress,
        )


if __name__ == "__main__":
    main()

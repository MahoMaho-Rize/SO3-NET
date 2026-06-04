#!/usr/bin/env python3
"""
Generate camera-simulated partial point clouds from UprightNet15 OFF meshes.

Pipeline:
    OFF mesh
    -> perspective camera depth map by first-hit ray casting
    -> depth-map back-projection to visible 3D points
    -> optional bottom-band occlusion mask
    -> resample to a fixed-size point cloud OFF

Run with Blender, not plain Python:

    blender --background --python scripts/blender_partial_uprightnet15.py -- \
        --input-root datasets/uprightnet15 \
        --output-root datasets/uprightnet15_partial_camera \
        --limit 10

The OFF parser is local because UprightNet15 contains both standard headers
("OFF" followed by counts) and inline headers ("OFF12636 8652 0").
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


Point = tuple[float, float, float]
Face = list[int]


def parse_args(argv: list[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Render partial UprightNet15 point clouds from mesh depth maps."
    )
    parser.add_argument("--input-root", default="datasets/uprightnet15")
    parser.add_argument("--output-root", default="datasets/uprightnet15_partial_camera")
    parser.add_argument("--glob", default="*/*/*.off")
    parser.add_argument("--limit", type=int, default=0, help="0 means all files")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--views-per-model", type=int, default=1)

    parser.add_argument(
        "--up-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Raw ModelNet-style UprightNet15 OFF files are normally z-up.",
    )
    parser.add_argument("--output-count", type=int, default=2048)
    parser.add_argument(
        "--depth-width",
        type=int,
        default=128,
        help="Initial synthetic depth-map width.",
    )
    parser.add_argument(
        "--depth-height",
        type=int,
        default=128,
        help="Initial synthetic depth-map height.",
    )
    parser.add_argument(
        "--max-depth-size",
        type=int,
        default=512,
        help="Auto-upscale depth map until enough points are visible, up to this size.",
    )
    parser.add_argument(
        "--pixel-jitter",
        action="store_true",
        help="Jitter ray location within each pixel instead of using pixel centers.",
    )

    parser.add_argument("--fov-deg", type=float, default=55.0)
    parser.add_argument(
        "--camera-distance-mult",
        type=float,
        default=0.0,
        help="If >0, use this bbox-diagonal multiplier; otherwise auto-frame.",
    )
    parser.add_argument(
        "--frame-fill",
        type=float,
        default=0.72,
        help="Approximate fraction of camera FOV occupied by the object in auto-frame mode.",
    )
    parser.add_argument(
        "--elevation-min-deg",
        type=float,
        default=8.0,
        help="Low elevation makes the visible cloud closer to a real side scan.",
    )
    parser.add_argument("--elevation-max-deg", type=float, default=24.0)
    parser.add_argument(
        "--target-up-bias",
        type=float,
        default=0.08,
        help="Aim above bbox center by this fraction of object height.",
    )

    parser.add_argument(
        "--bottom-band-ratio",
        type=float,
        default=0.18,
        help="Object-height band near the support plane where occlusion is strongest.",
    )
    parser.add_argument(
        "--bottom-drop-prob",
        type=float,
        default=0.65,
        help="Drop probability at the exact bottom, linearly decays to zero.",
    )
    parser.add_argument(
        "--keep-short",
        action="store_true",
        help="Write fewer than --output-count points if visibility is too sparse.",
    )
    return parser.parse_args(argv)


def clean_data_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                yield line


def read_off(path: Path) -> tuple[list[Point], list[Face]]:
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
        raise ValueError(f"{path}: malformed OFF counts")
    num_vertices, num_faces, _num_edges = [int(x) for x in counts[:3]]

    vertices: list[Point] = []
    for _ in range(num_vertices):
        parts = next(lines).split()
        if len(parts) < 3:
            raise ValueError(f"{path}: malformed vertex line")
        vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))

    faces: list[Face] = []
    for _ in range(num_faces):
        parts = next(lines).split()
        if not parts:
            continue
        n = int(parts[0])
        if n < 3 or len(parts) < n + 1:
            continue
        face = [int(x) for x in parts[1 : n + 1]]
        if all(0 <= idx < num_vertices for idx in face):
            faces.append(face)
    return vertices, faces


def write_points_off(path: Path, points: list[Point], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("OFF\n")
        for key in sorted(metadata):
            value = metadata[key]
            if isinstance(value, float):
                value_text = f"{value:.8g}"
            elif isinstance(value, (tuple, list)):
                value_text = ",".join(
                    f"{x:.8g}" if isinstance(x, float) else str(x) for x in value
                )
            else:
                value_text = str(value)
            handle.write(f"# {key}={value_text}\n")
        handle.write(f"{len(points)} 0 0\n")
        for x, y, z in points:
            handle.write(f"{x:.8g} {y:.8g} {z:.8g}\n")


def axis_vector(axis: str) -> Vector:
    return {
        "x": Vector((1.0, 0.0, 0.0)),
        "y": Vector((0.0, 1.0, 0.0)),
        "z": Vector((0.0, 0.0, 1.0)),
    }[axis]


def horizontal_basis(axis: str) -> tuple[Vector, Vector]:
    if axis == "x":
        return Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0))
    if axis == "y":
        return Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))
    return Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0))


def bbox(vertices: list[Point]):
    mins = [min(v[i] for v in vertices) for i in range(3)]
    maxs = [max(v[i] for v in vertices) for i in range(3)]
    center = Vector(tuple((mins[i] + maxs[i]) * 0.5 for i in range(3)))
    spans = [maxs[i] - mins[i] for i in range(3)]
    diag = math.sqrt(sum(s * s for s in spans))
    return mins, maxs, center, spans, max(diag, 1e-6)


def build_mesh_object(name: str, vertices: list[Point], faces: list[Face]):
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.update()
    return obj


def remove_object(obj) -> None:
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)


def get_camera():
    cam_data = bpy.data.cameras.new("depth_camera")
    cam_obj = bpy.data.objects.new("depth_camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def set_camera_pose(camera, args: argparse.Namespace, stats, rng: random.Random) -> dict[str, object]:
    _mins, _maxs, center, spans, diag = stats
    up = axis_vector(args.up_axis)
    h0, h1 = horizontal_basis(args.up_axis)
    azimuth = rng.random() * 2.0 * math.pi
    elevation = math.radians(rng.uniform(args.elevation_min_deg, args.elevation_max_deg))
    horizontal = math.cos(azimuth) * h0 + math.sin(azimuth) * h1

    up_span = max(spans["xyz".index(args.up_axis)], 1e-6)
    target = center + args.target_up_bias * up_span * up

    if args.camera_distance_mult > 0.0:
        distance = args.camera_distance_mult * diag
    else:
        bbox_radius = 0.5 * diag
        fill = min(max(args.frame_fill, 0.1), 0.95)
        half_angle = max(math.radians(args.fov_deg) * fill * 0.5, 1e-3)
        distance = bbox_radius / math.sin(half_angle)

    view_direction = math.cos(elevation) * horizontal + math.sin(elevation) * up
    camera.location = target + distance * view_direction.normalized()
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "PERSP"
    camera.data.angle = math.radians(args.fov_deg)
    camera.data.clip_start = max(diag * 1e-4, 1e-5)
    camera.data.clip_end = diag * 20.0
    bpy.context.view_layer.update()
    return {
        "camera_azimuth_deg": math.degrees(azimuth),
        "camera_elevation_deg": math.degrees(elevation),
        "camera_location": tuple(float(x) for x in camera.location),
        "camera_target": tuple(float(x) for x in target),
        "camera_distance": float((camera.location - target).length),
        "camera_fov_deg": float(args.fov_deg),
    }


def bottom_band_drop_probability(point: Point, stats, args: argparse.Namespace) -> float:
    if args.bottom_band_ratio <= 0.0 or args.bottom_drop_prob <= 0.0:
        return 0.0
    mins, maxs, _center, _spans, _diag = stats
    axis_idx = "xyz".index(args.up_axis)
    height = max(maxs[axis_idx] - mins[axis_idx], 1e-6)
    band = height * args.bottom_band_ratio
    rel = (point[axis_idx] - mins[axis_idx]) / band
    if rel >= 1.0:
        return 0.0
    return args.bottom_drop_prob * max(0.0, 1.0 - rel)


def camera_frame_bounds(camera, scene):
    frame = camera.data.view_frame(scene=scene)
    min_x = min(v.x for v in frame)
    max_x = max(v.x for v in frame)
    min_y = min(v.y for v in frame)
    max_y = max(v.y for v in frame)
    z = frame[0].z
    return min_x, max_x, min_y, max_y, z


def render_depth_backproject(
    obj,
    camera,
    width: int,
    height: int,
    stats,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[Point], dict[str, int]]:
    """First-hit depth rendering followed by back-projection to world points."""
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(obj, depsgraph)
    origin = camera.matrix_world.translation
    camera_rot = camera.matrix_world.to_quaternion()
    _mins, _maxs, _center, _spans, diag = stats
    max_distance = diag * 20.0
    min_x, max_x, min_y, max_y, z = camera_frame_bounds(camera, scene)

    points: list[Point] = []
    hit_pixels = 0
    bottom_band_hits = 0
    bottom_band_kept = 0
    for row in range(height):
        for col in range(width):
            if args.pixel_jitter:
                u = (col + rng.random()) / width
                v = (row + rng.random()) / height
            else:
                u = (col + 0.5) / width
                v = (row + 0.5) / height

            sensor_x = min_x + u * (max_x - min_x)
            sensor_y = min_y + v * (max_y - min_y)
            ray_dir = camera_rot @ Vector((sensor_x, sensor_y, z)).normalized()
            hit_loc, _hit_normal, _hit_index, _hit_distance = bvh.ray_cast(
                origin, ray_dir, max_distance
            )
            if hit_loc is None:
                continue

            # This is the back-projected depth sample: camera origin plus
            # first-hit range along the pixel ray, expressed in world coords.
            point = (hit_loc.x, hit_loc.y, hit_loc.z)
            hit_pixels += 1
            bottom_drop_prob = bottom_band_drop_probability(point, stats, args)
            if bottom_drop_prob > 0.0:
                bottom_band_hits += 1
            if rng.random() >= bottom_drop_prob:
                if bottom_drop_prob > 0.0:
                    bottom_band_kept += 1
                points.append(point)
    return points, {
        "hit_pixels": hit_pixels,
        "bottom_band_hits": bottom_band_hits,
        "bottom_band_kept": bottom_band_kept,
    }


def fixed_count(points: list[Point], count: int, rng: random.Random) -> list[Point]:
    if count <= 0 or len(points) == count:
        return list(points)
    if len(points) > count:
        return rng.sample(points, count)
    if not points:
        return []

    out = list(points)
    while len(out) < count:
        out.append(points[rng.randrange(len(points))])
    rng.shuffle(out)
    return out


def output_path_for(src: Path, input_root: Path, output_root: Path, view_idx: int, views: int):
    rel = src.relative_to(input_root)
    if views <= 1:
        return output_root / rel
    return output_root / rel.parent / f"{rel.stem}_view{view_idx:02d}{rel.suffix}"


def process_file(src: Path, input_root: Path, output_root: Path, args, rng: random.Random):
    vertices, faces = read_off(src)
    if not vertices or not faces:
        print(f"[skip] {src}: empty mesh")
        return 0

    stats = bbox(vertices)
    obj = build_mesh_object(src.stem, vertices, faces)
    camera = bpy.context.scene.camera or get_camera()

    written = 0
    try:
        for view_idx in range(args.views_per_model):
            camera_meta = set_camera_pose(camera, args, stats, rng)

            width = max(1, args.depth_width)
            height = max(1, args.depth_height)
            visible: list[Point] = []
            render_stats: dict[str, int] = {
                "hit_pixels": 0,
                "bottom_band_hits": 0,
                "bottom_band_kept": 0,
            }
            while True:
                visible, render_stats = render_depth_backproject(
                    obj, camera, width, height, stats, args, rng
                )
                if (
                    args.keep_short
                    or args.output_count <= 0
                    or len(visible) >= args.output_count
                    or max(width, height) >= args.max_depth_size
                ):
                    break
                width *= 2
                height *= 2

            if not visible:
                print(f"[skip] {src}: view {view_idx} produced no depth hits")
                continue

            final = visible if args.keep_short else fixed_count(visible, args.output_count, rng)
            if len(visible) < args.output_count and not args.keep_short:
                print(
                    f"[warn] {src}: only {len(visible)} unique visible samples after "
                    f"bottom mask; repeated points to reach {args.output_count}"
                )

            dst = output_path_for(src, input_root, output_root, view_idx, args.views_per_model)
            bottom_hits = render_stats["bottom_band_hits"]
            bottom_kept = render_stats["bottom_band_kept"]
            metadata = {
                "source": src.relative_to(input_root),
                "view_index": view_idx,
                "up_axis": args.up_axis,
                "depth_width": width,
                "depth_height": height,
                "output_count": len(final),
                "unique_visible_after_bottom": len(visible),
                "hit_pixels": render_stats["hit_pixels"],
                "bottom_band_ratio": args.bottom_band_ratio,
                "bottom_drop_prob": args.bottom_drop_prob,
                "bottom_band_hits": bottom_hits,
                "bottom_band_kept": bottom_kept,
                "bottom_band_retained_ratio": (
                    bottom_kept / bottom_hits if bottom_hits > 0 else 1.0
                ),
                "pixel_jitter": int(args.pixel_jitter),
                **camera_meta,
            }
            write_points_off(dst, final, metadata)
            print(
                f"[ok] {src} -> {dst} depth={width}x{height} "
                f"hit_pixels={render_stats['hit_pixels']} visible_after_bottom={len(visible)} "
                f"bottom_kept={bottom_kept}/{bottom_hits} "
                f"final={len(final)}"
            )
            written += 1
    finally:
        remove_object(obj)
    return written


def main() -> int:
    args = parse_args(sys.argv)
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    files = sorted(input_root.glob(args.glob))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"No OFF files matched {input_root / args.glob}")
        return 1

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    get_camera()

    rng = random.Random(args.seed)
    total = 0
    for src in files:
        total += process_file(src, input_root, output_root, args, rng)

    print(f"Done. wrote {total} partial point-cloud OFF files to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train a candidate-conditioned uprightness classifier.

This is intentionally not a direction-regression model.  The model receives a
point cloud normalized by a candidate upright hypothesis and classifies whether
that hypothesis is a valid upright explanation.

Training item:
    source partial cloud P
    random rotation R
    candidate h in the rotated coordinate frame
    aligned cloud P_h = Align(h -> +Y) @ (R @ P)
    label z = 1 if h is the GT upright hypothesis else 0

Main loss:
    BCEWithLogits(C(P_h), z)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CANONICAL_UP = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train candidate-conditioned uprightness classifier")
    p.add_argument("--train-npz", default="datasets/uprightness_partial_npz/train.npz")
    p.add_argument("--test-npz", default="datasets/uprightness_partial_npz/test.npz")
    p.add_argument("--out-dir", default="models/uprightness_classifier")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--samples-per-cloud", type=int, default=10)
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument(
        "--data-parallel",
        action="store_true",
        help="Use torch.nn.DataParallel across visible CUDA devices.",
    )
    p.add_argument("--pos-jitter-deg", type=float, default=5.0)
    p.add_argument("--neg-min-angle-deg", type=float, default=30.0)
    p.add_argument("--neg-max-angle-deg", type=float, default=150.0)
    p.add_argument("--pos-weight", type=float, default=4.0)
    p.add_argument("--vis-weight", type=float, default=0.2)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--eval-candidate-sets", type=int, default=512)
    p.add_argument("--log-csv", default="")
    return p.parse_args()


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    return (v / norm).astype(np.float32)


def random_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3).astype(np.float32)
    return normalize(v)


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    qx = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    qy = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def rotate_about_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis)
    v = v.astype(np.float32)
    return normalize(
        v * math.cos(angle_rad)
        + np.cross(axis, v) * math.sin(angle_rad)
        + axis * float(np.dot(axis, v)) * (1.0 - math.cos(angle_rad))
    )


def random_perpendicular_axis(
    direction: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    axis = random_unit(rng)
    axis = axis - direction * float(np.dot(axis, direction))
    if np.linalg.norm(axis) < 1e-6:
        axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        axis = axis - direction * float(np.dot(axis, direction))
    return normalize(axis)


def rotation_align_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return R such that R @ a ~= b."""
    a = normalize(a)
    b = normalize(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-6:
        return np.eye(3, dtype=np.float32)
    if c < -1.0 + 1e-6:
        axis = random_perpendicular_axis(a, np.random.default_rng(0))
        vx = np.asarray(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            dtype=np.float32,
        )
        return (np.eye(3, dtype=np.float32) + 2.0 * (vx @ vx)).astype(np.float32)
    vx = np.asarray(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
        dtype=np.float32,
    )
    return (np.eye(3, dtype=np.float32) + vx + vx @ vx * (1.0 / (1.0 + c))).astype(
        np.float32
    )


def normalize_cloud(points: np.ndarray) -> np.ndarray:
    out = points.astype(np.float32, copy=True)
    out -= out.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(out, axis=1).max()
    if scale > 1e-8:
        out /= scale
    return out.astype(np.float32)


def source_up_vector(axis: str) -> np.ndarray:
    if axis == "y":
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    if axis == "z":
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    raise ValueError(f"unsupported source up axis: {axis}")


def visibility_label(bottom_ratio: float) -> int:
    if not np.isfinite(bottom_ratio):
        return -1
    if bottom_ratio < 0.10:
        return 0
    if bottom_ratio < 0.50:
        return 1
    return 2


class UprightnessCandidateDataset(Dataset):
    candidate_types = ("pos", "flip", "tilt", "random", "pca")

    def __init__(
        self,
        npz_path: str | Path,
        samples_per_cloud: int,
        num_points: int,
        seed: int,
        pos_jitter_deg: float,
        neg_min_angle_deg: float,
        neg_max_angle_deg: float,
    ):
        data = np.load(npz_path, allow_pickle=True)
        self.points = data["points"].astype(np.float32)
        self.bottom_ratio = data.get(
            "bottom_band_retained_ratio",
            np.full((len(self.points),), np.nan, dtype=np.float32),
        ).astype(np.float32)
        source_axis = str(data.get("source_up_axis", np.asarray("z")).item())
        self.source_up = source_up_vector(source_axis)
        self.samples_per_cloud = samples_per_cloud
        self.num_points = min(num_points, self.points.shape[1])
        self.seed = seed
        self.pos_jitter_deg = pos_jitter_deg
        self.neg_min_angle_deg = neg_min_angle_deg
        self.neg_max_angle_deg = neg_max_angle_deg

    def __len__(self) -> int:
        return len(self.points) * self.samples_per_cloud

    def _candidate(
        self,
        candidate_type: str,
        points_rot: np.ndarray,
        gt_up: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float, int]:
        type_id = self.candidate_types.index(candidate_type)
        if candidate_type == "pos":
            angle = math.radians(rng.uniform(0.0, self.pos_jitter_deg))
            axis = random_perpendicular_axis(gt_up, rng)
            return rotate_about_axis(gt_up, axis, angle), 1.0, type_id

        if candidate_type == "flip":
            angle = math.radians(rng.uniform(0.0, self.pos_jitter_deg))
            axis = random_perpendicular_axis(gt_up, rng)
            return rotate_about_axis(-gt_up, axis, angle), 0.0, type_id

        if candidate_type == "tilt":
            angle = math.radians(
                rng.uniform(self.neg_min_angle_deg, self.neg_max_angle_deg)
            )
            axis = random_perpendicular_axis(gt_up, rng)
            return rotate_about_axis(gt_up, axis, angle), 0.0, type_id

        if candidate_type == "pca":
            centered = points_rot - points_rot.mean(axis=0, keepdims=True)
            cov = centered.T @ centered / max(len(centered), 1)
            _vals, vecs = np.linalg.eigh(cov)
            dirs = [normalize(vecs[:, i]) for i in range(3)]
            dirs += [-d for d in dirs]
            dirs.sort(key=lambda d: float(np.dot(d, gt_up)))
            cand = dirs[0]
            if math.degrees(math.acos(float(np.clip(np.dot(cand, gt_up), -1, 1)))) < self.neg_min_angle_deg:
                cand = -gt_up
            return normalize(cand), 0.0, type_id

        cand = random_unit(rng)
        dot = float(np.dot(cand, gt_up))
        angle = math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0))))
        if angle < self.neg_min_angle_deg:
            cand = -cand
        return normalize(cand), 0.0, type_id

    def __getitem__(self, index: int):
        cloud_idx = index // self.samples_per_cloud
        type_idx = index % len(self.candidate_types)
        candidate_type = self.candidate_types[type_idx]
        rng = np.random.default_rng(self.seed + index * 17)

        points = self.points[cloud_idx, : self.num_points]
        if len(points) > self.num_points:
            choice = rng.choice(len(points), size=self.num_points, replace=False)
            points = points[choice]

        R = random_rotation_matrix(rng)
        points_rot = (R @ points.T).T.astype(np.float32)
        gt_up = normalize(R @ self.source_up)
        candidate, label, candidate_type_id = self._candidate(
            candidate_type, points_rot, gt_up, rng
        )

        R_align = rotation_align_a_to_b(candidate, CANONICAL_UP)
        aligned = (R_align @ points_rot.T).T.astype(np.float32)
        aligned = normalize_cloud(aligned)
        vis = visibility_label(float(self.bottom_ratio[cloud_idx]))

        return (
            torch.from_numpy(aligned),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(vis, dtype=torch.long),
            torch.tensor(candidate_type_id, dtype=torch.long),
        )


class PointNetUprightnessClassifier(nn.Module):
    def __init__(self, hidden: int = 512, use_visibility_head: bool = True):
        super().__init__()
        self.use_visibility_head = use_visibility_head
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )
        self.vis_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, points_bnc: torch.Tensor) -> dict[str, torch.Tensor]:
        x = points_bnc.transpose(1, 2).contiguous()
        feat = self.encoder(x).max(dim=-1)[0]
        out = {"upright_logit": self.head(feat).squeeze(-1)}
        if self.use_visibility_head:
            out["visibility_logits"] = self.vis_head(feat)
        return out


def compute_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    visibility: torch.Tensor,
    pos_weight: float,
    vis_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pos_w = torch.tensor(pos_weight, dtype=torch.float32, device=labels.device)
    bce = F.binary_cross_entropy_with_logits(
        outputs["upright_logit"], labels, pos_weight=pos_w
    )
    total = bce
    stats = {"bce": float(bce.detach().cpu())}
    if vis_weight > 0 and "visibility_logits" in outputs:
        vis_loss = F.cross_entropy(
            outputs["visibility_logits"], visibility, ignore_index=-1
        )
        if torch.isfinite(vis_loss):
            total = total + vis_weight * vis_loss
            stats["vis"] = float(vis_loss.detach().cpu())
    stats["total"] = float(total.detach().cpu())
    return total, stats


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: float,
    vis_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    pos_total = pos_correct = neg_total = neg_correct = 0
    type_total: dict[int, int] = {}
    type_correct: dict[int, int] = {}

    for points, labels, visibility, type_id in loader:
        points = points.to(device)
        labels = labels.to(device)
        visibility = visibility.to(device)
        outputs = model(points)
        loss, _stats = compute_loss(outputs, labels, visibility, pos_weight, vis_weight)
        logits = outputs["upright_logit"]
        pred = (torch.sigmoid(logits) >= 0.5).float()
        batch_total = labels.numel()
        total_loss += float(loss.detach().cpu()) * batch_total
        total += batch_total
        correct += int((pred == labels).sum().item())

        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_total += int(pos_mask.sum().item())
        neg_total += int(neg_mask.sum().item())
        pos_correct += int((pred[pos_mask] == labels[pos_mask]).sum().item())
        neg_correct += int((pred[neg_mask] == labels[neg_mask]).sum().item())

        for t in type_id.unique().tolist():
            mask = type_id == t
            type_total[t] = type_total.get(t, 0) + int(mask.sum().item())
            type_correct[t] = type_correct.get(t, 0) + int(
                (pred.cpu()[mask] == labels.cpu()[mask]).sum().item()
            )

    metrics = {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "pos_acc": pos_correct / max(pos_total, 1),
        "neg_acc": neg_correct / max(neg_total, 1),
    }
    for t, name in enumerate(UprightnessCandidateDataset.candidate_types):
        if t in type_total:
            metrics[f"acc_{name}"] = type_correct[t] / max(type_total[t], 1)
    return metrics


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

    train_ds = UprightnessCandidateDataset(
        args.train_npz,
        samples_per_cloud=args.samples_per_cloud,
        num_points=args.num_points,
        seed=args.seed,
        pos_jitter_deg=args.pos_jitter_deg,
        neg_min_angle_deg=args.neg_min_angle_deg,
        neg_max_angle_deg=args.neg_max_angle_deg,
    )
    test_ds = UprightnessCandidateDataset(
        args.test_npz,
        samples_per_cloud=max(5, min(args.samples_per_cloud, 10)),
        num_points=args.num_points,
        seed=args.seed + 100000,
        pos_jitter_deg=args.pos_jitter_deg,
        neg_min_angle_deg=args.neg_min_angle_deg,
        neg_max_angle_deg=args.neg_max_angle_deg,
    )
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

    model = PointNetUprightnessClassifier(hidden=args.hidden).to(device)
    if args.data_parallel:
        if device.type != "cuda":
            raise ValueError("--data-parallel requires --device cuda or CUDA auto-detection")
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            raise ValueError(f"--data-parallel requested but only {gpu_count} CUDA device is visible")
        model = nn.DataParallel(model)
        print(f"data_parallel=True visible_cuda_devices={gpu_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv) if args.log_csv else out_dir / "train_log.csv"
    best_acc = -1.0

    print(f"device={device}")
    print(f"train candidates={len(train_ds)} test candidates={len(test_ds)}")
    print(f"loss=BCEWithLogits pos_weight={args.pos_weight} vis_weight={args.vis_weight}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for points, labels, visibility, _type_id in train_loader:
            points = points.to(device)
            labels = labels.to(device)
            visibility = visibility.to(device)
            outputs = model(points)
            loss, _stats = compute_loss(
                outputs, labels, visibility, args.pos_weight, args.vis_weight
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu()) * labels.numel()
            count += labels.numel()

        scheduler.step()
        train_loss = running / max(count, 1)
        metrics = evaluate_classifier(
            model, test_loader, device, args.pos_weight, args.vis_weight
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        write_log(log_csv, row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} acc={metrics['acc']*100:.2f}% "
            f"pos={metrics['pos_acc']*100:.2f}% neg={metrics['neg_acc']*100:.2f}%"
        )

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            ckpt = {
                "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "metrics": metrics,
                "candidate_types": UprightnessCandidateDataset.candidate_types,
            }
            torch.save(ckpt, out_dir / "best.pth")
            print(f"[save] {out_dir / 'best.pth'} acc={best_acc*100:.2f}%")

    torch.save(
        {
            "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "args": vars(args),
            "epoch": args.epochs,
            "candidate_types": UprightnessCandidateDataset.candidate_types,
        },
        out_dir / "final.pth",
    )
    print(f"[done] best_acc={best_acc*100:.2f}% out={out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train point-wise hierarchy segmentation and recover upright direction."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train hierarchical partial UprightNet")
    p.add_argument("--train-npz", default="datasets/upright_hierarchy_npz/train.npz")
    p.add_argument("--test-npz", default="datasets/upright_hierarchy_npz/test.npz")
    p.add_argument("--out-dir", default="models/hierarchical_uprightnet")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument(
        "--arch",
        default="dgcnn",
        choices=("dgcnn", "pointnet"),
        help="Point-wise hierarchy segmentation backbone.",
    )
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--class-balance", action="store_true")
    p.add_argument("--data-parallel", action="store_true")
    p.add_argument("--log-csv", default="")
    return p.parse_args()


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def normalize_cloud(points: np.ndarray) -> np.ndarray:
    out = points.astype(np.float32, copy=True)
    out -= out.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(out, axis=1).max()
    if scale > 1e-8:
        out /= scale
    return out


class HierarchyDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        num_points: int,
        seed: int,
        augment: bool,
    ) -> None:
        data = np.load(npz_path, allow_pickle=False)
        self.points = data["points"].astype(np.float32)
        self.labels = data["level_labels"].astype(np.int64)
        self.gt_up = data["gt_up"].astype(np.float32)
        self.category_id = data["category_id"].astype(np.int64)
        self.num_levels = int(data["num_levels"])
        self.num_points = num_points
        self.seed = seed
        self.augment = augment

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx * 1009)
        points = self.points[idx]
        labels = self.labels[idx]

        replace = len(points) < self.num_points
        choice = rng.choice(len(points), size=self.num_points, replace=replace)
        points = points[choice]
        labels = labels[choice]

        if self.augment:
            rot = random_rotation_matrix(rng)
        else:
            # Deterministic validation rotations prevent a fixed-axis shortcut.
            rot = random_rotation_matrix(np.random.default_rng(self.seed + idx * 9176))
        points = (rot @ points.T).T.astype(np.float32)
        up = (rot @ self.gt_up[idx]).astype(np.float32)
        up /= max(np.linalg.norm(up), 1e-12)
        points = normalize_cloud(points)

        return (
            torch.from_numpy(points),
            torch.from_numpy(labels.astype(np.int64)),
            torch.from_numpy(up),
            torch.tensor(int(self.category_id[idx]), dtype=torch.long),
        )


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(
    x: torch.Tensor, k: int = 20, idx: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)

    idx_out = idx
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    num_dims = x.size(1)
    x_t = x.transpose(2, 1).contiguous()
    feature = x_t.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    center = x_t.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - center, center), dim=3)
    return feature.permute(0, 3, 1, 2).contiguous(), idx_out


class SelfAttentionLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_q = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv_k = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv_v = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.conv_q(x).permute(0, 2, 1)
        x_k = self.conv_k(x)
        x_v = self.conv_v(x)
        attention = torch.bmm(x_q, x_k) / math.sqrt(x.size(1))
        attention = F.softmax(attention, dim=1)
        x_r = torch.bmm(x_v, attention)
        return x + self.conv(x_r)


class PointNetHierarchyNet(nn.Module):
    def __init__(self, num_levels: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.seg = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden // 2, 1, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden // 2, num_levels, 1),
        )

    def forward(self, points_bnc: torch.Tensor) -> torch.Tensor:
        x = points_bnc.transpose(1, 2).contiguous()
        local = self.local(x)
        global_feat = local.max(dim=-1, keepdim=True)[0].expand_as(local)
        return self.seg(torch.cat([local, global_feat], dim=1))


class DGCNNHierarchyNet(nn.Module):
    def __init__(self, num_levels: int, dropout: float = 0.2, k: int = 20):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.sa1 = SelfAttentionLayer(128)
        self.sa2 = SelfAttentionLayer(128)
        self.sa3 = SelfAttentionLayer(128)
        self.sa4 = SelfAttentionLayer(128)
        self.conv4 = nn.Sequential(
            nn.Conv1d(128 * 4, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(128 + 512 + 1024 + 1024, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(512, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
        )
        self.conv7 = nn.Conv1d(64, num_levels, 1)

    def forward(self, points_bnc: torch.Tensor) -> torch.Tensor:
        x = points_bnc.transpose(1, 2).contiguous()
        batch_size = x.size(0)
        num_points = x.size(2)

        x, _ = get_graph_feature(x, k=self.k)
        x = self.conv1(x).max(dim=-1, keepdim=False)[0]

        x, _ = get_graph_feature(x, k=self.k)
        x = self.conv2(x).max(dim=-1, keepdim=False)[0]

        x, _ = get_graph_feature(x, k=self.k)
        x = self.conv3(x)
        x_a = x.max(dim=-1, keepdim=False)[0]

        x1 = self.sa1(x_a)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x_b = torch.cat((x1, x2, x3, x4), dim=1)

        x_c = self.conv4(x_b)
        global_feat = F.adaptive_max_pool1d(x_c, 1).view(batch_size, -1)
        x_global = global_feat.view(batch_size, -1, 1).repeat(1, 1, num_points)

        x = torch.cat((x_a, x_b, x_c, x_global), dim=1)
        x = self.conv5(x)
        x = self.conv6(x)
        return self.conv7(x)


def direction_from_level_logits(points: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    num_levels = logits.shape[1]
    levels = torch.linspace(0.0, 1.0, num_levels, device=logits.device).view(1, num_levels, 1)
    score = (probs * levels).sum(dim=1)

    point_center = points.mean(dim=1, keepdim=True)
    score_center = score.mean(dim=1, keepdim=True)
    centered_points = points - point_center
    centered_score = score - score_center

    # Fit score ~= a + dot(direction, point).  Direct covariance is biased by
    # anisotropic object geometry; least squares whitens the point covariance.
    cov = torch.bmm(centered_points.transpose(1, 2), centered_points)
    rhs = torch.bmm(
        centered_points.transpose(1, 2), centered_score.unsqueeze(-1)
    ).squeeze(-1)
    eye = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0)
    ridge = 1e-4 * points.shape[1]
    try:
        direction = torch.linalg.solve(cov + ridge * eye, rhs.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        direction = rhs

    low_w = probs[:, 0, :]
    high_w = probs[:, -1, :]
    low = (points * low_w.unsqueeze(-1)).sum(dim=1) / low_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    high = (points * high_w.unsqueeze(-1)).sum(dim=1) / high_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    fallback = high - low

    use_fallback = direction.norm(dim=1, keepdim=True) < 1e-6
    direction = torch.where(use_fallback, fallback, direction)
    return F.normalize(direction, dim=1, eps=1e-6)


def angular_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cos = (pred * target).sum(dim=1).clamp(-1.0, 1.0)
    return torch.acos(cos) * (180.0 / math.pi)


def miou_from_confusion(conf: np.ndarray) -> float:
    ious = []
    for i in range(conf.shape[0]):
        denom = conf[i, :].sum() + conf[:, i].sum() - conf[i, i]
        if denom > 0:
            ious.append(conf[i, i] / denom)
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_levels: int, loss_weight):
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
        logits = model(points)
        loss = F.cross_entropy(logits, labels, weight=loss_weight)
        pred = logits.argmax(dim=1)

        total_loss += float(loss.detach().cpu()) * labels.numel()
        total_points += labels.numel()
        correct += int((pred == labels).sum().item())

        pred_up = direction_from_level_logits(points, logits)
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


def class_weights_from_hist(hist: np.ndarray, device: torch.device) -> torch.Tensor:
    hist = hist.astype(np.float64)
    freq = hist / max(hist.sum(), 1.0)
    weights = 1.0 / np.log(1.2 + np.maximum(freq, 1e-8))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


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
        model = DGCNNHierarchyNet(num_levels, args.dropout).to(device)
    else:
        model = PointNetHierarchyNet(num_levels, args.hidden, args.dropout).to(device)
    if args.data_parallel:
        if device.type != "cuda":
            raise ValueError("--data-parallel requires CUDA")
        if torch.cuda.device_count() < 2:
            raise ValueError("--data-parallel requested but fewer than 2 visible GPUs")
        model = nn.DataParallel(model)
        print(f"data_parallel=True visible_cuda_devices={torch.cuda.device_count()}")

    train_hist = np.load(args.train_npz, allow_pickle=False)["level_histogram"]
    loss_weight = class_weights_from_hist(train_hist, device) if args.class_balance else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv) if args.log_csv else out_dir / "train_log.csv"
    best_acc10 = -1.0

    print(f"device={device}")
    print(f"arch={args.arch}")
    print(f"train_clouds={len(train_ds)} test_clouds={len(test_ds)} levels={num_levels}")
    if loss_weight is not None:
        print(f"class_weights={[round(float(x), 4) for x in loss_weight.detach().cpu()]}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for points, labels, _gt_up, _cat in train_loader:
            points = points.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(points)
            loss = F.cross_entropy(logits, labels, weight=loss_weight)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * labels.numel()
            count += labels.numel()

        scheduler.step()
        metrics = evaluate(model, test_loader, device, num_levels, loss_weight)
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
        },
        out_dir / "final.pth",
    )
    print(f"[done] wrote {out_dir / 'final.pth'}")


if __name__ == "__main__":
    main()

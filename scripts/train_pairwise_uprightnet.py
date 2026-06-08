#!/usr/bin/env python3
"""Explore pairwise upright-order classification on partial point clouds."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from train_hierarchical_uprightnet import (
    SelfAttentionLayer,
    angular_error_deg,
    get_graph_feature,
    normalize_cloud,
    random_rotation_matrix,
)


LOWER = 0
SAME = 1
HIGHER = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train pairwise upright-order classifier")
    p.add_argument("--train-npz", default="datasets/upright_pairwise_npz/train.npz")
    p.add_argument("--test-npz", default="datasets/upright_pairwise_npz/test.npz")
    p.add_argument("--out-dir", default="models/pairwise_uprightnet_dgcnn")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-points", type=int, default=2048)
    p.add_argument("--pairs-per-cloud", type=int, default=4096)
    p.add_argument("--eval-pairs-per-cloud", type=int, default=8192)
    p.add_argument("--same-threshold-ratio", type=float, default=0.03)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--arch", default="dgcnn", choices=("dgcnn", "pointnet"))
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--same-weight", type=float, default=1.0)
    p.add_argument("--confidence-power", type=float, default=1.0)
    p.add_argument("--data-parallel", action="store_true")
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--test-limit", type=int, default=0)
    p.add_argument("--log-csv", default="")
    return p.parse_args()


class PairwiseUprightDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        num_points: int,
        seed: int,
        augment: bool,
    ) -> None:
        data = np.load(npz_path, allow_pickle=False)
        self.points = data["points"].astype(np.float32)
        self.gt_up = data["gt_up"].astype(np.float32)
        self.category_id = data["category_id"].astype(np.int64)
        self.num_points = num_points
        self.seed = seed
        self.augment = augment
        self.epoch = 0

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, idx: int):
        epoch_offset = self.epoch * 1000003 if self.augment else 0
        rng = np.random.default_rng(self.seed + idx * 1009 + epoch_offset)
        points = self.points[idx]
        replace = len(points) < self.num_points
        choice = rng.choice(len(points), size=self.num_points, replace=replace)
        points = points[choice]

        if self.augment:
            rot = random_rotation_matrix(rng)
        else:
            rot = random_rotation_matrix(np.random.default_rng(self.seed + idx * 9176))
        points = (rot @ points.T).T.astype(np.float32)
        up = (rot @ self.gt_up[idx]).astype(np.float32)
        up /= max(np.linalg.norm(up), 1e-12)
        points = normalize_cloud(points)

        return (
            torch.from_numpy(points),
            torch.from_numpy(up),
            torch.tensor(int(self.category_id[idx]), dtype=torch.long),
        )


class PointNetPairwiseNet(nn.Module):
    def __init__(self, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = PairHead(hidden, dropout)

    def encode(self, points_bnc: torch.Tensor) -> torch.Tensor:
        return self.encoder(points_bnc.transpose(1, 2).contiguous()).transpose(1, 2)

    def forward(
        self,
        points_bnc: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
    ) -> torch.Tensor:
        feat = self.encode(points_bnc)
        return self.head(points_bnc, feat, pair_i, pair_j)


class DGCNNPairwiseNet(nn.Module):
    def __init__(self, hidden: int = 256, dropout: float = 0.2, k: int = 20):
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
            nn.Conv1d(128 + 512 + 1024 + 1024, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
        )
        self.head = PairHead(hidden, dropout)

    def encode(self, points_bnc: torch.Tensor) -> torch.Tensor:
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
        return self.conv5(x).transpose(1, 2).contiguous()

    def forward(
        self,
        points_bnc: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
    ) -> torch.Tensor:
        feat = self.encode(points_bnc)
        return self.head(points_bnc, feat, pair_i, pair_j)


class PairHead(nn.Module):
    def __init__(self, feat_dim: int, dropout: float):
        super().__init__()
        in_dim = feat_dim * 4 + 9
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 3),
        )

    def forward(
        self,
        points: torch.Tensor,
        feat: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
    ) -> torch.Tensor:
        pi = gather_batched(points, pair_i)
        pj = gather_batched(points, pair_j)
        fi = gather_batched(feat, pair_i)
        fj = gather_batched(feat, pair_j)
        pair_feat = torch.cat(
            [pi, pj, pi - pj, fi, fj, fi - fj, (fi - fj).abs()],
            dim=-1,
        )
        flat = pair_feat.reshape(-1, pair_feat.shape[-1])
        logits = self.net(flat)
        return logits.view(pair_feat.shape[0], pair_feat.shape[1], 3)


def gather_batched(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    expand_shape = (*index.shape, values.shape[-1])
    gather_index = index.unsqueeze(-1).expand(expand_shape)
    return torch.gather(values, 1, gather_index)


def sample_pairs(
    batch_size: int,
    num_points: int,
    num_pairs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_i = torch.randint(num_points, (batch_size, num_pairs), device=device)
    offset = torch.randint(1, num_points, (batch_size, num_pairs), device=device)
    pair_j = (pair_i + offset) % num_points
    return pair_i, pair_j


def deterministic_pairs(
    batch_size: int,
    num_points: int,
    num_pairs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(num_pairs, device=device)
    pair_i = (base * 15485863) % num_points
    pair_j = (base * 32452843 + 1) % num_points
    pair_i = pair_i.view(1, -1).repeat(batch_size, 1)
    pair_j = pair_j.view(1, -1).repeat(batch_size, 1)
    same = pair_i == pair_j
    pair_j = torch.where(same, (pair_j + 1) % num_points, pair_j)
    return pair_i, pair_j


def pair_targets(
    points: torch.Tensor,
    up: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    same_threshold_ratio: float,
) -> torch.Tensor:
    height = (points * up.unsqueeze(1)).sum(dim=2)
    hi = torch.gather(height, 1, pair_i)
    hj = torch.gather(height, 1, pair_j)
    delta = hi - hj
    span = (height.max(dim=1).values - height.min(dim=1).values).clamp_min(1e-6)
    tau = (same_threshold_ratio * span).unsqueeze(1)
    target = torch.full_like(pair_i, SAME)
    target = torch.where(delta > tau, torch.full_like(target, HIGHER), target)
    target = torch.where(delta < -tau, torch.full_like(target, LOWER), target)
    return target


def direction_from_pair_logits(
    points: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    logits: torch.Tensor,
    confidence_power: float,
) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    sign_score = probs[..., HIGHER] - probs[..., LOWER]
    if confidence_power != 1.0:
        sign_score = sign_score.sign() * sign_score.abs().pow(confidence_power)
    pi = gather_batched(points, pair_i)
    pj = gather_batched(points, pair_j)
    vote = ((pi - pj) * sign_score.unsqueeze(-1)).sum(dim=1)
    return F.normalize(vote, dim=1, eps=1e-6)


def relation_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    out = {"pair_acc": float((pred == target).float().mean().detach().cpu())}
    for label, name in ((LOWER, "lower_acc"), (SAME, "same_acc"), (HIGHER, "higher_acc")):
        mask = target == label
        if bool(mask.any()):
            out[name] = float((pred[mask] == label).float().mean().detach().cpu())
        else:
            out[name] = 0.0
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eval_pairs_per_cloud: int,
    same_threshold_ratio: float,
    confidence_power: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_pairs = 0
    total_correct = 0
    class_correct = torch.zeros(3, dtype=torch.float64)
    class_total = torch.zeros(3, dtype=torch.float64)
    errors = []

    for points, gt_up, _cat in loader:
        points = points.to(device)
        gt_up = gt_up.to(device)
        pair_i, pair_j = deterministic_pairs(
            points.shape[0], points.shape[1], eval_pairs_per_cloud, device
        )
        target = pair_targets(points, gt_up, pair_i, pair_j, same_threshold_ratio)
        logits = model(points, pair_i, pair_j)
        loss = F.cross_entropy(logits.reshape(-1, 3), target.reshape(-1))
        pred = logits.argmax(dim=-1)

        total_loss += float(loss.detach().cpu()) * target.numel()
        total_pairs += target.numel()
        total_correct += int((pred == target).sum().item())
        for cls in range(3):
            mask = target == cls
            class_total[cls] += int(mask.sum().item())
            class_correct[cls] += int((pred[mask] == cls).sum().item())

        pred_up = direction_from_pair_logits(
            points, pair_i, pair_j, logits, confidence_power
        )
        errors.append(angular_error_deg(pred_up, gt_up).detach().cpu())

    err = torch.cat(errors).numpy() if errors else np.asarray([], dtype=np.float32)
    per_class = class_correct / class_total.clamp_min(1.0)
    return {
        "loss": total_loss / max(total_pairs, 1),
        "pair_acc": total_correct / max(total_pairs, 1),
        "lower_acc": float(per_class[LOWER]),
        "same_acc": float(per_class[SAME]),
        "higher_acc": float(per_class[HIGHER]),
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


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_ds: Dataset = PairwiseUprightDataset(
        args.train_npz, args.num_points, args.seed, augment=True
    )
    test_ds: Dataset = PairwiseUprightDataset(
        args.test_npz, args.num_points, args.seed + 100000, augment=False
    )
    if args.train_limit > 0:
        train_ds = Subset(train_ds, range(min(args.train_limit, len(train_ds))))
    if args.test_limit > 0:
        test_ds = Subset(test_ds, range(min(args.test_limit, len(test_ds))))

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
        model: nn.Module = DGCNNPairwiseNet(args.hidden, args.dropout).to(device)
    else:
        model = PointNetPairwiseNet(args.hidden, args.dropout).to(device)
    if args.data_parallel:
        if device.type != "cuda":
            raise ValueError("--data-parallel requires CUDA")
        model = nn.DataParallel(model)

    weight = torch.tensor([1.0, args.same_weight, 1.0], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv) if args.log_csv else out_dir / "train_log.csv"
    best_acc10 = -1.0

    print(f"device={device} arch={args.arch}")
    print(
        f"train_clouds={len(train_ds)} test_clouds={len(test_ds)} "
        f"pairs_per_cloud={args.pairs_per_cloud} eval_pairs_per_cloud={args.eval_pairs_per_cloud}"
    )
    print(
        f"same_threshold_ratio={args.same_threshold_ratio} same_weight={args.same_weight} "
        f"confidence_power={args.confidence_power}"
    )

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        elif isinstance(train_ds, Subset) and hasattr(train_ds.dataset, "set_epoch"):
            train_ds.dataset.set_epoch(epoch)

        model.train()
        running = 0.0
        count = 0
        rel_sum = {"pair_acc": 0.0, "lower_acc": 0.0, "same_acc": 0.0, "higher_acc": 0.0}
        rel_batches = 0
        for points, gt_up, _cat in train_loader:
            points = points.to(device)
            gt_up = gt_up.to(device)
            pair_i, pair_j = sample_pairs(
                points.shape[0], points.shape[1], args.pairs_per_cloud, device
            )
            target = pair_targets(
                points, gt_up, pair_i, pair_j, args.same_threshold_ratio
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(points, pair_i, pair_j)
            loss = F.cross_entropy(
                logits.reshape(-1, 3), target.reshape(-1), weight=weight
            )
            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu()) * target.numel()
            count += target.numel()
            batch_rel = relation_metrics(logits.argmax(dim=-1), target)
            for key, value in batch_rel.items():
                rel_sum[key] += value
            rel_batches += 1

        scheduler.step()
        metrics = evaluate(
            model,
            test_loader,
            device,
            args.eval_pairs_per_cloud,
            args.same_threshold_ratio,
            args.confidence_power,
        )
        train_loss = running / max(count, 1)
        train_pair_acc = rel_sum["pair_acc"] / max(rel_batches, 1)
        row = {"epoch": epoch, "train_loss": train_loss, "train_pair_acc": train_pair_acc, **metrics}
        write_log(log_csv, row)

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"train_pair_acc={train_pair_acc*100:.2f}% val_loss={metrics['loss']:.4f} "
            f"pair_acc={metrics['pair_acc']*100:.2f}% "
            f"lower={metrics['lower_acc']*100:.2f}% same={metrics['same_acc']*100:.2f}% "
            f"higher={metrics['higher_acc']*100:.2f}% "
            f"mean={metrics['mean_err']:.2f} median={metrics['median_err']:.2f} "
            f"acc10={metrics['acc10']*100:.2f}% flip={metrics['flip']*100:.2f}%",
            flush=True,
        )

        if metrics["acc10"] >= best_acc10:
            best_acc10 = metrics["acc10"]
            torch.save(
                {
                    "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    "args": vars(args),
                    "metrics": metrics,
                    "pair_classes": ["lower", "same", "higher"],
                },
                out_dir / "best.pth",
            )
            print(f"[save] {out_dir / 'best.pth'} acc10={best_acc10*100:.2f}%")

    torch.save(
        {
            "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "args": vars(args),
            "pair_classes": ["lower", "same", "higher"],
        },
        out_dir / "final.pth",
    )
    print(f"[done] wrote {out_dir / 'final.pth'}")


if __name__ == "__main__":
    main()

"""Sanity-check: overfit DifferentiableLieUprightRefineNet on a single fixed batch.

If the architecture, loss, and gradient flow are healthy, a few hundred AdamW
steps on the same batch should drive `geo_final` to well under 1°. If it
plateaus at the coarse-init error, the refine head is not effectively
receiving (or acting on) training signal.

Typical usage:
    python3 scripts/overfit_dlie_refine.py --steps 500 --batch_size 16

Prints per-step:
    step   loss    geo_init    geo_iter1 ...   geo_final
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from network_lie import DifferentiableLieUprightRefineNet
from scripts.train_dlie_refine import (
    UprightNet15RotatedDataset,
    compute_multi_iter_loss,
    geodesic_from_rotation_up_column,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(REPO / "datasets" / "uprightnet15"))
    ap.add_argument("--ref_ckpt", default=str(
        REPO.parent / "uprightnet-reference" / "model" / "model.pth"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_points", type=int, default=2048)
    ap.add_argument("--num_iters", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_trunk", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip_grad", type=float, default=5.0)
    ap.add_argument("--hard_only", action="store_true",
                    help="select samples with largest coarse-init error (tests "
                         "whether head can learn hard cases in isolation)")
    ap.add_argument("--hard_pool", type=int, default=500,
                    help="how many candidates to scan before picking hardest")
    ap.add_argument("--labels", type=int, nargs="+", default=None,
                    help="restrict to these class labels (e.g. 0 8 14 for bed/lamp/vase)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    print(f"Overfit: bs={args.batch_size} steps={args.steps} "
          f"num_iters={args.num_iters} (continuous head, no per-iter schedule)")
    print(f"lr_head={args.lr_head}  lr_trunk={args.lr_trunk}")

    # ---- Pick one fixed batch from train set ----
    ds = UprightNet15RotatedDataset(args.data_dir, "train", args.num_points)
    rng = np.random.default_rng(args.seed)

    # Optional label filter
    label_arr = np.asarray(ds.labels)
    if args.labels is not None:
        mask = np.isin(label_arr, args.labels)
        candidate_idx = np.where(mask)[0]
        print(f"Filtered to labels {args.labels}: {len(candidate_idx)} candidates")
    else:
        candidate_idx = np.arange(len(ds))

    # ---- Model (built early so --hard_only can score candidates with it) ----
    model = DifferentiableLieUprightRefineNet(
        num_iters=args.num_iters,
    ).to(device)
    model.load_backbone_from_ckpt(args.ref_ckpt)

    if args.hard_only:
        # Draw a pool, score each by coarse-init geodesic, keep the worst
        pool_size = min(args.hard_pool, len(candidate_idx))
        pool = rng.choice(candidate_idx, size=pool_size, replace=False)
        model.eval()
        errs_scored = []
        with torch.no_grad():
            chunk = 32
            for start in range(0, pool_size, chunk):
                sub = pool[start:start + chunk]
                Ps, ups = [], []
                for i in sub.tolist():
                    P_i, up_i, _ = ds[i]
                    Ps.append(P_i)
                    ups.append(up_i)
                Pb = torch.stack(Ps).to(device)
                ub = torch.stack(ups).to(device)
                out = model(Pb)
                err = geodesic_from_rotation_up_column(out["R_iters"][0], ub)
                errs_scored.extend(err.cpu().tolist())
        errs_scored = np.array(errs_scored)
        order = np.argsort(-errs_scored)  # descending
        idx = pool[order[:args.batch_size]]
        print(f"[hard_only] pool={pool_size}  hardest init errors (deg): "
              f"{[f'{math.degrees(errs_scored[order[i]]):.1f}' for i in range(min(args.batch_size, 5))]} ... "
              f"{[f'{math.degrees(errs_scored[order[i]]):.1f}' for i in range(max(0, args.batch_size - 3), args.batch_size)]}")
    else:
        idx = rng.choice(candidate_idx, size=args.batch_size, replace=False)

    Ps, ups, labels = [], [], []
    for i in idx.tolist():
        P, up, lab = ds[i]
        Ps.append(P)
        ups.append(up)
        labels.append(lab)
    P = torch.stack(Ps).to(device)
    gt_up = torch.stack(ups).to(device)
    labels = torch.stack(labels)
    print(f"Fixed batch: P={tuple(P.shape)}  gt_up={tuple(gt_up.shape)}  "
          f"label histogram={np.bincount(labels.numpy(), minlength=15).tolist()}")

    trunk_params = list(model.trunk.parameters())
    head_params = list(model.refine_head.parameters())
    groups = [{"params": head_params, "lr": args.lr_head, "name": "head"}]
    trunk_frozen = args.lr_trunk <= 0
    if args.lr_trunk > 0:
        groups.append({"params": trunk_params, "lr": args.lr_trunk, "name": "trunk"})
    else:
        for p in trunk_params:
            p.requires_grad = False
        print("Trunk frozen — BN running stats kept fixed (trunk in eval mode)")
    optim = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    model.train()
    if trunk_frozen:
        model.trunk.eval()
    print()
    header = "  step   loss   " + "  ".join(
        ["init"] + [f"it{i+1}" for i in range(args.num_iters)])
    print(header)

    best_final = float("inf")
    for step in range(1, args.steps + 1):
        out = model(P)
        loss, per_iter = compute_multi_iter_loss(
            out["R_iters"], gt_up, gamma=args.gamma)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=args.clip_grad)

        # Snapshot the gradient norm on the refine head (once, early) to see
        # whether signal actually reaches it.
        if step == 1:
            total_gn = 0.0
            for p in head_params:
                if p.grad is not None:
                    total_gn += p.grad.detach().pow(2).sum().item()
            print(f"  [step 1] refine_head grad norm = {total_gn ** 0.5:.4e}")

        optim.step()

        geo_final_deg = math.degrees(per_iter[-1])
        if geo_final_deg < best_final:
            best_final = geo_final_deg

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            per_deg = [math.degrees(p) for p in per_iter]
            cells = "  ".join(f"{d:5.2f}" for d in per_deg)
            print(f"  {step:5d}  {loss.item():.4f}  {cells}")

    # ---- Final snapshot ----
    model.eval()
    with torch.no_grad():
        out = model(P)
        err_init = geodesic_from_rotation_up_column(out["R_iters"][0], gt_up)
        err_final = geodesic_from_rotation_up_column(out["R_iters"][-1], gt_up)
        err_init_deg = (err_init * 180.0 / math.pi).cpu().numpy()
        err_final_deg = (err_final * 180.0 / math.pi).cpu().numpy()

    print()
    print("Per-sample final error (deg):")
    for i, (ei, ef, lab) in enumerate(zip(err_init_deg, err_final_deg, labels.tolist())):
        print(f"    [{i:2d}] label={lab:2d}  init={ei:6.2f}  final={ef:6.2f}")
    print()
    print(f"Best geo_final seen during training: {best_final:.3f}°")
    print(f"Eval-mode final: mean={err_final_deg.mean():.3f}°  "
          f"max={err_final_deg.max():.3f}°  "
          f"init mean={err_init_deg.mean():.3f}°")


if __name__ == "__main__":
    main()

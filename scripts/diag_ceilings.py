"""
Four ceiling diagnostics to separate "weight-transfer" from "architecture"
causes of the PolarLieUprightNet failure.

All experiments evaluate on a SUBSET of the UprightNet15 test set
(default 1000 samples) to stay fast; extend with --num_samples for full eval.

Experiment 1 · PCA-frame oracle ceiling
    For each sample, enumerate all 48 axis-sign combinations of the PCA
    eigenvectors (3! axis orderings × 2^3 signs = 48 rotations; restrict to
    determinant=+1 giving 24 proper rotations). Pick the rotation whose
    up-column has minimum angle to GT. Report oracle acc@10 / mean error.
    Answers: "What is the best-case performance if the flip head is perfect
             AND refine is not even used?"

Experiment 2 · Refine-path ceiling
    Starting from the Experiment 1 best PCA frame, enumerate 26-direction
    × 3-step beam search (each step Δω ∈ {±π/6 * e_x, ±π/6 * e_y, ±π/6 * e_z,
    and diagonals}, 26 options per step). Pick final rotation closest to GT.
    Answers: "If the refine head is also perfect, what does max_angle=π/6,
             T=3 achieve on top of the PCA-frame oracle?"

Experiment 3 · Trunk feature OOD diagnostic
    Run trunk on P_t = R_PCA^T @ P (best frame from Exp 1) and compute
    support_logits. The GT pid label is defined in canonical frame, so we
    rotate pid by the true rotation delta between PCA frame and canonical
    frame (i.e. pid stays with the object). We measure BCE and IoU@0.5.
    Compare to the in-distribution case: trunk on P_rotated (the original
    training input) vs same pid.
    Answers: "Does the Pang checkpoint generalise to PCA-frame inputs?"

Experiment 4 · Equivariant ckpt eval
    Load the latest L=1 equivariant ckpt and evaluate on the full or subset
    test set. Report mean/median/acc@10/flip-rate.
    Answers: "Does any equivariant approach actually work on this task?"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from network_lie import UprightNetTrunk


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def pca_eigenvectors(points: torch.Tensor, eps: float = 1e-6):
    """Return (B, 3, 3) matrix V whose columns are the PCA eigenvectors
    (ascending eigenvalues)."""
    B, N, _ = points.shape
    centered = points - points.mean(dim=1, keepdim=True)
    cov = torch.bmm(centered.transpose(1, 2), centered) / (N - 1)
    cov = cov + eps * torch.eye(3, device=points.device, dtype=points.dtype)
    _, V = torch.linalg.eigh(cov)
    return V


def all_proper_sign_axis_combinations():
    """Enumerate all 48 axis-order × sign combinations, keep the 24 with
    determinant = +1. Returns a (24, 3, 3) tensor of basis-permutation
    matrices acting on (v_small, v_mid, v_large) columns, where v_small is
    the smallest-eigenvalue eigenvector.
    """
    from itertools import permutations, product
    mats = []
    for perm in permutations([0, 1, 2]):
        for signs in product([-1, 1], repeat=3):
            M = torch.zeros(3, 3)
            for col_out, col_in in enumerate(perm):
                M[col_in, col_out] = signs[col_out]
            if torch.det(M) > 0:
                mats.append(M)
    return torch.stack(mats, dim=0)  # (24, 3, 3)


def angular_error_deg(up_pred, gt_up):
    cos = (up_pred * gt_up).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos) * (180.0 / np.pi)


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------


def exp1_pca_oracle(points: torch.Tensor, gt_up: torch.Tensor, device):
    """Return (B,) oracle angular error (deg) for best PCA frame."""
    V = pca_eigenvectors(points)          # (B, 3, 3)
    combos = all_proper_sign_axis_combinations().to(device)  # (24, 3, 3)

    # Build candidate rotations: R = V @ M, the up-column is R[:, 1]
    B = points.shape[0]
    K = combos.shape[0]
    Rs = torch.einsum("bij,kjl->bkil", V, combos)  # (B, K, 3, 3)
    ups = Rs[:, :, :, 1]                           # (B, K, 3) candidate up vectors
    gt = gt_up.unsqueeze(1)                        # (B, 1, 3)
    cos = (ups * gt).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)  # (B, K)
    err = torch.acos(cos) * (180.0 / np.pi)
    best_err, best_idx = err.min(dim=1)
    best_R = Rs.gather(1, best_idx.view(B, 1, 1, 1).expand(B, 1, 3, 3)).squeeze(1)
    return best_err, best_R


# ---------------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------------


def _skew(v):
    O = torch.zeros_like(v[:, 0])
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return torch.stack([
        torch.stack([O, -z, y], -1),
        torch.stack([z, O, -x], -1),
        torch.stack([-y, x, O], -1),
    ], dim=1)


def _exp_so3(omega, eps=1e-8):
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=eps)
    K = _skew(omega)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    a = (torch.sin(theta) / theta).unsqueeze(-1)
    b = ((1 - torch.cos(theta)) / (theta ** 2)).unsqueeze(-1)
    return I + a * K + b * (K @ K)


def _step_directions(step_size: float, device):
    """26-direction candidates at magnitude step_size: 3 axes + 6 diagonals +
    8 cube-corners and their negatives. Plus a 'no move' direction for
    realism. Total 27."""
    base = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                base.append([x, y, z])
    d = torch.tensor(base, dtype=torch.float32, device=device)  # (27, 3)
    norms = d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return d * step_size / norms  # non-zero directions at magnitude step_size; 'zero' row ≈ step_size/eps (ignore)
    # Note: the (0,0,0) row gets normalised to huge numbers; we filter it
    # in the caller by skipping rows with all-zero base direction.


def exp2_refine_oracle(best_R: torch.Tensor, gt_up: torch.Tensor,
                       device, max_angle: float, T: int):
    """Given per-sample starting rotation best_R, do beam search over T steps
    where each step picks an Δω from a 26-direction grid at magnitude
    max_angle. Returns (B,) oracle error after T steps.

    Beam is per-sample (no beam aggregation across samples) and we keep only
    the top-K = beam_size best candidates each step to stay tractable.
    """
    B = best_R.shape[0]
    # Build direction grid
    dirs_full = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                if (x, y, z) == (0, 0, 0):
                    continue
                dirs_full.append([x, y, z])
    dirs = torch.tensor(dirs_full, dtype=torch.float32, device=device)  # (26, 3)
    dirs = dirs * max_angle / dirs.norm(dim=-1, keepdim=True)
    # Also include "no move"
    dirs = torch.cat([torch.zeros(1, 3, device=device), dirs], dim=0)  # (27, 3)
    K = dirs.shape[0]

    beam_size = 8
    # State: (B, beam, 3, 3)
    beam = best_R.unsqueeze(1)  # (B, 1, 3, 3)
    gt = gt_up.unsqueeze(1)     # (B, 1, 3)

    for step in range(T):
        beam_n = beam.shape[1]
        # Expand beam × dirs: (B, beam, K, 3, 3)
        dR = _exp_so3(dirs).unsqueeze(0).unsqueeze(0)  # (1, 1, K, 3, 3)
        beam_expand = beam.unsqueeze(2)                  # (B, beam, 1, 3, 3)
        candidates = torch.matmul(beam_expand, dR)       # (B, beam, K, 3, 3)
        candidates = candidates.reshape(B, beam_n * K, 3, 3)
        # Score each candidate
        ups_cand = candidates[:, :, :, 1]  # (B, beam*K, 3)
        cos = (ups_cand * gt).sum(-1)       # (B, beam*K)
        # Keep top beam_size by cos
        top_cos, top_idx = cos.topk(beam_size, dim=1)
        beam = candidates.gather(1, top_idx.view(B, beam_size, 1, 1).expand(B, beam_size, 3, 3))

    # Final best
    ups_final = beam[:, :, :, 1]
    cos = (ups_final * gt).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    err = torch.acos(cos) * (180.0 / np.pi)
    return err.min(dim=1).values


# ---------------------------------------------------------------------------
# Experiment 3: trunk OOD
# ---------------------------------------------------------------------------


def exp3_trunk_ood(
    points_original: torch.Tensor,  # canonical (training) frame
    points_rotation: torch.Tensor,  # SO(3)-rotated version used at Pang training
    pid_gt: torch.Tensor,            # (B, N) support label in canonical frame
    best_R: torch.Tensor,            # (B, 3, 3) best PCA frame from Exp 1
    trunk: UprightNetTrunk,
):
    """Compare trunk support prediction quality on three inputs:
        (a) points_rotation   — original Pang training distribution
        (b) points_original   — the canonical pose directly (perfect alignment)
        (c) points @ best_R    — rotated into PCA frame (Exp 1 oracle choice)
    """
    trunk.eval()
    out = {}
    with torch.no_grad():
        for name, P in [
            ("train_rot", points_rotation),
            ("canonical", points_original),
            ("pca_frame", torch.bmm(points_rotation, best_R)),
            # Note: points @ R is equivalent to R^T @ p per row, which is the
            # canonicalised view after applying best_R as the frame. We use
            # the rotated input since that's what the network was trained on.
        ]:
            _, sup = trunk(P)
            sup_pred = (sup > 0.5).float()
            # Per-sample IoU
            inter = (sup_pred * pid_gt).sum(-1)
            union = ((sup_pred + pid_gt) > 0).float().sum(-1).clamp(min=1)
            iou = inter / union
            out[name] = {
                "iou_mean": iou.mean().item(),
                "iou_median": iou.median().item(),
                "sup_rate": sup_pred.mean().item(),
            }
    return out


# ---------------------------------------------------------------------------
# Experiment 4: equivariant ckpt eval
# ---------------------------------------------------------------------------


def exp4_equiv_ckpt(data, rotm, device, num_samples=None):
    """Evaluate the latest L=1 equivariant checkpoint on UprightNet15."""
    try:
        from config import opts  # noqa
        from network_equivariant import build_equivariant_model  # noqa
    except Exception as e:
        print(f"  [Skip] Could not import equivariant network: {e}")
        return None

    # Find latest L=1 checkpoint. Our logs show many, pick newest
    import glob
    ckpts = sorted(glob.glob(str(REPO / "models" / "equivariant_best_*.pth")))
    if not ckpts:
        return None
    ckpt_path = ckpts[-1]

    # Build model with L=1 config matching train_l1.sh
    class Opts:
        irreps_hidden = "128x0e+128x1o"
        lmax = 1
        max_radius = 0.1
        equi_layers = 8
        num_radial_basis = 16
        radial_neurons = 128
        num_neighbors = 32.0
        vmf_kappa_init = 1.0
        conv_type = "depthwise"

    try:
        model = build_equivariant_model(Opts()).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd)
    except Exception as e:
        print(f"  [Skip] Could not load equivariant ckpt {ckpt_path}: {e}")
        return None
    model.eval()

    from torch_geometric.data import Data, Batch
    N = data.shape[0] if num_samples is None else min(num_samples, data.shape[0])
    errs = []
    BS = 16
    with torch.no_grad():
        for s in range(0, N, BS):
            e = min(N, s + BS)
            batch_list = []
            for i in range(s, e):
                pos = torch.from_numpy(data[i]).float()
                batch_list.append(Data(pos=pos))
            batch = Batch.from_data_list(batch_list).to(device)
            out = model(batch)
            mu = out["direction_mu"]
            gt = torch.from_numpy(rotm[s:e, :, 1]).to(device).float()
            cos = (mu * gt).sum(-1).abs().clamp(-1 + 1e-7, 1 - 1e-7)  # antipodal
            err = torch.acos(cos) * (180.0 / np.pi)
            errs.append(err.cpu())
    errs = torch.cat(errs).numpy()
    return {
        "ckpt": ckpt_path,
        "mean": errs.mean(),
        "median": np.median(errs),
        "acc10": (errs < 10).mean() * 100,
        "flip_rate": (errs > 90).mean() * 100,
        "n": len(errs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_exp4", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    data_dir = REPO / "datasets" / "uprightnet15"

    original = np.load(data_dir / "test_original.npy").astype(np.float32)
    rotation = np.load(data_dir / "test_rotation.npy").astype(np.float32)
    rotm = np.load(data_dir / "rotm_test.npy").astype(np.float32)
    pid = np.load(data_dir / "pid_test.npy").astype(np.float32)
    labels = np.load(data_dir / "labels_test.npy").astype(np.int32).squeeze()

    # Random subsample for speed
    np.random.seed(args.seed)
    total = original.shape[0]
    idx = np.random.choice(total, size=min(args.num_samples, total), replace=False)
    original, rotation, rotm, pid, labels = (
        original[idx], rotation[idx], rotm[idx], pid[idx], labels[idx])
    N = len(idx)
    print(f"Subset: {N} samples")

    P_orig = torch.from_numpy(original).to(device)
    P_rot = torch.from_numpy(rotation).to(device)
    gt_up = torch.from_numpy(rotm[:, :, 1]).to(device)
    pid_t = torch.from_numpy(pid).to(device)

    # ====== EXPERIMENT 1 ======
    print()
    print("=" * 72)
    print("EXP 1 · PCA-frame oracle ceiling")
    print("=" * 72)
    all_err1 = []
    all_bestR = []
    for s in tqdm(range(0, N, args.batch_size)):
        e = min(N, s + args.batch_size)
        err, bR = exp1_pca_oracle(P_rot[s:e], gt_up[s:e], device)
        all_err1.append(err.cpu())
        all_bestR.append(bR)
    err1 = torch.cat(all_err1).numpy()
    bestR = torch.cat(all_bestR, dim=0)
    print(f"  mean={err1.mean():.2f}°  median={np.median(err1):.2f}°")
    print(f"  acc@10={(err1<10).mean()*100:.2f}%  acc@30={(err1<30).mean()*100:.2f}%")
    print(f"  flip rate={(err1>90).mean()*100:.2f}%")

    # Per-category breakdown
    print()
    print("  Per-category acc@10 (PCA oracle):")
    CATS = ["bed", "bench", "bottle", "bowl", "car", "chair", "cone", "cup",
            "lamp", "monitor", "sofa", "stool", "table", "toilet", "vase"]
    for lab in sorted(np.unique(labels)):
        m = labels == lab
        if m.sum() == 0:
            continue
        e1 = err1[m]
        print(f"    {CATS[lab]:>8}  N={m.sum():>4}  mean={e1.mean():6.2f}  "
              f"acc@10={(e1<10).mean()*100:5.1f}%")

    # ====== EXPERIMENT 2 ======
    print()
    print("=" * 72)
    print("EXP 2 · Refine-path oracle (from PCA oracle, T=3, max_angle=π/6=30°)")
    print("=" * 72)
    all_err2 = []
    for s in tqdm(range(0, N, args.batch_size)):
        e = min(N, s + args.batch_size)
        err = exp2_refine_oracle(
            bestR[s:e], gt_up[s:e], device, max_angle=np.pi / 6, T=3)
        all_err2.append(err.cpu())
    err2 = torch.cat(all_err2).numpy()
    print(f"  mean={err2.mean():.2f}°  median={np.median(err2):.2f}°")
    print(f"  acc@10={(err2<10).mean()*100:.2f}%  acc@30={(err2<30).mean()*100:.2f}%")
    print(f"  Gain over Exp1: +{(err2<10).mean()*100 - (err1<10).mean()*100:.2f}%")

    # Also test T=5, max_angle=π/3
    print()
    print("EXP 2b · T=5, max_angle=π/3=60° (loosened)")
    all_err2b = []
    for s in tqdm(range(0, N, args.batch_size)):
        e = min(N, s + args.batch_size)
        err = exp2_refine_oracle(
            bestR[s:e], gt_up[s:e], device, max_angle=np.pi / 3, T=5)
        all_err2b.append(err.cpu())
    err2b = torch.cat(all_err2b).numpy()
    print(f"  mean={err2b.mean():.2f}°  median={np.median(err2b):.2f}°")
    print(f"  acc@10={(err2b<10).mean()*100:.2f}%")

    # ====== EXPERIMENT 3 ======
    print()
    print("=" * 72)
    print("EXP 3 · Trunk OOD diagnostic (support IoU across input frames)")
    print("=" * 72)
    trunk = UprightNetTrunk().to(device)
    sd = torch.load(
        str(REPO.parent / "uprightnet-reference" / "model" / "model.pth"),
        map_location=device, weights_only=False,
    )
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    trunk.net.load_state_dict(sd)
    # Evaluate in batches
    results = {"train_rot": [], "canonical": [], "pca_frame": []}
    for s in tqdm(range(0, N, args.batch_size)):
        e = min(N, s + args.batch_size)
        out = exp3_trunk_ood(
            P_orig[s:e], P_rot[s:e], pid_t[s:e], bestR[s:e], trunk)
        for k in results:
            results[k].append(out[k])
    for k, lst in results.items():
        iou = np.mean([x["iou_mean"] for x in lst])
        sup = np.mean([x["sup_rate"] for x in lst])
        print(f"  {k:>10}  support_IoU(mean)={iou:.3f}  sup_rate={sup:.3f}")
    print()
    print("  Interpretation:")
    print("    train_rot  ≫  canonical  ≫  pca_frame  →  trunk is brittle to frame changes")
    print("    train_rot  ≈  pca_frame            →  trunk transfers to PCA frame OK")
    print("    train_rot  ≫  pca_frame            →  trunk is OOD on PCA frame")

    # ====== EXPERIMENT 4 ======
    if not args.skip_exp4:
        print()
        print("=" * 72)
        print("EXP 4 · Latest L=1 equivariant checkpoint")
        print("=" * 72)
        res = exp4_equiv_ckpt(rotation, rotm, device, num_samples=N)
        if res is None:
            print("  (skipped — no ckpt or import failed)")
        else:
            print(f"  ckpt: {res['ckpt']}")
            print(f"  N={res['n']}  mean={res['mean']:.2f}°  median={res['median']:.2f}°")
            print(f"  acc@10={res['acc10']:.2f}%  flip={res['flip_rate']:.2f}%")

    # ====== VERDICT ======
    print()
    print("=" * 72)
    print("VERDICT TEMPLATE (fill in)")
    print("=" * 72)
    print(f"  Exp1 acc@10 ({(err1<10).mean()*100:.1f}%) tells us:")
    if (err1 < 10).mean() > 0.9:
        print("    ✅ PCA architecture is fine — oracle can reach top-tier")
    elif (err1 < 10).mean() > 0.5:
        print("    ⚠️ PCA partially fine — fails on some classes; may work with refine")
    else:
        print("    ❌ PCA architecture is limiting — need different init")

    print(f"  Exp2 - Exp1 ({(err2<10).mean()*100 - (err1<10).mean()*100:.1f}% gain) tells us:")
    if (err2 < 10).mean() - (err1 < 10).mean() > 0.2:
        print("    ✅ Refine head has room to help — beam-search gains big")
    else:
        print("    ⚠️ Refine provides little oracle gain at this max_angle/T")

    trunk_ood_gap = results["train_rot"][0]["iou_mean"] - results["pca_frame"][0]["iou_mean"]
    print(f"  Exp3 IoU gap (train_rot - pca_frame) across batches: check above")


if __name__ == "__main__":
    main()

"""Zero-shot evaluation of DifferentiableLieUprightNet (flip+refine head variant)
with trunk loaded from the reference (Pang 2022) checkpoint and the flip/
refine heads randomly initialised.

Expectation: since iteration 0 is a differentiable support-plane baseline,
and the flip head is initialised with a strong negative bias so that
sigmoid(output) ≈ 0, the zero-shot performance should match the reference
baseline (~93% acc@10) at hard_flip=True. With STE soft flip, the forward
pass is identical to hard_flip, so hard/soft outputs agree.

Reports both modes + per-category.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from network_lie import DifferentiableLieUprightNet


def angular_error_deg(up, gt):
    cos = (up * gt).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos) * (180.0 / np.pi)


def main():
    device = torch.device("cuda")
    data_dir = REPO / "datasets" / "uprightnet15"
    rotation = np.load(data_dir / "test_rotation.npy").astype(np.float32)
    rotm = np.load(data_dir / "rotm_test.npy").astype(np.float32)
    labels = np.load(data_dir / "labels_test.npy").astype(np.int32).squeeze()
    N = rotation.shape[0]
    gt_up = rotm[:, :, 1]
    print(f"Loaded {N} test samples")

    model = DifferentiableLieUprightNet(num_iters=3).to(device)
    model.load_backbone_from_ckpt(
        str(REPO.parent / "uprightnet-reference" / "model" / "model.pth")
    )
    model.eval()

    BS = 32
    errs_hard = np.zeros(N, dtype=np.float32)
    errs_soft = np.zeros(N, dtype=np.float32)
    errs_init = np.zeros(N, dtype=np.float32)
    flip_probs = np.zeros(N, dtype=np.float32)
    with torch.no_grad():
        for s in tqdm(range(0, N, BS)):
            e = min(N, s + BS)
            P = torch.from_numpy(rotation[s:e]).to(device)
            gt = torch.from_numpy(gt_up[s:e]).to(device)
            out_h = model(P, hard_flip=True)
            errs_hard[s:e] = angular_error_deg(out_h["up"], gt).cpu().numpy()
            errs_init[s:e] = angular_error_deg(out_h["up_0"], gt).cpu().numpy()
            flip_probs[s:e] = out_h["flip_prob"].cpu().numpy()
            out_s = model(P, hard_flip=False)
            errs_soft[s:e] = angular_error_deg(out_s["up"], gt).cpu().numpy()

    def _summary(name, err):
        print(f"{name}:")
        print(f"  mean={err.mean():.2f}°  median={np.median(err):.2f}°")
        print(f"  acc@5={(err<5).mean()*100:.2f}%  acc@10={(err<10).mean()*100:.2f}%  "
              f"acc@30={(err<30).mean()*100:.2f}%")
        print(f"  flip(>90°)={(err>90).mean()*100:.2f}%  "
              f"near180(>170°)={(err>170).mean()*100:.2f}%")

    print()
    _summary("[A] up_0 (coarse init only, pure differentiable baseline)", errs_init)
    print()
    _summary("[B] hard_flip=True (discrete flip decision, eval mode)", errs_hard)
    print()
    _summary("[C] hard_flip=False (STE soft flip, training mode)", errs_soft)

    print()
    print(f"flip_prob stats: mean={flip_probs.mean():.4f}  std={flip_probs.std():.4f}")
    print(f"  fraction flipped at threshold 0.5: {(flip_probs > 0.5).mean() * 100:.2f}%")

    CATS = ["bed", "bench", "bottle", "bowl", "car", "chair", "cone", "cup",
            "lamp", "monitor", "sofa", "stool", "table", "toilet", "vase"]
    print()
    print("Per-category for [A] up_0 (coarse init):")
    print(f"{'cat':>8}  {'N':>5}  {'mean':>7}  {'acc@10':>7}  {'flip%':>7}")
    for lab in sorted(np.unique(labels)):
        m = labels == lab
        e = errs_init[m]
        print(f"{CATS[lab]:>8}  {m.sum():>5}  {e.mean():7.2f}  "
              f"{(e<10).mean()*100:6.2f}  {(e>90).mean()*100:6.2f}")

    out_path = REPO / "logs" / "dlie_zeroshot.npz"
    np.savez(out_path,
             errs_init=errs_init, errs_hard=errs_hard, errs_soft=errs_soft,
             flip_prob=flip_probs, label=labels)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

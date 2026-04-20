"""Zero-shot evaluation of PolarLieUprightNet with trunk loaded from Pang ckpt
and randomly-initialised flip/refine heads."""

from pathlib import Path
import sys
import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from network_lie import PolarLieUprightNet


def angular_error_deg(up, gt):
    cos = (up * gt).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos) * (180.0 / torch.pi)


def main():
    device = torch.device("cuda")
    data_dir = REPO / "datasets" / "uprightnet15"
    rotation = np.load(data_dir / "test_rotation.npy").astype(np.float32)
    rotm = np.load(data_dir / "rotm_test.npy").astype(np.float32)
    N = rotation.shape[0]
    gt_up = rotm[:, :, 1]

    model = PolarLieUprightNet(num_iters=3).to(device)
    model.load_backbone_from_ckpt(
        str(REPO.parent / "uprightnet-reference" / "model" / "model.pth")
    )
    model.eval()

    BS = 32
    errs = np.zeros(N, dtype=np.float32)
    flip_probs = np.zeros(N, dtype=np.float32)
    with torch.no_grad():
        for s in tqdm(range(0, N, BS)):
            e = min(N, s + BS)
            P = torch.from_numpy(rotation[s:e]).to(device)
            gt = torch.from_numpy(gt_up[s:e]).to(device)
            out = model(P, hard_flip=True)  # discrete flip at eval
            errs[s:e] = angular_error_deg(out["up"], gt).cpu().numpy()
            flip_probs[s:e] = out["flip_prob"].cpu().numpy()

    print()
    print("Zero-shot PolarLieUprightNet (random flip head, hard_flip=True):")
    print(f"  mean:   {errs.mean():.2f}°")
    print(f"  median: {np.median(errs):.2f}°")
    print(f"  acc@10: {(errs < 10).mean() * 100:.2f}%")
    print(f"  acc@30: {(errs < 30).mean() * 100:.2f}%")
    print(f"  flip rate (>90°): {(errs > 90).mean() * 100:.2f}%")
    print(f"  flip_prob distribution: mean={flip_probs.mean():.3f}, "
          f"std={flip_probs.std():.3f}")
    print(f"  fraction flipped (flip_prob>0.5): {(flip_probs > 0.5).mean() * 100:.2f}%")

    # Also test without flipping (hard_flip=False but prob < 0.5 always by init)
    # to see the pure PCA axis baseline
    print()
    print("For reference — what PCA axis alone achieves, counting")
    print("      each sample twice with both sign choices:")
    errs_flip_all = errs.copy()
    # simulate "what if we ALWAYS flipped or NEVER flipped" — since random
    # init gives near 50/50, the error distribution already tells us: for
    # samples that got err<10, PCA axis + chosen sign was correct.
    # The gap between (err<10) and (err>170) tells us: how much is the
    # flip signal capturable vs lost.
    hit_rate = (errs < 10).mean() * 100
    flip_rate = (errs > 170).mean() * 100
    correct_axis_rate = hit_rate + flip_rate
    print(f"  correct-axis rate (err<10 OR err>170): {correct_axis_rate:.2f}%")
    print(f"    → PCA axis gets the up axis right in ~{correct_axis_rate:.1f}% of samples;")
    print(f"      the remaining {100 - correct_axis_rate:.1f}% need refinement "
          f"to find the axis itself.")

    out_path = REPO / "logs" / "polar_zeroshot.npz"
    np.savez(out_path, error_deg=errs, flip_prob=flip_probs)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

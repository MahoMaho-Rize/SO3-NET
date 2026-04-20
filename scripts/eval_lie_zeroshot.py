"""
Zero-shot evaluation of LieUprightNet initialised from the Pang 2022 checkpoint.

The SO3Head is randomly initialised (near-zero output by design), so the
model's predictions come almost entirely from the PCA equivariant initialisation
passed through T iterations of near-identity updates.

This establishes the starting point from which fine-tuning begins.
"""

from pathlib import Path
import sys
import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from network_lie import LieUprightNet


def angular_error_deg(up_pred, gt_up):
    cos = (up_pred * gt_up).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    # No antipodal collapse here — the flip IS the failure we want to measure
    return torch.acos(cos) * (180.0 / torch.pi)


def main():
    device = torch.device("cuda")
    data_dir = REPO / "datasets" / "uprightnet15"
    rotation = np.load(data_dir / "test_rotation.npy").astype(np.float32)
    rotm = np.load(data_dir / "rotm_test.npy").astype(np.float32)
    N = rotation.shape[0]
    print(f"Loaded {N} test samples")

    # The ground-truth upright direction in the rotated frame is R @ [0,1,0] = R[:,1]
    gt_up = rotm[:, :, 1]  # (N, 3)

    # Build model and load backbone
    model = LieUprightNet(num_iters=3).to(device)
    model.load_backbone_from_ckpt(
        str(REPO.parent / "uprightnet-reference" / "model" / "model.pth")
    )
    model.eval()

    BS = 32
    all_errors = np.zeros(N, dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, N, BS)):
            end = min(N, start + BS)
            P = torch.from_numpy(rotation[start:end]).to(device)
            gt = torch.from_numpy(gt_up[start:end]).to(device)
            out = model(P)
            up = out["up"]
            err = angular_error_deg(up, gt)
            all_errors[start:end] = err.cpu().numpy()

    print()
    print("Zero-shot LieUprightNet (PCA init + untrained so3_head):")
    print(f"  mean:   {all_errors.mean():.2f}°")
    print(f"  median: {np.median(all_errors):.2f}°")
    print(f"  acc@10: {(all_errors < 10).mean() * 100:.2f}%")
    print(f"  acc@30: {(all_errors < 30).mean() * 100:.2f}%")
    print(f"  flip(>90°): {(all_errors > 90).mean() * 100:.2f}%")

    out_path = REPO / "logs" / "lie_zeroshot.npz"
    np.savez(out_path, error_deg=all_errors)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

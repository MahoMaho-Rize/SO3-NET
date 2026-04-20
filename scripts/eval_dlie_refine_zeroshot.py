"""Zero-shot evaluation of DifferentiableLieUprightRefineNet with trunk
loaded from the reference (Pang 2022) checkpoint and refine head randomly
initialised (near-zero output by design).

Expectation:
  - [A] up_0 (coarse init): ~82% acc@10, same as DifferentiableLieUprightNet
  - [B] up (after 3 refine iters): slightly worse than [A] because the
        untrained refine head injects small noise. Magnitude ≲ 5°, so
        acc@10 should drop a few percent. After training it should exceed [A].
"""

import sys
from pathlib import Path
import math

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from network_lie import DifferentiableLieUprightRefineNet


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

    model = DifferentiableLieUprightRefineNet(num_iters=3).to(device)
    model.load_backbone_from_ckpt(
        str(REPO.parent / "uprightnet-reference" / "model" / "model.pth")
    )
    model.eval()
    print(f"Schedule: {[f'{math.degrees(a):.0f}°' for a in model.max_angle_schedule]}")

    BS = 32
    errs_init = np.zeros(N, dtype=np.float32)
    errs_final = np.zeros(N, dtype=np.float32)
    deltas_all = [[] for _ in range(model.num_iters)]
    with torch.no_grad():
        for s in tqdm(range(0, N, BS)):
            e = min(N, s + BS)
            P = torch.from_numpy(rotation[s:e]).to(device)
            gt = torch.from_numpy(gt_up[s:e]).to(device)
            out = model(P)
            errs_init[s:e] = angular_error_deg(out["up_0"], gt).cpu().numpy()
            errs_final[s:e] = angular_error_deg(out["up"], gt).cpu().numpy()
            for t, dw in enumerate(out["delta_omegas"]):
                deltas_all[t].append(dw.norm(dim=-1).cpu().numpy())

    def _summary(name, err):
        print(f"{name}:")
        print(f"  mean={err.mean():.2f}°  median={np.median(err):.2f}°")
        print(f"  acc@5={(err<5).mean()*100:.2f}%  acc@10={(err<10).mean()*100:.2f}%  "
              f"acc@30={(err<30).mean()*100:.2f}%")
        print(f"  flip(>90°)={(err>90).mean()*100:.2f}%  "
              f"near180(>170°)={(err>170).mean()*100:.2f}%")

    print()
    _summary("[A] up_0 (coarse init only, differentiable)", errs_init)
    print()
    _summary("[B] up after 3 refine iters (untrained refine head)", errs_final)

    print()
    print("Delta magnitude (degrees) per iteration, mean over all samples:")
    for t, dl in enumerate(deltas_all):
        all_d = np.concatenate(dl)
        max_a = math.degrees(model.max_angle_schedule[t])
        print(f"  iter {t+1} (cap={max_a:5.1f}°): mean={math.degrees(all_d.mean()):6.3f}°  "
              f"max={math.degrees(all_d.max()):6.3f}°")

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

    out_path = REPO / "logs" / "dlie_refine_zeroshot.npz"
    np.savez(out_path,
             errs_init=errs_init, errs_final=errs_final, label=labels)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

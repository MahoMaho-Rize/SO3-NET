"""Standalone Exp 4: evaluate the latest L=1 equivariant ckpt on a subset
of UprightNet15 test (same subset as diag_ceilings.py for fair comparison)."""

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data, Batch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from network_equivariant import build_equivariant_model


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


def main():
    device = torch.device("cuda")
    data_dir = REPO / "datasets" / "uprightnet15"
    rotation = np.load(data_dir / "test_rotation.npy").astype(np.float32)
    rotm = np.load(data_dir / "rotm_test.npy").astype(np.float32)
    labels = np.load(data_dir / "labels_test.npy").astype(np.int32).squeeze()
    total = rotation.shape[0]
    np.random.seed(0)
    idx = np.random.choice(total, size=1500, replace=False)
    rotation, rotm, labels = rotation[idx], rotm[idx], labels[idx]
    N = 1500

    ckpt = str(REPO / "models" / "equivariant_best_20260416-0214.pth")
    model = build_equivariant_model(Opts()).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded {ckpt}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    BS = 16
    errs = []
    with torch.no_grad():
        for s in tqdm(range(0, N, BS)):
            e = min(N, s + BS)
            batch_list = [Data(pos=torch.from_numpy(rotation[i]).float())
                          for i in range(s, e)]
            batch = Batch.from_data_list(batch_list).to(device)
            out = model(batch)
            mu = out["direction_mu"]
            gt = torch.from_numpy(rotm[s:e, :, 1]).to(device).float()
            # Evaluate BOTH antipodal (how the model was trained) and direct
            # (true angular error without absorbing flips)
            cos_direct = (mu * gt).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
            cos_anti = cos_direct.abs().clamp(max=1 - 1e-7)
            err_direct = torch.acos(cos_direct) * (180.0 / np.pi)
            err_anti = torch.acos(cos_anti) * (180.0 / np.pi)
            errs.append(torch.stack([err_direct.cpu(), err_anti.cpu()], dim=-1))
    errs = torch.cat(errs, dim=0).numpy()
    err_d, err_a = errs[:, 0], errs[:, 1]

    print()
    print("Direct (no antipodal collapse):")
    print(f"  mean={err_d.mean():.2f}°  median={np.median(err_d):.2f}°  "
          f"acc@10={(err_d<10).mean()*100:.2f}%  flip={(err_d>90).mean()*100:.2f}%")
    print("Antipodal (how it was trained):")
    print(f"  mean={err_a.mean():.2f}°  median={np.median(err_a):.2f}°  "
          f"acc@10={(err_a<10).mean()*100:.2f}%")

    # Per-category (antipodal mostly — that's what the training objective was)
    print()
    CATS = ["bed", "bench", "bottle", "bowl", "car", "chair", "cone", "cup",
            "lamp", "monitor", "sofa", "stool", "table", "toilet", "vase"]
    print("Per-category acc@10 (antipodal):")
    for lab in sorted(np.unique(labels)):
        m = labels == lab
        print(f"  {CATS[lab]:>8}  N={m.sum():>4}  mean_dir={err_d[m].mean():6.2f}  "
              f"mean_ant={err_a[m].mean():6.2f}  "
              f"acc@10_ant={(err_a[m]<10).mean()*100:5.1f}%  "
              f"flip={(err_d[m]>90).mean()*100:5.1f}%")


if __name__ == "__main__":
    main()

"""
Verify E(3)-equivariance of the EquivariantUprightNet.

Tests:
    1. Direction equivariance: f(R*P) ≈ R * f(P)
    2. Kappa invariance:       kappa(R*P) ≈ kappa(P)
    3. Support invariance:     support(R*P) ≈ support(P)

Run:
    python scripts/verify_equivariance.py [--lmax 2] [--equi_layers 4] ...
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch_geometric.data import Data, Batch

from e3nn import o3
from network_equivariant import build_equivariant_model
from config import opts


def random_rotation_matrix():
    """Generate a random rotation matrix from SO(3)."""
    R = o3.rand_matrix()  # (3, 3)
    return R


def create_dummy_point_cloud(num_points=512, batch_size=2):
    """Create a dummy batch of point clouds for testing."""
    data_list = []
    for _ in range(batch_size):
        pos = torch.randn(num_points, 3) * 0.5
        data_list.append(Data(pos=pos))
    batch = Batch.from_data_list(data_list)
    return batch


def test_direction_equivariance(model, batch, R, atol=1e-4):
    """
    Test: f_direction(R * P) ≈ R * f_direction(P)

    The l=1 direction output should rotate with the input.
    """
    model.eval()
    with torch.no_grad():
        # f(P)
        out_original = model(batch)
        mu_original = out_original["direction_mu"]  # (B, 3)

        # Rotate all positions: R * P
        batch_rotated = batch.clone()
        batch_rotated.pos = batch.pos @ R.T  # (N, 3) @ (3, 3)

        # f(R * P)
        out_rotated = model(batch_rotated)
        mu_rotated = out_rotated["direction_mu"]  # (B, 3)

        # R * f(P)
        mu_expected = mu_original @ R.T  # (B, 3)

        # Check: f(R*P) ≈ R * f(P)  or  f(R*P) ≈ -R * f(P)  (antipodal)
        error_pos = (mu_rotated - mu_expected).norm(dim=-1)  # (B,)
        error_neg = (mu_rotated + mu_expected).norm(dim=-1)  # (B,)
        error = torch.min(error_pos, error_neg)

        max_error = error.max().item()
        mean_error = error.mean().item()

    return max_error, mean_error


def test_kappa_invariance(model, batch, R, atol=1e-4):
    """
    Test: kappa(R * P) ≈ kappa(P)

    The l=0 concentration parameter should be rotation-invariant.
    """
    model.eval()
    with torch.no_grad():
        out_original = model(batch)
        kappa_original = out_original["direction_kappa"]  # (B, 1)

        batch_rotated = batch.clone()
        batch_rotated.pos = batch.pos @ R.T

        out_rotated = model(batch_rotated)
        kappa_rotated = out_rotated["direction_kappa"]  # (B, 1)

        error = (kappa_rotated - kappa_original).abs()
        max_error = error.max().item()
        mean_error = error.mean().item()

    return max_error, mean_error


def test_support_invariance(model, batch, R, atol=1e-4):
    """
    Test: support(R * P) ≈ support(P)

    Per-point support predictions should be permutation-consistent
    (same point gets same prediction regardless of rotation).
    """
    model.eval()
    with torch.no_grad():
        out_original = model(batch)
        support_original = out_original["support_pred"]  # (N,)

        batch_rotated = batch.clone()
        batch_rotated.pos = batch.pos @ R.T

        out_rotated = model(batch_rotated)
        support_rotated = out_rotated["support_pred"]  # (N,)

        error = (support_rotated - support_original).abs()
        max_error = error.max().item()
        mean_error = error.mean().item()

    return max_error, mean_error


def main():
    print("=" * 60)
    print("Equivariance Verification for EquivariantUprightNet")
    print("=" * 60)

    # Force CPU for reproducible testing
    device = torch.device("cpu")

    # Build model (random weights)
    model = build_equivariant_model(opts).to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print("Model parameters: %d" % num_params)
    print("Hidden irreps: %s" % opts.irreps_hidden)
    print("Layers: %d, lmax: %d" % (opts.equi_layers, opts.lmax))
    print()

    # Create test data
    num_tests = 10
    num_points = 256
    batch_size = 2

    dir_errors = []
    kappa_errors = []
    support_errors = []

    for i in range(num_tests):
        R = random_rotation_matrix().to(device)
        batch = create_dummy_point_cloud(num_points, batch_size).to(device)

        dir_max, dir_mean = test_direction_equivariance(model, batch, R)
        kap_max, kap_mean = test_kappa_invariance(model, batch, R)
        sup_max, sup_mean = test_support_invariance(model, batch, R)

        dir_errors.append(dir_max)
        kappa_errors.append(kap_max)
        support_errors.append(sup_max)

        print(
            "Test %2d: dir_err=%.2e, kappa_err=%.2e, support_err=%.2e"
            % (i + 1, dir_max, kap_max, sup_max)
        )

    print()
    print("-" * 60)
    print("Summary over %d random rotations:" % num_tests)
    print(
        "  Direction equivariance error:  max=%.2e, mean=%.2e"
        % (max(dir_errors), np.mean(dir_errors))
    )
    print(
        "  Kappa invariance error:        max=%.2e, mean=%.2e"
        % (max(kappa_errors), np.mean(kappa_errors))
    )
    print(
        "  Support invariance error:      max=%.2e, mean=%.2e"
        % (max(support_errors), np.mean(support_errors))
    )

    # Pass/fail
    threshold = 1e-3
    all_pass = (
        max(dir_errors) < threshold
        and max(kappa_errors) < threshold
        and max(support_errors) < threshold
    )

    print()
    if all_pass:
        print("PASSED: All equivariance errors below %.0e" % threshold)
    else:
        print("WARNING: Some errors exceed %.0e threshold." % threshold)
        print("  This may be due to numerical precision in radius_graph")
        print("  (points near the radius boundary may switch neighbors).")
        print("  Errors < 1e-2 are generally acceptable for practical use.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

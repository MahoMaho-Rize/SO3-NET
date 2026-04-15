"""
Geometric utility functions for equivariant upright orientation estimation.

Includes:
  - Angular error computation on S^2
  - GT upright direction extraction from rotation matrices
  - von Mises-Fisher distribution utilities
  - Differentiable weighted SVD for plane normal estimation (ablation)
"""

import torch
import torch.nn.functional as F
import math


# ============================================================
# Direction / Angular Error
# ============================================================


def angular_error_deg(pred, gt, antipodal=True):
    """
    Compute angular error in degrees between predicted and GT direction vectors.

    Args:
        pred: (B, 3) predicted direction vectors (need not be unit)
        gt:   (B, 3) ground truth direction vectors (need not be unit)
        antipodal: if True, treat n and -n as equivalent (|cos| instead of cos)

    Returns:
        error: (B,) angular error in degrees
    """
    pred = F.normalize(pred, dim=-1)
    gt = F.normalize(gt, dim=-1)
    cos_sim = (pred * gt).sum(dim=-1)  # (B,)
    if antipodal:
        cos_sim = cos_sim.abs()
    cos_sim = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_sim) * (180.0 / math.pi)


def rotation_matrix_to_upright(rotm):
    """
    Extract the ground truth upright direction from rotation matrix.

    The canonical upright direction is [0, 1, 0] (y-axis up).
    After rotation R, the upright direction becomes R @ [0, 1, 0] = R[:, 1].

    Args:
        rotm: (..., 3, 3) rotation matrix

    Returns:
        upright: (..., 3) upright direction vector (unit)
    """
    return rotm[..., :, 1]  # second column of rotation matrix


# ============================================================
# von Mises-Fisher Distribution Utilities
# ============================================================


def vmf_log_normalizer(kappa, dim=3):
    """
    Compute log normalizing constant of the von Mises-Fisher distribution.

    For dim=3 (distribution on S^2):
        C_3(kappa) = kappa / (4 * pi * sinh(kappa))
        log C_3(kappa) = log(kappa) - log(4*pi) - log(sinh(kappa))

    Uses numerically stable computation for large kappa:
        log(sinh(kappa)) = kappa + log(1 - exp(-2*kappa)) - log(2)

    Args:
        kappa: (...,) concentration parameter (positive)
        dim: dimension of the ambient space (default 3 for S^2)

    Returns:
        log_c: (...,) log normalizing constant
    """
    if dim != 3:
        raise NotImplementedError("Only dim=3 (S^2) is currently supported")

    # Numerically stable log(sinh(kappa))
    # For large kappa, sinh(kappa) ≈ exp(kappa)/2, so log(sinh) ≈ kappa - log(2)
    log_sinh_kappa = kappa + torch.log1p(-torch.exp(-2.0 * kappa)) - math.log(2.0)

    log_c = torch.log(kappa + 1e-10) - math.log(4.0 * math.pi) - log_sinh_kappa
    return log_c


def vmf_log_prob(x, mu, kappa):
    """
    Compute log probability under von Mises-Fisher distribution.

    log p(x | mu, kappa) = kappa * mu^T x + log C_3(kappa)

    Args:
        x:     (..., 3) unit vectors (samples)
        mu:    (..., 3) unit mean direction
        kappa: (...,) or (..., 1) concentration parameter

    Returns:
        log_p: (...,) log probability
    """
    if kappa.dim() > x.dim() - 1:
        kappa = kappa.squeeze(-1)
    cos_sim = (x * mu).sum(dim=-1)
    log_p = kappa * cos_sim + vmf_log_normalizer(kappa)
    return log_p


# ============================================================
# Differentiable Weighted SVD (for ablation / comparison)
# ============================================================


def weighted_plane_normal(points, weights, eps=1e-6):
    """
    Compute plane normal from weighted point cloud using differentiable SVD.

    This is a differentiable alternative to RANSAC plane fitting.
    The normal is the eigenvector of the weighted covariance matrix
    corresponding to the smallest eigenvalue.

    Args:
        points:  (B, N, 3) point coordinates
        weights: (B, N) soft weights in (0, 1) from network sigmoid output
        eps:     regularization for numerical stability

    Returns:
        normal:    (B, 3) plane normal (unit vector)
        eigenvals: (B, 3) eigenvalues (ascending order)
    """
    # Weighted centroid
    w = weights.unsqueeze(-1)  # (B, N, 1)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=eps)  # (B, 1, 1)
    centroid = (w * points).sum(dim=1, keepdim=True) / w_sum  # (B, 1, 3)

    # Centered points
    centered = points - centroid  # (B, N, 3)

    # Weighted covariance matrix
    # C = (1/W) * sum_i w_i * (p_i - mu)(p_i - mu)^T
    weighted_centered = centered * w.sqrt()  # (B, N, 3)
    cov = torch.bmm(weighted_centered.transpose(1, 2), weighted_centered)  # (B, 3, 3)
    cov = cov / w_sum.squeeze(-1).squeeze(-1).unsqueeze(-1).unsqueeze(
        -1
    )  # (B, 1, 1) for broadcast to (B, 3, 3)

    # Regularization for numerical stability near degenerate cases
    cov = cov + eps * torch.eye(3, device=cov.device).unsqueeze(0)

    # Eigendecomposition (eigenvalues in ascending order)
    eigenvals, eigenvecs = torch.linalg.eigh(cov)  # (B, 3), (B, 3, 3)

    # Normal = eigenvector of smallest eigenvalue
    normal = eigenvecs[:, :, 0]  # (B, 3)

    # Orient normal towards mass center (away from support plane)
    mass_center = points.mean(dim=1)  # (B, 3)
    support_center = centroid.squeeze(1)  # (B, 3)
    direction = mass_center - support_center  # (B, 3)
    sign = torch.sign((normal * direction).sum(dim=-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    normal = normal * sign

    return F.normalize(normal, dim=-1), eigenvals

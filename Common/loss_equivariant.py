"""
Loss functions for equivariant upright orientation estimation.

Includes:
    - Geodesic loss on S^2 (deterministic direction regression)
    - von Mises-Fisher negative log-likelihood (probabilistic direction)
    - Auxiliary support point BCE loss
    - Combined total loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Common.geometric_utils import vmf_log_normalizer


# ============================================================
# Geodesic Loss on S^2
# ============================================================


def geodesic_loss(pred, gt, antipodal=True, reduction="mean"):
    """
    Geodesic (arc-length) loss on the unit sphere S^2.

    L = arccos(cos_sim)   where cos_sim = pred . gt  (or |pred . gt| if antipodal)

    This directly measures the angle between predicted and GT directions,
    which is the most natural loss for directional regression on S^2.

    Args:
        pred:      (B, 3) predicted direction (will be normalized)
        gt:        (B, 3) ground truth direction (will be normalized)
        antipodal: if True, n and -n are treated as equivalent
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: scalar (or (B,) if reduction='none')
    """
    pred = F.normalize(pred, dim=-1)
    gt = F.normalize(gt, dim=-1)

    cos_sim = (pred * gt).sum(dim=-1)  # (B,)
    if antipodal:
        cos_sim = cos_sim.abs()

    # Clamp for numerical stability (arccos gradient → ∞ at ±1)
    cos_sim = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    loss = torch.acos(cos_sim)  # (B,) in radians

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


# ============================================================
# von Mises-Fisher Negative Log-Likelihood
# ============================================================


def vmf_nll_loss(mu, kappa, gt, antipodal=True, reduction="mean"):
    """
    Negative log-likelihood under the von Mises-Fisher distribution on S^2.

    vMF pdf: p(x | mu, kappa) = C_3(kappa) * exp(kappa * mu^T x)
    NLL:     -log p = -kappa * mu^T x - log C_3(kappa)

    For antipodal symmetry (direction ≡ -direction), we use:
        NLL = -kappa * |mu^T x| - log C_3(kappa)

    The loss naturally balances accuracy (kappa * cos) and uncertainty
    (log normalizer penalizes overconfidence when predictions are wrong).

    Args:
        mu:        (B, 3) predicted mean direction (unit vector)
        kappa:     (B, 1) predicted concentration parameter (> 0)
        gt:        (B, 3) ground truth direction (unit vector)
        antipodal: if True, treat n and -n as equivalent
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: scalar (or (B,) if reduction='none')
    """
    mu = F.normalize(mu, dim=-1)
    gt = F.normalize(gt, dim=-1)
    kappa = kappa.squeeze(-1)  # (B,)

    cos_sim = (mu * gt).sum(dim=-1)  # (B,)
    if antipodal:
        cos_sim = cos_sim.abs()

    # NLL = -kappa * cos_sim - log C_3(kappa)
    # Note: log C_3(kappa) is the log normalizing constant
    # Minimizing NLL means maximizing kappa*cos - log_C
    log_c = vmf_log_normalizer(kappa, dim=3)
    nll = -kappa * cos_sim - log_c  # (B,)

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    return nll


# ============================================================
# Auxiliary Support Point BCE Loss
# ============================================================


def auxiliary_support_bce(support_pred, support_gt, reduction="mean"):
    """
    Binary cross-entropy loss for per-point support classification.

    This is an auxiliary task that provides:
    1. Interpretability (which points are predicted as support)
    2. Regularization (additional supervision signal)
    3. Fair comparison with original UprightNet

    Args:
        support_pred: (N,) predicted support probability in [0, 1]
        support_gt:   (N,) ground truth support labels (0 or 1)
        reduction:    'mean', 'sum', or 'none'

    Returns:
        loss: scalar
    """
    return F.binary_cross_entropy(
        support_pred,
        support_gt.float(),
        reduction=reduction,
    )


# ============================================================
# Combined Loss
# ============================================================


class EquivariantLoss(nn.Module):
    """
    Combined loss for equivariant upright orientation estimation.

    L_total = L_direction + beta * L_support

    where L_direction is either geodesic or vMF NLL loss,
    and L_support is auxiliary per-point BCE.

    Args:
        loss_type:  'geodesic' or 'vmf'
        beta:       weight for auxiliary support loss
        antipodal:  whether to use antipodal symmetry
    """

    def __init__(self, loss_type="vmf", beta=0.1, antipodal=True):
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta
        self.antipodal = antipodal

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict from EquivariantUprightNet.forward()
                'direction_mu':    (B, 3)
                'direction_kappa': (B, 1)
                'support_pred':    (N,)
            targets: dict
                'y_direction':     (B, 3) GT upright direction
                'y_support':       (N,)   GT support labels

        Returns:
            loss_dict: dict with 'total', 'direction', 'support', and optional 'kappa_mean'
        """
        mu = outputs["direction_mu"]
        kappa = outputs["direction_kappa"]
        support_pred = outputs["support_pred"]

        gt_direction = targets["y_direction"]
        gt_support = targets["y_support"]

        # Direction loss
        if self.loss_type == "vmf":
            loss_direction = vmf_nll_loss(
                mu,
                kappa,
                gt_direction,
                antipodal=self.antipodal,
            )
        elif self.loss_type == "geodesic":
            loss_direction = geodesic_loss(
                mu,
                gt_direction,
                antipodal=self.antipodal,
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Auxiliary support loss
        loss_support = auxiliary_support_bce(support_pred, gt_support)

        # Total
        loss_total = loss_direction + self.beta * loss_support

        return {
            "total": loss_total,
            "direction": loss_direction,
            "support": loss_support,
            "kappa_mean": kappa.mean().detach(),
        }

"""
Loss functions for SHS-style signed direction prediction.

Key difference from loss_equivariant.py: we use SIGNED geodesic loss
(no antipodal folding).  The network must learn the correct polarity
from global context, not be given a free pass via |cos|.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def signed_geodesic_loss(pred, gt, reduction="mean"):
    """Geodesic loss on S² WITHOUT antipodal identification.

    The network must predict the correct sign of the upright direction.
    This is the core design choice borrowed from SHS-Net: learn signed
    orientation end-to-end rather than estimating unsigned + flipping.

    Args:
        pred: (B, 3) predicted direction (will be normalized)
        gt:   (B, 3) ground truth direction (will be normalized)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: scalar (or (B,) if reduction='none')
    """
    pred = F.normalize(pred, dim=-1)
    gt = F.normalize(gt, dim=-1)
    cos_sim = (pred * gt).sum(dim=-1)
    cos_sim = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    loss = torch.acos(cos_sim)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def auxiliary_support_bce(support_pred, support_gt, reduction="mean"):
    """BCE loss for per-point support classification (auxiliary task)."""
    return F.binary_cross_entropy(
        support_pred,
        support_gt.float(),
        reduction=reduction,
    )


class SHSLoss(nn.Module):
    """Combined loss for SHS-style signed direction prediction.

    L_total = L_signed_geo + beta * L_support

    Args:
        beta: weight for auxiliary support BCE loss
    """

    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict from SHSUprightNet.forward()
            targets: dict with 'y_direction' (B, 3) and 'y_support' (N,)

        Returns:
            loss_dict with 'total', 'direction', 'support'
        """
        pred_up = outputs["up"]
        support_pred = outputs["support_logits"]

        gt_direction = targets["y_direction"]
        gt_support = targets["y_support"]

        loss_direction = signed_geodesic_loss(pred_up, gt_direction)
        loss_support = auxiliary_support_bce(support_pred, gt_support)
        loss_total = loss_direction + self.beta * loss_support

        return {
            "total": loss_total,
            "direction": loss_direction,
            "support": loss_support,
        }

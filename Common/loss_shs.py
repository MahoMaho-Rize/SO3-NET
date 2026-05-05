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


def antipodal_geodesic_loss(axis, gt, reduction="mean"):
    """Geodesic loss on S² WITH antipodal identification.

    Measures the angle between the line spanned by `axis` and the line
    spanned by `gt` — polarity is ignored by taking |cos|.  Used to train
    the axis branch of DecomposedDirectionHead.
    """
    axis = F.normalize(axis, dim=-1)
    gt = F.normalize(gt, dim=-1)
    cos_sim = (axis * gt).sum(dim=-1).abs()
    cos_sim = cos_sim.clamp(1e-7, 1.0 - 1e-7)
    loss = torch.acos(cos_sim)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def polarity_bce_loss(sign_logit, axis, gt, reduction="mean"):
    """Binary cross-entropy for polarity classification.

    The sign GT is derived at runtime from the predicted axis (detached) and
    the GT direction: +1 when axis already points roughly toward gt, 0 when
    it points away.  Detaching the axis is essential — without it, the sign
    loss would back-propagate into the axis branch, distorting the axis to
    make polarity classification easier.

    Args:
        sign_logit: (B,) raw logit from DecomposedDirectionHead
        axis:       (B, 3) predicted unsigned axis
        gt:         (B, 3) GT direction
    """
    with torch.no_grad():
        axis_d = F.normalize(axis.detach(), dim=-1)
        gt_d = F.normalize(gt, dim=-1)
        sign_gt = ((axis_d * gt_d).sum(dim=-1) > 0).float()
    return F.binary_cross_entropy_with_logits(
        sign_logit, sign_gt, reduction=reduction,
    )


def stability_loss(up, points, support_prob, reduction="mean", eps=1e-6):
    """Physical stability prior.

    An object standing upright has its mass center *above* its support
    region along the predicted up direction.  Let

        c_mass = mean(points)
        c_sup  = weighted mean of points by support_prob
        gap    = <up, c_mass - c_sup>

    A positive `gap` means the mass center is above the support (stable);
    a negative `gap` means the predicted up inverted the object (the heavy
    base ends up "above" the shade — the classic lamp flip).  We penalise
    only the unstable case via ReLU(-gap).

    Self-supervised: uses no additional annotation beyond support_prob,
    which already supervises the trunk via L_sup.
    """
    # points: (B, N, 3), support_prob: (B, N), up: (B, 3)
    c_mass = points.mean(dim=1)  # (B, 3)
    w = support_prob.unsqueeze(-1)  # (B, N, 1)
    c_sup = (w * points).sum(dim=1) / (w.sum(dim=1) + eps)  # (B, 3)
    up = F.normalize(up, dim=-1)
    gap = (up * (c_mass - c_sup)).sum(dim=-1)  # (B,)
    loss = F.relu(-gap)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


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


class DecomposedLoss(nn.Module):
    """Decomposed loss for DecomposedDirectionHead.

        L = λ_axis * L_axis
          + λ_sign * L_sign
          + λ_sup  * L_sup
          + λ_stab * L_stab

    Setting λ_stab=0 disables the physical stability prior (useful for
    ablation A3).  Setting λ_sign=0 reduces to axis-only training (A1).
    """

    def __init__(
        self,
        lambda_axis=1.0,
        lambda_sign=0.5,
        lambda_sup=0.1,
        lambda_stab=0.2,
    ):
        super().__init__()
        self.lambda_axis = lambda_axis
        self.lambda_sign = lambda_sign
        self.lambda_sup = lambda_sup
        self.lambda_stab = lambda_stab

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict from SHSUprightNet.forward() with head_type='decomposed'.
                Must contain 'up', 'axis', 'sign_logit', 'support_logits'.
            targets: dict with:
                'y_direction' (B, 3) — GT upright
                'y_support'   (B, N) — per-point support labels
                'points'      (B, N, 3) — input point cloud (for L_stab)
        """
        if "axis" not in outputs or "sign_logit" not in outputs:
            raise ValueError(
                "DecomposedLoss requires outputs from head_type='decomposed'; "
                "got keys: " + ", ".join(outputs.keys())
            )

        axis = outputs["axis"]
        sign_logit = outputs["sign_logit"]
        up = outputs["up"]
        support_pred = outputs["support_logits"]

        gt_direction = targets["y_direction"]
        gt_support = targets["y_support"]
        points = targets["points"]

        l_axis = antipodal_geodesic_loss(axis, gt_direction)
        l_sign = polarity_bce_loss(sign_logit, axis, gt_direction)
        l_sup = auxiliary_support_bce(support_pred, gt_support)

        if self.lambda_stab > 0:
            l_stab = stability_loss(up, points, support_pred.detach())
        else:
            l_stab = torch.zeros((), device=up.device)

        loss_total = (
            self.lambda_axis * l_axis
            + self.lambda_sign * l_sign
            + self.lambda_sup * l_sup
            + self.lambda_stab * l_stab
        )

        return {
            "total": loss_total,
            "axis": l_axis,
            "sign": l_sign,
            "support": l_sup,
            "stab": l_stab,
        }

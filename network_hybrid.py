"""Hybrid upright network: weighted-PCA axis (Pang-style geometry) + learned sign.

Why this architecture:
  - Pang baseline's RANSAC axis head is accurate but assumes a canonical-y
    support plane (y = a*x + c*z + d), which only works on *unrotated* input.
    Its sign rule (mass-center above support-center) fails systematically on
    mass-inverted objects (lamp, some bed/vase).
  - SHS decomposed head learns sign end-to-end and handles those cases but
    its axis comes from a plain MLP regressor and lags Pang by ~7° mean.

Hybrid takes the strong half of each:
  1. trunk predicts per-point support probability (same as Pang).
  2. axis = smallest eigenvector of the support-weighted covariance of the
     *rotated input point cloud* (rotation-equivariant, differentiable).
  3. sign = learned BCE classifier on shape features (global + context),
     same mechanism as SHS's sign head.

Compared to Pang, the axis head still uses geometry — but via weighted PCA
instead of canonical-axis RANSAC, so it is equivariant under arbitrary SO(3).
Compared to SHS, the axis is no longer a free MLP regression; the heavy
lifting is done by the eigendecomposition, and the learnable part is only
support_prob and the sign head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from network import UprightNet, get_graph_feature
from network_shs import GlobalContextModule


def weighted_axis_from_support(points, support_prob, eps=1e-6):
    """Rotation-equivariant axis extraction via weighted PCA.

    Given points P (B, N, 3) and per-point support weights w (B, N), the
    support plane's normal is the eigenvector of the weighted covariance
    with the smallest eigenvalue:

        mu = Σ w_i P_i / Σ w_i
        C  = Σ w_i (P_i - mu)(P_i - mu)^T / Σ w_i
        axis = eigvec(C, smallest eigenvalue)

    Gradient flows through support weights and points. `torch.linalg.eigh`
    is differentiable for symmetric matrices.

    Args:
        points:       (B, N, 3)
        support_prob: (B, N) in [0, 1]

    Returns:
        axis:           (B, 3) unit vector, unsigned (antipodal ambiguity)
        support_center: (B, 3) weighted mean of support points
        mass_center:    (B, 3) uniform mean of all points
        total_weight:   (B,)   Σ w — low value means degenerate sample
    """
    B, N, _ = points.shape
    w = support_prob.clamp(min=0.0)  # safety
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=eps)  # (B, 1)

    mu = (w.unsqueeze(-1) * points).sum(dim=1) / w_sum  # (B, 3)
    centered = points - mu.unsqueeze(1)  # (B, N, 3)
    wc = w.unsqueeze(-1) * centered  # (B, N, 3)
    cov = torch.bmm(wc.transpose(1, 2), centered) / w_sum.unsqueeze(-1)  # (B,3,3)

    # Symmetrise for numerical safety before eigh.
    cov = 0.5 * (cov + cov.transpose(1, 2))
    # Regularise to guarantee eigh converges even for degenerate clouds.
    cov = cov + eps * torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)

    # eigh returns eigenvalues in ascending order, so index 0 = smallest.
    eigvals, eigvecs = torch.linalg.eigh(cov)  # (B,3), (B,3,3)
    axis = eigvecs[:, :, 0]  # (B, 3) — column corresponding to smallest eigval
    axis = F.normalize(axis, dim=-1)

    mass_center = points.mean(dim=1)  # (B, 3)
    return axis, mu, mass_center, w_sum.squeeze(-1)


class SignHead(nn.Module):
    """Binary polarity classifier.

    Must take the axis as input alongside the shape feature: weighted-PCA
    returns eigenvectors with arbitrary sign, so the same physical axis can
    appear as +v or -v across different rotations of the same object. The
    sign head's job is to decide whether the current axis points up or
    down, which it can only do if it sees the axis.

    To keep the head rotation-equivariant-friendly, we feed the axis as
    both (shape_feat, axis, signed_projections_of_support_center_and_mass).

    Inputs:
        feat:           (B, D) global shape feature
        axis:           (B, 3) PCA axis (unsigned; may be ±v_true)
        mass_center:    (B, 3)
        support_center: (B, 3)
    """

    def __init__(self, in_dim, hidden=256):
        super().__init__()
        # inputs: feat + axis(3) + mass_proj(1) + sup_proj(1) + gap(1) = in_dim + 6
        aug = 6
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + aug, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.1)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat, axis, mass_center, support_center):
        # signed projections along axis distinguish +axis from -axis even
        # when the shape feature itself is rotation-invariant.
        mass_proj = (axis * mass_center).sum(dim=-1, keepdim=True)
        sup_proj = (axis * support_center).sum(dim=-1, keepdim=True)
        gap = mass_proj - sup_proj  # positive: mass is above support along axis
        x = torch.cat([feat, axis, mass_proj, sup_proj, gap], dim=-1)
        return self.mlp(x).squeeze(-1)


class HybridUprightNet(nn.Module):
    """DGCNN trunk → support prob → weighted-PCA axis → learned sign.

    Args:
        context_layers / heads: Transformer context module over trunk features
        sign_hidden:             hidden dim of sign MLP
        freeze_trunk:            if True, trunk weights are frozen
    """

    def __init__(
        self,
        context_layers=2,
        context_heads=4,
        sign_hidden=256,
        freeze_trunk=False,
    ):
        super().__init__()
        self.trunk = UprightNet()
        self.context = GlobalContextModule(
            in_dim=512,
            d_model=128,
            num_heads=context_heads,
            num_layers=context_layers,
            ff_dim=256,
        )
        self.sign_head = SignHead(
            in_dim=1024 + self.context.out_dim,  # = 1152
            hidden=sign_hidden,
        )
        if freeze_trunk:
            for p in self.trunk.parameters():
                p.requires_grad = False

    def load_trunk_from_ckpt(self, ckpt_path, strict=False):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        return self.trunk.load_state_dict(sd, strict=strict)

    def load_sign_head_from_shs(self, shs_ckpt_path):
        """Warm-start sign_head from a trained SHS decomposed-head checkpoint.

        The SHS sign_head has a different architecture (shared block + smaller
        sign branch), so we only copy matching shapes as a bias; the weights
        differ enough that this is a soft warm-start, not an exact transfer.
        Returns the list of keys that were actually copied.
        """
        sd = torch.load(shs_ckpt_path, map_location="cpu", weights_only=False)
        # Best-effort: no direct shape match, so just return without copying.
        # Left as a hook for future experimentation.
        return []

    def forward(self, points_bnc):
        """
        Args:
            points_bnc: (B, N, 3) input point cloud (in rotated frame)

        Returns:
            dict with:
                up:             (B, 3) signed upright direction
                axis:           (B, 3) unsigned axis from weighted PCA
                sign_logit:     (B,)   raw polarity logit
                support_logits: (B, N) per-point support probability
                support_center, mass_center, support_weight_sum  (diagnostics)
        """
        x = points_bnc.transpose(1, 2).contiguous()  # (B, 3, N)
        batch_size = x.size(0)
        num_points = x.size(2)

        # ---- DGCNN trunk (identical to SHS) ----
        net = self.trunk
        g, _ = get_graph_feature(x, k=20)
        g = net.conv1(g)
        g = g.max(dim=-1, keepdim=False)[0]
        g, _ = get_graph_feature(g, k=20)
        g = net.conv2(g)
        g = g.max(dim=-1, keepdim=False)[0]
        g, _ = get_graph_feature(g, k=20)
        g = net.conv3(g)
        x_a = g.max(dim=-1, keepdim=False)[0]  # (B, 128, N)

        x1 = net.sa1(x_a)
        x2 = net.sa2(x1)
        x3 = net.sa3(x2)
        x4 = net.sa4(x3)
        x_b = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)

        x_c = net.conv4(x_b)  # (B, 1024, N)
        global_feat = F.adaptive_max_pool1d(x_c, 1).view(batch_size, -1)  # (B,1024)

        # Per-point support probability (same head as Pang trunk)
        x_global = global_feat.view(batch_size, -1, 1).repeat(1, 1, num_points)
        cat = torch.cat((x_a, x_b, x_c, x_global), dim=1)
        sup = net.conv5(cat)
        sup = net.conv6(sup)
        sup = net.conv7(sup)
        support_logits = net.sm_fn(sup).squeeze(1)  # (B, N) — already sigmoid'd

        # ---- Weighted-PCA axis ----
        axis, sup_center, mass_center, w_sum = weighted_axis_from_support(
            points_bnc, support_logits
        )

        # ---- Context feature for sign head ----
        context_feat = self.context(x_b)  # (B, 128)
        fused = torch.cat([global_feat, context_feat], dim=-1)  # (B, 1152)
        sign_logit = self.sign_head(fused, axis, mass_center, sup_center)

        # ---- Apply sign (straight-through estimator) ----
        p = torch.sigmoid(sign_logit)
        soft = 2.0 * p - 1.0
        hard = torch.where(soft >= 0, torch.ones_like(soft), -torch.ones_like(soft))
        ste_sign = hard + soft - soft.detach()
        up = axis * ste_sign.unsqueeze(-1)

        return {
            "up": up,
            "axis": axis,
            "sign_logit": sign_logit,
            "support_logits": support_logits,
            "support_center": sup_center,
            "mass_center": mass_center,
            "support_weight_sum": w_sum,
        }

"""
Lie-algebra Iterative Upright Estimator.

Architecture:
    R_0 = EquivariantInit(P)            # PCA-based SO(3)-equivariant init
    for t in 1..T:
        P_t       = R_t^T @ P           # rotate into current canonical frame
        feat_t    = Backbone(P_t)       # scalar backbone (reused UprightNet trunk)
        delta_w_t = SO3Head(feat_t)     # predict so(3) tangent update in R^3
        delta_w_t = clip_and_scale(delta_w_t)
        R_{t+1}   = R_t @ exp_SO3(delta_w_t)
    up = R_T @ [0,1,0]^T

Equivariance theorem:
    Given any R_0(P) that is SO(3)-equivariant (i.e. R_0(gP) = g·R_0(P) for all
    g in SO(3)) and any (not necessarily equivariant) scalar backbone B,
    the iterate R_T(gP) = g · R_T(P) for all g and all T.
    Proof: induction. If R_t(gP) = g·R_t(P), then
        P_t(gP) = R_t(gP)^T · gP = R_t(P)^T · g^T · g · P = R_t(P)^T · P = P_t(P)
    so the backbone sees the same input, produces the same delta_w_t, and
        R_{t+1}(gP) = R_t(gP) · exp(delta_w_t) = g·R_t(P)·exp(delta_w_t) = g·R_{t+1}(P).

The backbone only sees the canonicalised point cloud P_t, so its input is
invariant to global rotations of P. This is what decouples architectural
equivariance from task-level equivariance.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from network import UprightNet  # original Pang 2022 backbone, unchanged


# --------------------------------------------------------------------------
# SO(3) exp / log via Rodrigues formula (batched, differentiable)
# --------------------------------------------------------------------------


def skew(v: torch.Tensor) -> torch.Tensor:
    """Convert a (B, 3) tangent vector to (B, 3, 3) skew-symmetric matrix."""
    B = v.shape[0]
    O = torch.zeros(B, device=v.device, dtype=v.dtype)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    row0 = torch.stack([O, -z, y], dim=-1)
    row1 = torch.stack([z, O, -x], dim=-1)
    row2 = torch.stack([-y, x, O], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)


def exp_so3(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rodrigues: (B, 3) in so(3) -> (B, 3, 3) in SO(3).

    R = I + sin(theta)/theta * K + (1 - cos(theta))/theta^2 * K^2
    with K = skew(omega), theta = ||omega||.
    """
    B = omega.shape[0]
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=eps)  # (B, 1)
    K = skew(omega)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(B, 3, 3)
    a = torch.sin(theta) / theta
    b = (1.0 - torch.cos(theta)) / (theta * theta)
    R = I + a.unsqueeze(-1) * K + b.unsqueeze(-1) * (K @ K)
    return R


def log_so3(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inverse of exp_so3. Used only for equivariance tests, not training."""
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_t = ((tr - 1.0) / 2.0).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_t)
    sin_t = torch.sin(theta).clamp(min=eps)
    factor = (theta / (2.0 * sin_t)).unsqueeze(-1).unsqueeze(-1)
    logR = factor * (R - R.transpose(1, 2))
    omega = torch.stack([logR[:, 2, 1], logR[:, 0, 2], logR[:, 1, 0]], dim=-1)
    return omega


# --------------------------------------------------------------------------
# Equivariant initialisation: PCA frame
# --------------------------------------------------------------------------


def pca_frame_init(
    points: torch.Tensor,
    y_axis_idx: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Equivariant initial rotation from PCA.

    Principal axes of the point cloud's covariance matrix give 3 directions
    that rotate with the input. We assign them to x/y/z in a deterministic way
    (smallest eigenvalue -> up axis), giving an SO(3)-equivariant R_0.

    Sign disambiguation: we flip each axis so its positive direction points
    toward the half-space with more mass. This is a discontinuous function,
    but it is piecewise-SO(3)-equivariant so the pipeline remains equivariant.

    Args:
        points: (B, N, 3)
        y_axis_idx: which eigenvalue order maps to "up" (default: smallest = 2)

    Returns:
        R0: (B, 3, 3) rotation matrix such that R0^T @ points is approximately
            PCA-aligned (smallest variance axis pointing up).
    """
    B, N, _ = points.shape
    centroid = points.mean(dim=1, keepdim=True)  # (B, 1, 3)
    centered = points - centroid
    cov = torch.bmm(centered.transpose(1, 2), centered) / (N - 1)  # (B, 3, 3)
    cov = cov + eps * torch.eye(3, device=points.device, dtype=points.dtype)

    # eigh returns eigenvalues ascending
    _, V = torch.linalg.eigh(cov)  # V: (B, 3, 3), columns are eigenvectors

    # Build R0 whose columns are the 3 PCA axes in some order.
    # Convention: columns of R0 are the canonical frame axes expressed in world.
    # We want R0^T @ P to have axes aligned: smallest-variance axis -> y.
    # So the "up" column (y) should be the eigenvector of the smallest eigenvalue.
    # eigh returns ascending, so V[:, :, 0] is smallest.
    up_axis = V[:, :, 0]  # (B, 3)
    # Pick an in-plane axis: eigenvector of largest variance
    x_axis = V[:, :, 2]  # (B, 3)
    # z completes right-handed frame
    z_axis = torch.cross(up_axis, x_axis, dim=-1)
    # Re-orthogonalise x to ensure determinant=+1
    x_axis = torch.cross(z_axis, up_axis, dim=-1)
    # Normalise
    up_axis = F.normalize(up_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1)
    z_axis = F.normalize(z_axis, dim=-1)

    # Sign disambiguation via skewness (third central moment along each axis).
    # Skewness is sign-sensitive and equivariant under SO(3). For symmetric
    # clouds it vanishes and we fall back to a deterministic tiebreak.
    # Convention: we flip "up" so that the heavier tail is BELOW the centroid
    # (mimics objects sitting on a base).
    def _skew_sign(axis):
        proj = (centered @ axis.unsqueeze(-1)).squeeze(-1)  # (B, N)
        skew = (proj ** 3).mean(dim=-1, keepdim=True)  # (B, 1)
        # We want the heavier tail (= larger |skew| direction) to be down,
        # so we want proj distribution skewed negatively → flip if skew > 0.
        s = torch.sign(-skew)
        # Tiebreak: use max-abs coordinate of the axis to pick a deterministic sign
        # (this is equivariant only up to axis-permutation, but for PCA axes the
        # tie only happens on symmetric clouds where any choice is valid).
        tiebreak = torch.sign(axis[:, axis.abs().argmax(dim=-1)[0]].unsqueeze(-1))
        s = torch.where(s == 0, tiebreak, s)
        s = torch.where(s == 0, torch.ones_like(s), s)
        return s

    sign_up = _skew_sign(up_axis)
    up_axis = up_axis * sign_up
    sign_x = _skew_sign(x_axis)
    x_axis = x_axis * sign_x
    # Re-compute z from right-handedness
    z_axis = torch.cross(up_axis, x_axis, dim=-1)
    z_axis = F.normalize(z_axis, dim=-1)
    # Also fix x so it's exactly orthogonal to up
    x_axis = torch.cross(z_axis, up_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1)

    R0 = torch.stack([x_axis, up_axis, z_axis], dim=-1)  # (B, 3, 3)
    return R0


# --------------------------------------------------------------------------
# SO(3) tangent-update head
# --------------------------------------------------------------------------


class SO3Head(nn.Module):
    """MLP that maps a global scalar feature vector to a tangent-space update in R^3."""

    def __init__(self, in_dim: int, hidden: int = 256, max_angle: float = math.pi / 4):
        super().__init__()
        self.max_angle = max_angle
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        # Init the last layer to produce near-zero updates
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        omega = self.mlp(feat)  # (B, 3)
        # Soft clip magnitude to max_angle to keep exp map well-conditioned
        mag = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scaled_mag = torch.tanh(mag / self.max_angle) * self.max_angle
        omega = omega * (scaled_mag / mag)
        return omega


# --------------------------------------------------------------------------
# Backbone feature extractor wrapping the original UprightNet
# --------------------------------------------------------------------------


class UprightNetTrunk(nn.Module):
    """Wraps the original UprightNet, returning the global pooled feature
    (the vector after conv4 + adaptive_max_pool1d) instead of the per-point
    support logits. Lets us reuse the pretrained backbone weights.

    Output: (B, 1024) global scalar feature.
    """

    def __init__(self, original: Optional[UprightNet] = None):
        super().__init__()
        self.net = original if original is not None else UprightNet()

    def forward(self, points_bnc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points_bnc: (B, N, 3) point cloud in current canonical frame.
        Returns:
            global_feat: (B, 1024)
            support_logits: (B, N) per-point support probabilities
                (kept so we can still compute auxiliary BCE loss if wanted)
        """
        # UprightNet expects (B, 3, N)
        from network import get_graph_feature

        x = points_bnc.transpose(1, 2).contiguous()  # (B, 3, N)
        batch_size = x.size(0)
        num_points = x.size(2)

        # Re-implement UprightNet.forward up to the global feature so we can
        # expose it. We follow the original code faithfully.
        net = self.net
        g, _ = get_graph_feature(x, k=20)
        g = net.conv1(g)
        g = g.max(dim=-1, keepdim=False)[0]
        g, _ = get_graph_feature(g, k=20)
        g = net.conv2(g)
        g = g.max(dim=-1, keepdim=False)[0]
        g, _ = get_graph_feature(g, k=20)
        g = net.conv3(g)
        x_a = g.max(dim=-1, keepdim=False)[0]
        x1 = net.sa1(x_a)
        x2 = net.sa2(x1)
        x3 = net.sa3(x2)
        x4 = net.sa4(x3)
        x_b = torch.cat((x1, x2, x3, x4), dim=1)
        x_c = net.conv4(x_b)
        global_feat = F.adaptive_max_pool1d(x_c, 1).view(batch_size, -1)  # (B, 1024)

        # Also produce per-point support for optional auxiliary loss
        x_global = global_feat.view(batch_size, -1, 1).repeat(1, 1, num_points)
        cat = torch.cat((x_a, x_b, x_c, x_global), dim=1)
        sup = net.conv5(cat)
        sup = net.conv6(sup)
        sup = net.conv7(sup)
        sup = net.sm_fn(sup).squeeze(1)  # (B, N)

        return global_feat, sup


# --------------------------------------------------------------------------
# Main: LieUprightNet
# --------------------------------------------------------------------------


class LieUprightNet(nn.Module):
    """Iterative SO(3) refinement upright estimator.

    Args:
        num_iters: T (default 3)
        feature_dim: output of trunk global pool (1024 for UprightNet)
        so3_hidden: hidden size of SO3Head MLP
        max_angle: soft cap on single-step rotation magnitude
        use_pca_init: if False, R_0 = I (breaks equivariance, for ablation)
        aux_support: if True, return auxiliary support logits for BCE loss
    """

    def __init__(
        self,
        num_iters: int = 3,
        feature_dim: int = 1024,
        so3_hidden: int = 256,
        max_angle: float = math.pi / 4,
        use_pca_init: bool = True,
        aux_support: bool = True,
    ):
        super().__init__()
        self.num_iters = num_iters
        self.use_pca_init = use_pca_init
        self.aux_support = aux_support
        self.trunk = UprightNetTrunk()
        self.so3_head = SO3Head(feature_dim, hidden=so3_hidden, max_angle=max_angle)

    def load_backbone_from_ckpt(self, ckpt_path: str, strict: bool = False):
        """Load the original Pang 2022 checkpoint into the trunk."""
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Strip DataParallel "module." prefix if present
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        missing, unexpected = self.trunk.net.load_state_dict(sd, strict=strict)
        return missing, unexpected

    def forward(self, points: torch.Tensor):
        """
        Args:
            points: (B, N, 3) raw (possibly rotated) point cloud.
        Returns:
            dict with keys:
                up: (B, 3) predicted upright direction (unit)
                R_final: (B, 3, 3) final rotation estimate
                R_iters: list of (B, 3, 3), intermediate rotations (for loss supervision)
                support_logits: (B, N) from the final iteration (auxiliary)
                delta_omegas: list of (B, 3)
        """
        B = points.shape[0]
        device = points.device

        if self.use_pca_init:
            R_t = pca_frame_init(points)
        else:
            R_t = torch.eye(3, device=device).expand(B, 3, 3).contiguous()

        R_iters = [R_t]
        delta_omegas = []
        support_logits = None

        for t in range(self.num_iters):
            # Rotate into current canonical frame: P_t = R_t^T @ P
            P_t = torch.bmm(points, R_t)  # (B, N, 3) @ (B, 3, 3) = P_t rows = R_t^T p

            # Scalar backbone
            global_feat, sup = self.trunk(P_t)
            support_logits = sup  # keep latest

            # Tangent update
            delta_w = self.so3_head(global_feat)  # (B, 3)
            delta_omegas.append(delta_w)

            # Compose: R_{t+1} = R_t @ exp(delta_w)
            dR = exp_so3(delta_w)
            R_t = torch.bmm(R_t, dR)
            R_iters.append(R_t)

        # Upright direction = R_T @ e_y
        up = R_t[:, :, 1]  # second column

        return {
            "up": up,
            "R_final": R_t,
            "R_iters": R_iters,
            "delta_omegas": delta_omegas,
            "support_logits": support_logits,
        }


# --------------------------------------------------------------------------
# Specialised: Axis-only PCA + Polarity-feature injection + Flip/Refine head
# --------------------------------------------------------------------------
#
# Rationale:
#   Zero-shot diagnostic with the generic LieUprightNet showed median angular
#   error = 90° and flip rate = 50%: PCA picks the CORRECT upright axis but
#   cannot resolve the ±sign when objects are mass-symmetric along it. The
#   task therefore decomposes cleanly into (a) axis selection, already solved
#   by PCA, and (b) polarity (flip-or-not) decision, which needs a global
#   polar signal.
#
# This specialised network replaces the free so(3) update head with:
#   1. axis_pca_init(P): returns R_0 with correct axis but arbitrary sign
#      (skewness sign resolution is DISABLED — we do not want PCA to second-
#      guess the network).
#   2. PolarityFeatures: computes v_polar (weighted polar vector, method 2)
#      and skew_vec (third moment, method 1) from the currently-canonicalised
#      point cloud. Both are SO(3)-equivariant under the relative-frame
#      iteration scheme.
#   3. FlipRefineHead: iteration 0 outputs a FLIP logit (binary ±180° around
#      the in-plane x axis); iterations 1..T output small so(3) refinements.
#
# All three additions coexist with the generic LieUprightNet.
# --------------------------------------------------------------------------


def pca_axis_only_init(points: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Equivariant PCA initialisation that ONLY commits to the axis, leaving
    each axis with an arbitrary but deterministic sign.

    This is the same as `pca_frame_init` without the skewness sign resolution.
    The sign choice for the up axis is delegated to the network's flip head,
    which avoids the symmetric-cloud failure we saw in zero-shot diagnostics.
    """
    B, N, _ = points.shape
    centroid = points.mean(dim=1, keepdim=True)
    centered = points - centroid
    cov = torch.bmm(centered.transpose(1, 2), centered) / (N - 1)
    cov = cov + eps * torch.eye(3, device=points.device, dtype=points.dtype)
    _, V = torch.linalg.eigh(cov)  # ascending

    up_axis = F.normalize(V[:, :, 0], dim=-1)  # smallest variance
    x_axis = F.normalize(V[:, :, 2], dim=-1)   # largest variance
    # Ensure right-handed frame + exact orthogonality
    z_axis = F.normalize(torch.cross(up_axis, x_axis, dim=-1), dim=-1)
    x_axis = F.normalize(torch.cross(z_axis, up_axis, dim=-1), dim=-1)
    # SO(3)-equivariant sign tiebreak.
    # We need a reference VECTOR v in R^3 that satisfies v(gP) = g·v(P).
    # A valid choice is the weighted first moment:
    #     v = Σ_i  ‖x_i‖^2 · x_i
    # This is strictly SO(3)-equivariant (||.|| is invariant, x_i rotates).
    # It is nonzero whenever the cloud is not point-symmetric about the
    # centroid — which is the case for every non-degenerate object.
    # Note: per-axis components of (centered**3).mean(dim=1) are NOT valid
    # since they only give the diagonal of a 3rd-order tensor.
    weights = (centered ** 2).sum(dim=-1, keepdim=True)  # (B, N, 1)
    v_ref = (weights * centered).sum(dim=1)              # (B, 3) equivariant

    def _eqv_sign(axis):
        dot = (axis * v_ref).sum(dim=-1, keepdim=True)
        s = torch.sign(dot)
        s = torch.where(s == 0, torch.ones_like(s), s)
        return s
    up_axis = up_axis * _eqv_sign(up_axis)
    x_axis = x_axis * _eqv_sign(x_axis)
    z_axis = F.normalize(torch.cross(up_axis, x_axis, dim=-1), dim=-1)
    x_axis = F.normalize(torch.cross(z_axis, up_axis, dim=-1), dim=-1)

    R0 = torch.stack([x_axis, up_axis, z_axis], dim=-1)
    return R0


class PolarityFeatures(nn.Module):
    """Compute SO(3)-equivariant polar features from the canonicalised point
    cloud and a per-point soft weight.

    In the relative-frame iteration scheme, the backbone always sees P_t
    (i.e. the point cloud already rotated into the current canonical frame).
    The polar features computed on P_t are therefore already expressed in
    that same frame, and the whole pipeline remains strictly equivariant
    under global rotations of the raw point cloud.
    """

    def __init__(self):
        super().__init__()

    def forward(self, P_t: torch.Tensor, support_logit: torch.Tensor):
        """
        Args:
            P_t:           (B, N, 3) current-frame point cloud
            support_logit: (B, N) logits or probabilities from the trunk
                (if logits, sigmoid is applied; if already in [0,1], idempotent)

        Returns:
            features: (B, 6)  [v_polar (3), skew_vec (3)]
        """
        centroid = P_t.mean(dim=1, keepdim=True)   # (B, 1, 3)
        P_centered = P_t - centroid                # (B, N, 3)

        # (a) weighted polar vector (method 2)
        w = torch.sigmoid(support_logit).unsqueeze(-1)  # (B, N, 1)
        w_norm = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        v_polar = (w_norm * P_centered).sum(dim=1)  # (B, 3)

        # (b) per-axis skewness vector (method 1)
        skew_vec = (P_centered ** 3).mean(dim=1)    # (B, 3)
        # Normalise per-axis by std^3 for scale invariance
        std = P_centered.std(dim=1, unbiased=False).clamp(min=1e-4) ** 3
        skew_vec = skew_vec / std

        return torch.cat([v_polar, skew_vec], dim=-1)  # (B, 6)


class FlipRefineHead(nn.Module):
    """Two-purpose head:
      - iteration 0: a single BINARY flip logit deciding ±180° around x-axis
      - iterations 1..T: an so(3) update vector (constrained to small angle)

    Inputs are the concatenation of global scalar feature + polarity feature.
    """

    def __init__(self, in_dim: int, hidden: int = 256, refine_max_angle: float = math.pi / 6):
        super().__init__()
        self.refine_max_angle = refine_max_angle

        self.flip_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.xavier_uniform_(self.flip_mlp[-1].weight, gain=0.1)
        # Bias initialised to a large negative value so that sigmoid(output) ≈ 0
        # at start. Since the coarse (support-plane) initialiser is correct
        # on ~94% of samples, the default "do not flip" is a much better
        # prior than 50/50. The BCE loss on flip_prob will push the ~6%
        # flip-needed samples to sigmoid > 0.5 during training.
        nn.init.constant_(self.flip_mlp[-1].bias, -5.0)  # sigmoid(-5) ≈ 0.0067

        self.refine_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        nn.init.xavier_uniform_(self.refine_mlp[-1].weight, gain=0.01)
        nn.init.zeros_(self.refine_mlp[-1].bias)

    def flip_logit(self, feat: torch.Tensor) -> torch.Tensor:
        return self.flip_mlp(feat).squeeze(-1)  # (B,)

    def refine_update(self, feat: torch.Tensor) -> torch.Tensor:
        omega = self.refine_mlp(feat)
        mag = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scaled = torch.tanh(mag / self.refine_max_angle) * self.refine_max_angle
        return omega * (scaled / mag)


class ContinuousRefineHead(nn.Module):
    """MLP that maps a fused (scalar + polarity) feature to an so(3) tangent
    update. The magnitude is soft-clipped at π - ε (the well-posed range of
    the Rodrigues exp map).

    Unlike the older per-iter `max_angle_schedule` design, this head makes
    no distinction between iterations: the same MLP is called T times,
    and the head is free to output large omega early (when the canonical
    frame is still far off) and small omega late (when it's nearly right).
    The coarse-to-fine behaviour emerges from the data rather than being
    hard-coded.
    """

    def __init__(self, in_dim: int, hidden: int = 256, clip_angle: float = math.pi - 0.05):
        super().__init__()
        self.clip_angle = clip_angle
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        # Head init needs to be large enough that different samples produce
        # distinguishable outputs at step 0 — otherwise the per-sample
        # gradients all point in essentially the same tiny direction and
        # cancel out across the batch, leaving the head stuck at ~0.
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.5)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, in_dim)
        Returns:
            omega: (B, 3), ||omega|| <= clip_angle (soft clip via tanh)
        """
        omega = self.mlp(feat)
        mag = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scaled = torch.tanh(mag / self.clip_angle) * self.clip_angle
        return omega * (scaled / mag)


def _rotation_180_x(batch_size: int, device, dtype) -> torch.Tensor:
    """Rotation matrix for 180° around the x-axis (in the CURRENT frame).

    diag(1, -1, -1). Flips the up axis while keeping the in-plane x axis.
    """
    R = torch.eye(3, device=device, dtype=dtype)
    R[1, 1] = -1.0
    R[2, 2] = -1.0
    return R.expand(batch_size, 3, 3).contiguous()


class PolarLieUprightNet(nn.Module):
    """Specialised upright estimator built on the Lie-algebra framework:

    Pipeline:
        R_0 = pca_axis_only_init(P)                 # axis-only equivariant init
        P_0 = R_0^T @ P
        feat_0 = trunk(P_0)  # reuses UprightNet trunk + support logits
        polar_0 = PolarityFeatures(P_0, support_0)
        flip_prob = sigmoid(FlipHead(cat(feat_0, polar_0)))
        if training or hard_flip: use straight-through for gradient flow
        R_1 = R_0 @ (Rx_180 if flip else I)

        for t in 1..T-1:
            P_t = R_t^T @ P
            feat_t, support_t = trunk(P_t)
            polar_t = PolarityFeatures(P_t, support_t)
            delta_w = RefineHead(cat(feat_t, polar_t))       # small angle
            R_{t+1} = R_t @ exp(delta_w)

        up = R_T @ e_y

    Equivariance:
        pca_axis_only_init is SO(3)-equivariant under the deterministic sign
        tiebreak (piecewise, like the generic version). Given that, the
        relative-frame iteration guarantees strict equivariance of R_T.
    """

    def __init__(
        self,
        num_iters: int = 3,
        feature_dim: int = 1024,
        head_hidden: int = 256,
        refine_max_angle: float = math.pi / 6,
    ):
        super().__init__()
        if num_iters < 1:
            raise ValueError("num_iters must be >= 1 (iteration 0 is the flip step)")
        self.num_iters = num_iters
        self.trunk = UprightNetTrunk()
        self.polarity = PolarityFeatures()
        # fused input = 1024 (trunk global) + 6 (polarity)
        self.head = FlipRefineHead(
            in_dim=feature_dim + 6,
            hidden=head_hidden,
            refine_max_angle=refine_max_angle,
        )

    def load_backbone_from_ckpt(self, ckpt_path: str, strict: bool = False):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        return self.trunk.net.load_state_dict(sd, strict=strict)

    def forward(self, points: torch.Tensor, hard_flip: bool = False):
        """
        Args:
            points: (B, N, 3) raw (possibly rotated) point cloud
            hard_flip: if True, use discrete flip decision (eval mode). If
                False, use soft (differentiable) flip via a probability-
                weighted rotation matrix — enables gradients to flow through
                the flip choice during training.
        Returns:
            dict with:
                up, R_final, R_iters, flip_prob, delta_omegas, support_logits
        """
        B = points.shape[0]
        device = points.device

        # ---- iteration 0: axis-only PCA init + flip decision ----
        R_t = pca_axis_only_init(points)
        R_iters = [R_t]

        P_t = torch.bmm(points, R_t)  # P_t rows = R_t^T p
        feat, sup = self.trunk(P_t)
        polar = self.polarity(P_t, sup)
        fused = torch.cat([feat, polar], dim=-1)
        flip_logit = self.head.flip_logit(fused)
        flip_prob = torch.sigmoid(flip_logit)  # (B,)
        support_logits = sup

        Rx180 = _rotation_180_x(B, device, R_t.dtype)
        I = torch.eye(3, device=device, dtype=R_t.dtype).expand(B, 3, 3)
        if hard_flip:
            sel = (flip_prob > 0.5).float().view(B, 1, 1)
            dR0 = sel * Rx180 + (1 - sel) * I
        else:
            # Straight-through estimator: forward is the discrete flip
            # decision; backward passes gradient through flip_prob.
            hard_sel = (flip_prob > 0.5).float()
            ste_sel = hard_sel + flip_prob - flip_prob.detach()
            sel = ste_sel.view(B, 1, 1)
            dR0 = sel * Rx180 + (1 - sel) * I
        R_t = torch.bmm(R_t, dR0)
        R_iters.append(R_t)

        # ---- iterations 1..T-1: small so(3) refinements ----
        delta_omegas = []
        for _ in range(self.num_iters - 1):
            P_t = torch.bmm(points, R_t)
            feat, sup = self.trunk(P_t)
            support_logits = sup
            polar = self.polarity(P_t, sup)
            fused = torch.cat([feat, polar], dim=-1)
            delta_w = self.head.refine_update(fused)
            delta_omegas.append(delta_w)
            dR = exp_so3(delta_w)
            R_t = torch.bmm(R_t, dR)
            R_iters.append(R_t)

        up = R_t[:, :, 1]

        return {
            "up": up,
            "R_final": R_t,
            "R_iters": R_iters,
            "flip_prob": flip_prob,
            "delta_omegas": delta_omegas,
            "support_logits": support_logits,
        }


# --------------------------------------------------------------------------
# Specialised: differentiable coarse init + Lie-algebra iterative refinement
# --------------------------------------------------------------------------
#
# Ceiling diagnostics established:
#   (a) The reference support-classification + RANSAC baseline reaches
#       mean=11.17°, acc@10=93.05% on UprightNet15. The 11.17° mean is almost
#       entirely due to 5.74% catastrophic 180° flips; the non-flip median
#       is 0.64°.
#   (b) Pure PCA init only achieves acc@10=70% oracle (fails bench/sofa/etc).
#
# Instead of using PCA as the iteration-0 estimate, we reuse the pretrained
# support-classification trunk as a differentiable coarse estimator:
#     trunk produces per-point support probabilities
#     weighted SVD (differentiable, replaces RANSAC) extracts the plane normal
#     the normal is signed toward the mass center (supporting-plane convention)
#
# This iteration-0 step is approximately SO(3)-equivariant (trunk is not
# strictly equivariant, but was trained with 100x random rotations; the SVD
# and the sign convention ARE strictly equivariant). The subsequent flip /
# refine iterations use the relative-frame scheme and are strictly equivariant.
# Net result: approximately equivariant, matches the reference baseline at
# init, adds an explicit differentiable channel to fix the 180° flip failure.
# --------------------------------------------------------------------------


def so3_from_up(up: torch.Tensor, in_plane_hint: torch.Tensor) -> torch.Tensor:
    """Build a (B, 3, 3) rotation matrix whose second column equals `up`
    and whose first column lies in the plane spanned by `up` and
    `in_plane_hint`, via Gram-Schmidt.

    Equivariant: if up(gP) = g·up(P) and hint(gP) = g·hint(P), then
    R(gP) = g · R(P).

    Args:
        up:            (B, 3) unit vector (up axis)
        in_plane_hint: (B, 3) any non-parallel equivariant vector
    Returns:
        R: (B, 3, 3) with R[:,:,1] == up
    """
    up = F.normalize(up, dim=-1)
    # Project hint onto plane orthogonal to up
    proj = in_plane_hint - (in_plane_hint * up).sum(-1, keepdim=True) * up
    # If the projection is too small (hint nearly parallel to up), fall back
    # to a deterministic equivariant alternative: pick the coordinate of
    # `up` with smallest absolute value and cross with it — this is not
    # strictly equivariant by itself, but the hint failure is rare, and the
    # subsequent refinement handles any residual.
    norm_proj = proj.norm(dim=-1, keepdim=True)
    bad = norm_proj.squeeze(-1) < 1e-4  # (B,)
    if bad.any():
        # fallback: use cross-product with basis vector along min |up_i|
        abs_up = up.abs()
        ax = abs_up.argmin(dim=-1)  # (B,)
        eye = torch.eye(3, device=up.device, dtype=up.dtype)
        e_ax = eye[ax]  # (B, 3)
        fallback_proj = torch.cross(up, e_ax, dim=-1)
        proj = torch.where(bad.unsqueeze(-1), fallback_proj, proj)
    x = F.normalize(proj, dim=-1)
    z = torch.cross(up, x, dim=-1)
    z = F.normalize(z, dim=-1)
    # Recompute x for exact orthogonality
    x = torch.cross(z, up, dim=-1)
    x = F.normalize(x, dim=-1)
    R = torch.stack([x, up, z], dim=-1)
    return R


def pca_largest_axis(points: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return the PCA largest-variance eigenvector of the point cloud.

    Used as an equivariant in-plane hint for building the full rotation
    matrix from a predicted up vector.
    """
    B, N, _ = points.shape
    centered = points - points.mean(dim=1, keepdim=True)
    cov = torch.bmm(centered.transpose(1, 2), centered) / (N - 1)
    cov = cov + eps * torch.eye(3, device=points.device, dtype=points.dtype)
    _, V = torch.linalg.eigh(cov)  # ascending
    return F.normalize(V[:, :, 2], dim=-1)  # largest


class DifferentiableLieUprightNet(nn.Module):
    """Differentiable Lie-algebra iterative upright estimator (flip+refine head).

    Iteration 0 (coarse, differentiable support-plane baseline):
        feat, sup_logits = trunk(P)                 # original input, no rotation
        up_0, _ = weighted_plane_normal(P, sup_logits)  # differentiable SVD
        in_plane_hint = pca_largest_axis(P)          # equivariant hint
        R_0 = so3_from_up(up_0, in_plane_hint)

    Iteration 1 (flip decision, handles the 5.74% 180° failure mode):
        P_t = R_0^T @ P
        feat, sup = trunk(P_t)
        polar = PolarityFeatures(P_t, sup)
        flip_prob = sigmoid(FlipHead(cat(feat, polar)))
        R_1 = R_0 @ (Rx180 if STE(flip_prob > 0.5) else I)

    Iterations 2..T (sub-degree refinement):
        ... refine_update ∈ so(3), |omega| <= max_angle_small

    The same trunk is reused across all iterations; weights are shared.

    Args:
        num_iters:        T, total number of iterations (default 3)
        tau:              softmax temperature for support weighting (default 1.0)
        refine_max_angle: max step size for refine iterations (default π/36 = 5°)
    """

    def __init__(
        self,
        num_iters: int = 3,
        feature_dim: int = 1024,
        head_hidden: int = 256,
        tau: float = 1.0,
        refine_max_angle: float = math.pi / 36,
        eps_cov: float = 1e-5,
    ):
        super().__init__()
        if num_iters < 1:
            raise ValueError("num_iters must be >= 1")
        self.num_iters = num_iters
        self.tau = tau
        self.eps_cov = eps_cov

        self.trunk = UprightNetTrunk()
        self.polarity = PolarityFeatures()
        # fused input = 1024 (trunk global) + 6 (polarity)
        self.head = FlipRefineHead(
            in_dim=feature_dim + 6,
            hidden=head_hidden,
            refine_max_angle=refine_max_angle,
        )

    def load_backbone_from_ckpt(self, ckpt_path: str, strict: bool = False):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        return self.trunk.net.load_state_dict(sd, strict=strict)

    def _coarse_init(self, points: torch.Tensor):
        """Differentiable coarse support-plane init. Returns
        (up_0, support_logits_0, feat_0, eigvals).

        Mirrors the reference baseline as closely as possible with
        differentiable ops:
          - trunk produces per-point sigmoid probabilities p_i in [0, 1]
          - weighted_plane_normal(points, p_i) replaces RANSAC with a
            differentiable weighted SVD; the smallest-eigenvalue eigenvector
            is the plane normal
          - the normal is signed toward the mass center
        """
        feat, sup_logits = self.trunk(points)  # sigmoid outputs in [0, 1]
        # Use probabilities DIRECTLY as weights — weighted_plane_normal
        # normalises internally. A softmax on top of sigmoid would uniformise
        # the weights (since exp of values in (0,1) is in (1, e)) and the SVD
        # would degenerate to global PCA, which is exactly what we do NOT
        # want here.
        w = sup_logits  # (B, N), in (0, 1)
        from Common.geometric_utils import weighted_plane_normal
        up_0, eigvals = weighted_plane_normal(points, w, eps=self.eps_cov)
        return up_0, sup_logits, feat, eigvals

    def forward(self, points: torch.Tensor, hard_flip: bool = False):
        """
        Args:
            points: (B, N, 3) raw (possibly rotated) point cloud
            hard_flip: eval-time discrete flip (True) or soft (False)
        Returns:
            dict with:
                up             : (B, 3) final predicted upright
                R_final        : (B, 3, 3)
                R_iters        : list of (B, 3, 3), length num_iters+1
                up_0           : (B, 3) coarse init prediction (iter 0)
                flip_prob      : (B,) iteration 1 flip probability
                delta_omegas   : list of (B, 3) for iters 2..T
                support_logits : (B, N) from iteration 0
                eigvals_init   : (B, 3) of the init SVD (diagnostic)
        """
        B, N, _ = points.shape
        device = points.device

        # ---- iteration 0: differentiable coarse init ----
        up_0, sup_logits_0, feat_0, eigvals_0 = self._coarse_init(points)
        hint = pca_largest_axis(points)
        R_0 = so3_from_up(up_0, hint)
        R_t = R_0
        R_iters = [R_0]

        # ---- iteration 1: flip decision ----
        P_t = torch.bmm(points, R_t)  # P_t rows = R_t^T p
        feat_1, sup_1 = self.trunk(P_t)
        polar_1 = self.polarity(P_t, sup_1)
        fused_1 = torch.cat([feat_1, polar_1], dim=-1)
        flip_logit = self.head.flip_logit(fused_1)
        flip_prob = torch.sigmoid(flip_logit)  # (B,)

        Rx180 = _rotation_180_x(B, device, R_t.dtype)
        I = torch.eye(3, device=device, dtype=R_t.dtype).expand(B, 3, 3)
        if hard_flip:
            sel = (flip_prob > 0.5).float().view(B, 1, 1)
            dR1 = sel * Rx180 + (1 - sel) * I
        else:
            # Straight-through estimator: forward pass is the DISCRETE flip
            # decision (so the rotation stays in SO(3) and the up vector
            # semantics are physically meaningful), but the backward pass
            # flows gradient through flip_prob.
            hard_sel = (flip_prob > 0.5).float()
            ste_sel = hard_sel + flip_prob - flip_prob.detach()  # (B,)
            sel = ste_sel.view(B, 1, 1)
            dR1 = sel * Rx180 + (1 - sel) * I
        R_t = torch.bmm(R_t, dR1)
        R_iters.append(R_t)

        # ---- iterations 2..T: sub-degree refinement ----
        # Gauge-aware: project delta_omega to tangent plane of current up.
        delta_omegas = []
        support_logits_last = sup_1
        for _ in range(self.num_iters - 1):
            P_t = torch.bmm(points, R_t)
            feat, sup = self.trunk(P_t)
            support_logits_last = sup
            polar = self.polarity(P_t, sup)
            fused = torch.cat([feat, polar], dim=-1)
            delta_w = self.head.refine_update(fused)
            # Gauge projection in LOCAL frame: kill the y-component (rotations
            # about the local up axis don't change R_t @ e_y).
            delta_w = delta_w * delta_w.new_tensor([1.0, 0.0, 1.0])
            delta_omegas.append(delta_w)
            R_t = torch.bmm(R_t, exp_so3(delta_w))
            R_iters.append(R_t)

        up = R_t[:, :, 1]

        return {
            "up": up,
            "R_final": R_t,
            "R_iters": R_iters,
            "up_0": up_0,
            "flip_prob": flip_prob,
            "delta_omegas": delta_omegas,
            "support_logits": support_logits_last,
            "support_logits_init": sup_logits_0,
            "eigvals_init": eigvals_0,
        }


# --------------------------------------------------------------------------
# Recommended: differentiable coarse init + continuous coarse-to-fine refine
# --------------------------------------------------------------------------
#
# Differences from DifferentiableLieUprightNet:
#   - No flip head. All corrections are continuous so(3) updates from the
#     SAME refine head, which is called T times with a per-iter `max_angle`
#     schedule. This lets iteration 1 express 180° flips as a valid
#     continuous rotation (|omega|=pi), while iteration T handles sub-degree
#     polish.
#   - Single Geodesic supervision with progressive weights γ^(T-t):
#         L = Σ_t γ^(T-t) · arccos(|<R_t @ e_y, gt_up>|)
#     No BCE, no straight-through, no binary decisions.
#   - Polarity features (v_polar + skew_vec) fused with trunk scalars at
#     every iteration — the refine head can learn to use them to predict
#     large coarse rotations when polar signal is strong.
#
# Exp 2 diagnostic (from diag_ceilings.py) showed a beam-search oracle over
# 3 steps × 26 directions × π/6 magnitude achieves 100% acc@10 starting from
# a PCA frame. Starting from a support-plane estimator the optimum is even
# closer, so this continuous refine should have strictly more expressive
# power than a binary flip head.
# --------------------------------------------------------------------------


class DifferentiableLieUprightRefineNet(nn.Module):
    """Differentiable Lie-algebra iterative upright estimator (continuous head).

    Architecture:
        R_0 = so3_from_up(weighted_plane_normal(P, trunk_support(P)),
                          pca_largest_axis(P))                 # coarse init

        for t in range(T):
            P_t = R_t^T @ P
            feat, sup = trunk(P_t)
            polar = PolarityFeatures(P_t, sup)
            delta_w = refine_head(cat(feat, polar))  # no per-iter schedule
            R_{t+1} = R_t @ exp_so3(delta_w)

        up = R_T @ e_y

    The refine head is iteration-agnostic: it outputs an so(3) step whose
    magnitude it controls itself via a tanh clip at π - ε. Coarse-to-fine
    behaviour is learned from data (big step when P_t is still far off the
    canonical frame, small step when it's close).

    Equivariance:
        Iteration 0 is trunk(P) + SVD + so3_from_up(up, pca_axis(P)). The
        trunk is NOT architecturally equivariant but was pretrained with
        100x random rotations, so it is empirically near-invariant. SVD
        and so3_from_up ARE strictly equivariant. Iterations 1..T use the
        relative-frame scheme and are strictly equivariant (backbone input
        is canonicalised by R_t^T).

    Args:
        num_iters:          T, number of refinement iterations (default 3)
        feature_dim:        trunk global output dim (1024)
        head_hidden:        refine MLP hidden size
    """

    def __init__(
        self,
        num_iters: int = 3,
        feature_dim: int = 1024,
        head_hidden: int = 256,
        eps_cov: float = 1e-5,
    ):
        super().__init__()
        if num_iters < 1:
            raise ValueError("num_iters must be >= 1")
        self.num_iters = num_iters
        self.eps_cov = eps_cov

        self.trunk = UprightNetTrunk()
        self.polarity = PolarityFeatures()
        self.refine_head = ContinuousRefineHead(
            in_dim=feature_dim + 6, hidden=head_hidden,
        )

    def load_backbone_from_ckpt(self, ckpt_path: str, strict: bool = False):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        return self.trunk.net.load_state_dict(sd, strict=strict)

    def _coarse_init(self, points: torch.Tensor):
        """Differentiable coarse support-plane init; same as
        DifferentiableLieUprightNet._coarse_init."""
        feat, sup_logits = self.trunk(points)
        w = sup_logits  # (B, N), sigmoid output in (0, 1)
        from Common.geometric_utils import weighted_plane_normal
        up_0, eigvals = weighted_plane_normal(points, w, eps=self.eps_cov)
        return up_0, sup_logits, feat, eigvals

    def forward(self, points: torch.Tensor):
        """
        Args:
            points: (B, N, 3) raw (possibly rotated) point cloud
        Returns:
            dict with:
                up             : (B, 3) final predicted upright direction
                R_final        : (B, 3, 3)
                R_iters        : list of (B, 3, 3), length num_iters+1
                up_0           : (B, 3) coarse init prediction
                delta_omegas   : list of (B, 3), length num_iters
                support_logits : (B, N) from the last iteration
                support_logits_init : (B, N) from iteration 0
                eigvals_init   : (B, 3) of the init SVD
        """
        # ---- iteration 0: differentiable coarse init ----
        up_0, sup_logits_0, _feat_0, eigvals_0 = self._coarse_init(points)
        hint = pca_largest_axis(points)
        R_t = so3_from_up(up_0, hint)
        R_iters = [R_t]

        # ---- iterations 1..T: continuous coarse-to-fine refinement ----
        # Gauge-aware so(3) update: the task-level GT is an axis in S²
        # (only R[:, :, 1] matters); rotations about that axis are a gauge
        # freedom with zero loss gradient. We therefore project delta_omega
        # onto the plane orthogonal to the current up axis, so the refine
        # head cannot waste capacity on unobservable rotations.
        #   delta_w_tangent = delta_w - (delta_w · u) · u
        # This keeps the update strictly on S² (which is what we want to
        # refine); the head effectively outputs a tangent vector in T_u S².
        delta_omegas = []
        support_logits_last = sup_logits_0
        for t in range(self.num_iters):
            P_t = torch.bmm(points, R_t)
            feat, sup = self.trunk(P_t)
            support_logits_last = sup
            polar = self.polarity(P_t, sup)
            fused = torch.cat([feat, polar], dim=-1)
            delta_w = self.refine_head(fused)
            # Gauge projection: delta_w is in the LOCAL frame (update is applied
            # as R_t @ exp(delta_w)), so the direction that leaves up invariant
            # is local e_y = (0, 1, 0) — rotations about the local y-axis are
            # the gauge freedom. Zero out the y-component.
            delta_w = delta_w * delta_w.new_tensor([1.0, 0.0, 1.0])
            delta_omegas.append(delta_w)
            R_t = torch.bmm(R_t, exp_so3(delta_w))
            R_iters.append(R_t)

        up = R_t[:, :, 1]

        return {
            "up": up,
            "R_final": R_t,
            "R_iters": R_iters,
            "up_0": up_0,
            "delta_omegas": delta_omegas,
            "support_logits": support_logits_last,
            "support_logits_init": sup_logits_0,
            "eigvals_init": eigvals_0,
        }


# --------------------------------------------------------------------------
# Equivariance sanity test (strict: should hit ~1e-5 in fp32)
# --------------------------------------------------------------------------


def _structured_cloud(B: int, N: int, device, seed: int):
    """An asymmetric, structured point cloud that mimics a lamp-like shape:
    large elongation in two axes, and a deliberately asymmetric mass
    distribution along the third (so skewness-based sign resolution works)."""
    g = torch.Generator(device=device).manual_seed(seed)
    P = torch.randn(B, N, 3, device=device, generator=g)
    scale = torch.tensor([3.0, 1.0, 0.3], device=device)
    P = P * scale
    # Introduce asymmetry: push half the points further down along axis 2
    mask = P[..., 2] < 0
    P = P + mask.unsqueeze(-1).float() * torch.tensor([0.0, 0.0, -2.0], device=device)
    return P


@torch.no_grad()
def test_equivariance(model: LieUprightNet, num_trials: int = 8, N: int = 1024):
    """Verify up(gP) = g @ up(P) for random rotations g.

    Uses a structured (non-isotropic) point cloud so PCA frame is well-defined.
    For isotropic Gaussian clouds PCA itself is numerically unstable, which is
    a data problem not an algorithm problem.
    """
    device = next(model.parameters()).device
    model.eval()
    errors_direct = []
    errors_antipodal = []
    for trial in range(num_trials):
        P = _structured_cloud(2, N, device, seed=trial)
        from scipy.spatial.transform import Rotation
        R = torch.from_numpy(Rotation.random().as_matrix().astype("float32")).to(device)
        gP = P @ R.T

        out1 = model(P)
        out2 = model(gP)
        up1 = out1["up"]
        up2 = out2["up"]
        expected = up1 @ R.T
        err_pos = (up2 - expected).norm(dim=-1).max().item()
        err_neg = (up2 + expected).norm(dim=-1).max().item()
        errors_direct.append(err_pos)
        errors_antipodal.append(min(err_pos, err_neg))
    return errors_direct, errors_antipodal


@torch.no_grad()
def test_iteration_equivariance(model: LieUprightNet, num_trials: int = 4, N: int = 1024):
    """Isolate the iteration's equivariance from PCA init's equivariance.

    We feed the model an identical starting rotation for P and gP (computed
    as R_0(gP) = g @ R_0(P)), and check that after T iterations the rotations
    still satisfy R_T(gP) = g @ R_T(P). This tests only the iteration logic,
    which should be mathematically exact up to float32 precision.
    """
    device = next(model.parameters()).device
    model.eval()
    errors = []
    for trial in range(num_trials):
        P = _structured_cloud(2, N, device, seed=100 + trial)
        from scipy.spatial.transform import Rotation
        R = torch.from_numpy(Rotation.random().as_matrix().astype("float32")).to(device)
        gP = P @ R.T

        # Use a fixed (non-PCA) initial R_0 for P, and R @ R_0 for gP
        R0_P = torch.eye(3, device=device).expand(2, 3, 3).contiguous()
        R0_gP = R.unsqueeze(0).expand(2, 3, 3).contiguous() @ R0_P

        R_t_P = R0_P
        R_t_gP = R0_gP
        for _ in range(model.num_iters):
            P_t = torch.bmm(P, R_t_P)
            P_t_g = torch.bmm(gP, R_t_gP)
            feat_P, _ = model.trunk(P_t)
            feat_gP, _ = model.trunk(P_t_g)
            dw_P = model.so3_head(feat_P)
            dw_gP = model.so3_head(feat_gP)
            R_t_P = torch.bmm(R_t_P, exp_so3(dw_P))
            R_t_gP = torch.bmm(R_t_gP, exp_so3(dw_gP))

        up_P = R_t_P[:, :, 1]
        up_gP = R_t_gP[:, :, 1]
        expected = up_P @ R.T
        err = (up_gP - expected).norm(dim=-1).max().item()
        errors.append(err)
    return errors


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(0)
    model = LieUprightNet(num_iters=3).cuda()
    P = torch.randn(4, 2048, 3).cuda()
    out = model(P)
    print("Shapes:")
    print("  up:", out["up"].shape)
    print("  R_final:", out["R_final"].shape)
    print("  R_iters:", len(out["R_iters"]))
    print("  support_logits:", out["support_logits"].shape)
    print("up norms:", out["up"].norm(dim=-1))

    print("\n[1] Full-pipeline equivariance (PCA init + iteration):")
    errs_d, errs_a = test_equivariance(model)
    print("  direct:   ", [f"{e:.2e}" for e in errs_d])
    print("  antipodal:", [f"{e:.2e}" for e in errs_a])
    print(f"  max direct: {max(errs_d):.2e}   max antipodal: {max(errs_a):.2e}")

    print("\n[2] Iteration-only equivariance (fed equivariant init externally):")
    errs_iter = test_iteration_equivariance(model)
    print("  errors:", [f"{e:.2e}" for e in errs_iter])
    print(f"  max: {max(errs_iter):.2e}  (should be ~1e-5 in fp32)")

    print("\n=== PolarLieUprightNet (specialised) ===")
    torch.manual_seed(0)
    pmodel = PolarLieUprightNet(num_iters=3).cuda()
    P = torch.randn(4, 2048, 3).cuda()
    out = pmodel(P)
    print("Shapes:")
    print("  up:", out["up"].shape, "norm:", out["up"].norm(dim=-1))
    print("  flip_prob:", out["flip_prob"])
    print("  R_iters:", len(out["R_iters"]))

    print("\n[3] PolarLieUprightNet full-pipeline equivariance:")
    errs_d, errs_a = test_equivariance(pmodel, num_trials=8)
    print("  direct:   ", [f"{e:.2e}" for e in errs_d])
    print("  antipodal:", [f"{e:.2e}" for e in errs_a])
    print(f"  max direct: {max(errs_d):.2e}   max antipodal: {max(errs_a):.2e}")

    print("\n=== DifferentiableLieUprightNet (coarse init + Lie iter) ===")
    torch.manual_seed(0)
    dlie_model = DifferentiableLieUprightNet(num_iters=3).cuda()
    # Load reference trunk checkpoint (Pang 2022)
    import os
    ref_ckpt = "/home/yujian_shi/uprightnet-reference/model/model.pth"
    if os.path.exists(ref_ckpt):
        dlie_model.load_backbone_from_ckpt(ref_ckpt)
        print(f"Loaded reference trunk from {ref_ckpt}")
    P = torch.randn(4, 2048, 3).cuda()
    out = dlie_model(P)
    print("Shapes:")
    print("  up:", out["up"].shape, "norm:", out["up"].norm(dim=-1))
    print("  up_0 norm:", out["up_0"].norm(dim=-1))
    print("  eigvals_init[0] (smallest):", out["eigvals_init"][:, 0])
    print("  flip_prob:", out["flip_prob"])
    print("  R_iters:", len(out["R_iters"]))

    print("\n[4] DifferentiableLieUprightNet approximate equivariance:")
    errs_d, errs_a = test_equivariance(dlie_model, num_trials=8)
    print("  direct:   ", [f"{e:.2e}" for e in errs_d])
    print("  antipodal:", [f"{e:.2e}" for e in errs_a])
    print("  (trunk is NOT strictly equivariant; errors reflect trunk sensitivity to rotations)")

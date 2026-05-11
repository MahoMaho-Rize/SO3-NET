"""
SHS-style Signed Direction Network for upright orientation estimation.

Inspired by SHS-Net (CVPR 2023): instead of estimating an unsigned axis and
then resolving polarity as a separate step, we directly predict a SIGNED
upright direction end-to-end.  The key insight from SHS-Net is that sign
(polarity) is a *global* property that requires shape-level context — local
geometry alone cannot distinguish lamp-base from lamp-shade.

Architecture:
    Input: (B, N, 3) point cloud (randomly rotated)

    1. Local encoder (DGCNN backbone from Pang 2022, reusable pretrained weights)
       -> per-point features x_a (B, 128, N) + global feat (B, 1024)

    2. Global context module (lightweight Transformer on per-point features)
       -> context-enriched global feature (B, D_global)

    3. Signed direction head: MLP that directly predicts u ∈ S²
       -> (B, 3), trained with signed geodesic loss (no antipodal folding)

No RANSAC, no Lie iteration, no flip head, no polarity features.
The network learns polarity from data through global context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from network import UprightNet, get_graph_feature


class GlobalContextModule(nn.Module):
    """Lightweight Transformer that enriches the global feature with
    shape-level context.  Operates on the post-self-attention per-point
    features from the DGCNN backbone (512D = 4x128 from 4 SA layers).

    The Transformer lets distant points communicate (e.g. lamp-base
    and lamp-shade), which is exactly what's needed to resolve polarity.

    We project 512D -> d_model first to keep the Transformer compact.
    """

    def __init__(self, in_dim=512, d_model=128, num_heads=4, num_layers=2, ff_dim=256):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.out_dim = d_model

    def forward(self, x):
        """
        Args:
            x: (B, C, N) per-point feature map (C=512, post-self-attention)
        Returns:
            global_feat: (B, d_model) context-enriched global feature
        """
        x = x.permute(0, 2, 1)  # (B, N, C)
        x = self.proj_in(x)  # (B, N, d_model)
        x = self.transformer(x)  # (B, N, d_model)
        return x.mean(dim=1)  # (B, d_model) global average pool


class SignedDirectionHead(nn.Module):
    """MLP head that predicts a signed unit direction on S².

    Input: concatenation of trunk global feat + context feat.
    Output: (B, 3) unit vector — the predicted upright direction WITH sign.
    """

    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3),
        )
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.1)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat):
        raw = self.mlp(feat)  # (B, 3)
        return F.normalize(raw, dim=-1)


class DecomposedDirectionHead(nn.Module):
    """Predicts unsigned axis + polarity sign separately.

    Rationale: the axis is a local-geometric quantity (lines through origin),
    while the sign is a global-semantic property (which end points "up").
    Splitting them into separate heads lets each branch receive a gradient
    tailored to its sub-task:

    - axis_head: trained with antipodal geodesic loss (|cos|).
    - sign_head: trained with BCE; gradient is strongest near the decision
      boundary, which is exactly the flip-prone regime.

    Input: concatenation of trunk global feat + context feat.
    Outputs:
        axis:       (B, 3) unit vector (unsigned)
        sign_logit: (B,)   raw logit; sigmoid(.) in (0,1) is P(keep axis)
    """

    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.axis_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3),
        )
        self.sign_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.xavier_uniform_(self.axis_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.axis_head[-1].bias)
        nn.init.xavier_uniform_(self.sign_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.sign_head[-1].bias)

    def forward(self, feat):
        h = self.shared(feat)
        axis = F.normalize(self.axis_head(h), dim=-1)
        sign_logit = self.sign_head(h).squeeze(-1)
        return axis, sign_logit


class SHSUprightNet(nn.Module):
    """SHS-style signed direction network for upright estimation.

    Reuses the Pang 2022 DGCNN+SelfAttention trunk as a local encoder,
    adds a Transformer-based global context module, and predicts the
    signed upright direction directly.

    Args:
        context_layers: number of Transformer layers in global context module
        context_heads:  number of attention heads
        head_hidden:    hidden dim of the signed direction MLP head
        freeze_trunk:   if True, freeze the DGCNN trunk (use pretrained weights)
        head_type:      'signed' (single head, original SHS baseline) or
                        'decomposed' (separate axis + sign heads)
    """

    def __init__(
        self,
        context_layers=2,
        context_heads=4,
        head_hidden=512,
        freeze_trunk=True,
        head_type="decomposed",
    ):
        super().__init__()
        if head_type not in ("signed", "decomposed"):
            raise ValueError(f"head_type must be 'signed' or 'decomposed', got {head_type}")
        self.head_type = head_type
        self.trunk = UprightNet()
        self.freeze_trunk = freeze_trunk

        self.context = GlobalContextModule(
            in_dim=512,  # post-SA features: cat(x1,x2,x3,x4) each 128
            d_model=128,
            num_heads=context_heads,
            num_layers=context_layers,
            ff_dim=256,
        )

        fused_dim = 1024 + self.context.out_dim  # = 1152
        if head_type == "signed":
            self.direction_head = SignedDirectionHead(
                in_dim=fused_dim, hidden=head_hidden,
            )
        else:
            self.direction_head = DecomposedDirectionHead(
                in_dim=fused_dim, hidden=head_hidden,
            )

        # Auxiliary support head (reuses trunk's support prediction)
        # No extra parameters — we just expose the trunk's sigmoid output.

    def load_trunk_from_ckpt(self, ckpt_path, strict=False):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        missing, unexpected = self.trunk.load_state_dict(sd, strict=strict)
        if self.freeze_trunk:
            for p in self.trunk.parameters():
                p.requires_grad = False
        return missing, unexpected

    def forward(self, points_bnc):
        """
        Args:
            points_bnc: (B, N, 3) point cloud

        Returns:
            dict with:
                up:             (B, 3) signed upright direction (unit vector)
                support_logits: (B, N) per-point support probability from trunk
                axis:           (B, 3) unsigned axis, decomposed head only
                sign_logit:     (B,)   polarity logit, decomposed head only
        """
        x = points_bnc.transpose(1, 2).contiguous()  # (B, 3, N)
        batch_size = x.size(0)
        num_points = x.size(2)

        # ---- DGCNN local encoder (same as Pang 2022) ----
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
        global_feat = F.adaptive_max_pool1d(x_c, 1).view(batch_size, -1)  # (B, 1024)

        # Per-point support prediction (auxiliary, from trunk)
        x_global = global_feat.view(batch_size, -1, 1).repeat(1, 1, num_points)
        cat = torch.cat((x_a, x_b, x_c, x_global), dim=1)
        sup = net.conv5(cat)
        sup = net.conv6(sup)
        sup = net.conv7(sup)
        support_logits = net.sm_fn(sup).squeeze(1)  # (B, N)

        # ---- Global context module ----
        # Feed post-self-attention features (512D) for richer context.
        # x_b = cat(x1,x2,x3,x4) captures multi-scale attention output.
        context_feat = self.context(x_b)  # (B, 128)

        # ---- Direction head ----
        fused = torch.cat([global_feat, context_feat], dim=-1)  # (B, 1152)

        if self.head_type == "signed":
            up = self.direction_head(fused)
            return {
                "up": up,
                "support_logits": support_logits,
            }

        axis, sign_logit = self.direction_head(fused)
        # Straight-through estimator: forward uses hard sign, gradient flows via
        # the soft (2p - 1) path so the sign head is trainable end-to-end.
        p = torch.sigmoid(sign_logit)
        soft = 2.0 * p - 1.0
        hard = torch.where(soft >= 0, torch.ones_like(soft), -torch.ones_like(soft))
        ste_sign = hard + soft - soft.detach()  # (B,)
        up = axis * ste_sign.unsqueeze(-1)  # (B, 3)

        return {
            "up": up,
            "axis": axis,
            "sign_logit": sign_logit,
            "support_logits": support_logits,
        }

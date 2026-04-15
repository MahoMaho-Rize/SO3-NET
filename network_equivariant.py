"""
E(3)-Equivariant Upright Orientation Estimation Network.

Built with e3nn (Euclidean Neural Networks) library.

Architecture:
    Input:  Point cloud positions (N, 3)
    Graph:  Radius graph with spherical harmonic edge features
    Backbone: Stacked equivariant convolution layers with Gate nonlinearities
    Output: Per-point support probability (l=0 scalar, auxiliary)
            + Global upright direction as vMF distribution (l=1 vector + l=0 scalar)

Key equivariance properties:
    - Direction output (l=1): rotates with input   f(Rx) = R f(x)
    - Kappa output (l=0):    invariant to rotation  g(Rx) = g(x)
    - Support output (l=0):  invariant to rotation  h(Rx) = h(x)

References:
    - e3nn: https://docs.e3nn.org/
    - Original UprightNet: Pang et al., CVPR 2022
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate, FullyConnectedNet

from torch_cluster import radius_graph
from torch_scatter import scatter


# ============================================================
# Building Block: Equivariant Convolution Layer
# ============================================================


class EquivariantConvLayer(nn.Module):
    """
    Single equivariant message passing layer.

    Implements:
        m_ij = TP(f_j, Y^l(r_ij), weight=MLP(||r_ij||))   [message]
        f'_i = (1/sqrt(N)) * sum_j m_ij + self_connection    [aggregate]
        f'_i = Gate(f'_i)                                     [nonlinearity]

    where TP is a FullyConnectedTensorProduct and Y^l are spherical harmonics.

    Args:
        irreps_in:         input irreps (e.g., "32x0e+16x1o+8x2e")
        irreps_out:        output irreps (before gate; gate determines actual output)
        irreps_sh:         spherical harmonic irreps for edges
        max_radius:        cutoff radius for graph
        num_radial_basis:  number of radial basis functions
        radial_neurons:    hidden size of radial MLP
        num_neighbors:     average neighbor count for normalization
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_sh,
        max_radius,
        num_radial_basis=10,
        radial_neurons=64,
        num_neighbors=20.0,
    ):
        super().__init__()
        self.max_radius = max_radius
        self.num_neighbors = num_neighbors
        self.num_radial_basis = num_radial_basis

        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)
        irreps_sh = o3.Irreps(irreps_sh)

        # --- Build Gate ---
        # Separate irreps_out into scalars (directly activated) and non-scalars (gated)
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l == 0])
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0])
        # Gate scalars: one even scalar (0e) per gated irrep, paired with sigmoid.
        # Using 0e (even parity) ensures compatibility with sigmoid (neither even nor odd).
        irreps_gates = o3.Irreps(
            [(mul, (0, 1)) for mul, ir in irreps_gated]  # (0, 1) = 0e
        )

        # Activation functions
        # For even scalars (0e): silu works (no parity constraint on even scalars)
        # For odd scalars (0o): must use odd function (tanh) or absolute value
        act_scalars = []
        for _, ir in irreps_scalars:
            if ir.p == 1:  # even
                act_scalars.append(torch.nn.functional.silu)
            else:  # odd
                act_scalars.append(torch.tanh)
        act_gates = [torch.sigmoid for _ in irreps_gates]

        self.gate = Gate(
            irreps_scalars,
            act_scalars,
            irreps_gates,
            act_gates,
            irreps_gated,
        )
        # The input irreps to the gate (what the conv must produce)
        irreps_conv_out = self.gate.irreps_in

        # --- Tensor Product (message function) ---
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in,
            irreps_sh,
            irreps_conv_out,
            shared_weights=False,
        )

        # --- Radial MLP: distance → tensor product weights ---
        self.radial_mlp = FullyConnectedNet(
            [num_radial_basis, radial_neurons, radial_neurons, self.tp.weight_numel],
            act=torch.nn.functional.silu,
        )

        # --- Self-connection (skip connection) ---
        self.self_connection = o3.Linear(irreps_in, irreps_conv_out)

        self.irreps_in = irreps_in
        self.irreps_out = self.gate.irreps_out

    def forward(self, node_features, edge_index, edge_sh, edge_length_embedded):
        """
        Args:
            node_features:        (num_nodes, irreps_in.dim) node features
            edge_index:           (2, num_edges) graph connectivity
            edge_sh:              (num_edges, irreps_sh.dim) edge spherical harmonics
            edge_length_embedded: (num_edges, num_radial_basis) radial basis embedding

        Returns:
            node_features_out:    (num_nodes, irreps_out.dim) updated features
        """
        edge_src, edge_dst = edge_index

        # Compute per-edge weights from radial MLP
        tp_weight = self.radial_mlp(edge_length_embedded)

        # Message: tensor product of source features with edge SH, weighted by radial MLP
        messages = self.tp(node_features[edge_src], edge_sh, tp_weight)

        # Aggregate messages (sum over neighbors, normalized)
        num_nodes = node_features.shape[0]
        aggregated = scatter(
            messages, edge_dst, dim=0, dim_size=num_nodes, reduce="sum"
        )
        aggregated = aggregated / (self.num_neighbors**0.5)

        # Self-connection (skip)
        self_feat = self.self_connection(node_features)

        # Combine and apply gate nonlinearity
        node_features_out = self.gate(aggregated + self_feat)

        return node_features_out


# ============================================================
# Building Block: Efficient (MACE-style) Depthwise Separable Convolution
# ============================================================


def _build_depthwise_tp_instructions(irreps_in, irreps_sh, target_irreps):
    """
    Build "uvu" mode tensor product instructions for depthwise separable TP.

    In "uvu" mode, the output multiplicity inherits from input1 (node features),
    and only 1 weight per path is needed (since SH has multiplicity 1).
    This reduces per-edge weights from ~3456 (FCTP) to ~11 (depthwise) for L=2.

    Args:
        irreps_in:      node feature irreps (e.g., "32x0e+16x1o+8x2e")
        irreps_sh:      edge spherical harmonic irreps (e.g., "1x0e+1x1o+1x2e")
        target_irreps:  desired output irreps (to filter valid paths)

    Returns:
        irreps_mid:     output irreps of the depthwise TP
        instructions:   list of (i, j, k, "uvu", True) tuples for o3.TensorProduct
    """
    irreps_in = o3.Irreps(irreps_in)
    irreps_sh = o3.Irreps(irreps_sh)
    target_irreps = o3.Irreps(target_irreps)

    # Collect target (l, p) pairs for filtering
    target_lp = {(ir.l, ir.p) for _, ir in target_irreps}

    irreps_mid_list = []
    instructions = []

    for i, (mul_i, ir_i) in enumerate(irreps_in):
        for j, (mul_j, ir_j) in enumerate(irreps_sh):
            for ir_out in ir_i * ir_j:  # CG selection rule: all valid output irreps
                if (ir_out.l, ir_out.p) in target_lp:
                    k = len(irreps_mid_list)
                    irreps_mid_list.append((mul_i, ir_out))  # output mul = input mul
                    instructions.append((i, j, k, "uvu", True))

    irreps_mid = o3.Irreps(irreps_mid_list)
    return irreps_mid, instructions


class EfficientConvLayer(nn.Module):
    """
    MACE-style depthwise separable equivariant convolution layer.

    Decomposes the standard FullyConnectedTensorProduct into:
        1. Linear_up:      channel mixing (shared weights, no per-edge cost)
        2. DepthwiseTP:     CG angular coupling only ("uvu" mode, ~11 weights/edge)
        3. scatter:         message aggregation
        4. Linear_down:     channel mixing (shared weights, no per-edge cost)
        5. Gate:            equivariant nonlinearity
        + self-connection (skip)

    Cost comparison vs FullyConnectedTensorProduct for L=2:
        FCTP:      ~3456 weights per edge  (channel mixing + CG coupling entangled)
        Depthwise: ~11 weights per edge    (CG coupling only, channels mixed by Linears)
        Speedup:   ~5-8x

    Mathematical equivalence:
        The full tensor product z_w = Σ_{u,v} w_{uvw} (x_u ⊗ y_v) is factored into:
        Step 1:  x'_u = Σ_u' A_{u'u} x_{u'}             (Linear_up, shared)
        Step 2:  m_u  = Σ_v  w_v (x'_u ⊗ y_v)           (DepthwiseTP, per-edge)
        Step 3:  z_w  = Σ_u  B_{uw} m_u                  (Linear_down, shared)
        This is a low-rank factorization that preserves equivariance exactly.

    Args:
        irreps_in:         input irreps
        irreps_out:        desired output irreps
        irreps_sh:         spherical harmonic irreps for edges
        max_radius:        cutoff radius
        num_radial_basis:  number of radial basis functions
        radial_neurons:    radial MLP hidden size
        num_neighbors:     average neighbor count for normalization
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_sh,
        max_radius,
        num_radial_basis=10,
        radial_neurons=64,
        num_neighbors=20.0,
    ):
        super().__init__()
        self.max_radius = max_radius
        self.num_neighbors = num_neighbors

        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)
        irreps_sh = o3.Irreps(irreps_sh)

        # --- Build Gate (same logic as EquivariantConvLayer) ---
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in irreps_out if ir.l == 0]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in irreps_out if ir.l > 0]
        )
        irreps_gates = o3.Irreps(
            [(mul, (0, 1)) for mul, ir in irreps_gated]
        )

        act_scalars = []
        for _, ir in irreps_scalars:
            if ir.p == 1:
                act_scalars.append(torch.nn.functional.silu)
            else:
                act_scalars.append(torch.tanh)
        act_gates = [torch.sigmoid for _ in irreps_gates]

        self.gate = Gate(
            irreps_scalars, act_scalars,
            irreps_gates, act_gates,
            irreps_gated,
        )
        irreps_conv_out = self.gate.irreps_in

        # --- Step 1: Pre-TP linear (channel mixing, shared weights) ---
        self.linear_up = o3.Linear(irreps_in, irreps_in)

        # --- Step 2: Depthwise tensor product ("uvu" mode) ---
        # Build instructions: only CG-valid paths, output mul = input mul
        irreps_mid, instructions = _build_depthwise_tp_instructions(
            irreps_in, irreps_sh, target_irreps=irreps_out,
        )

        self.tp = o3.TensorProduct(
            irreps_in, irreps_sh, irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # --- Radial MLP (now outputs TINY weight vector, e.g. ~11 for L=2) ---
        self.radial_mlp = FullyConnectedNet(
            [num_radial_basis, radial_neurons, self.tp.weight_numel],
            act=torch.nn.functional.silu,
        )

        # --- Step 3: Post-TP linear (channel mixing, shared weights) ---
        self.linear_down = o3.Linear(irreps_mid, irreps_conv_out)

        # --- Self-connection (skip) ---
        self.self_connection = o3.Linear(irreps_in, irreps_conv_out)

        self.irreps_in = irreps_in
        self.irreps_out = self.gate.irreps_out

    def forward(self, node_features, edge_index, edge_sh, edge_length_embedded):
        """
        Args:
            node_features:        (num_nodes, irreps_in.dim) node features
            edge_index:           (2, num_edges) graph connectivity
            edge_sh:              (num_edges, irreps_sh.dim) edge spherical harmonics
            edge_length_embedded: (num_edges, num_radial_basis) radial basis embedding

        Returns:
            node_features_out:    (num_nodes, irreps_out.dim) updated features
        """
        edge_src, edge_dst = edge_index

        # Step 1: Channel mixing (shared, cheap)
        node_up = self.linear_up(node_features)

        # Step 2: Depthwise tensor product (per-edge, but very cheap)
        tp_weight = self.radial_mlp(edge_length_embedded)
        messages = self.tp(node_up[edge_src], edge_sh, tp_weight)

        # Aggregate messages
        num_nodes = node_features.shape[0]
        aggregated = scatter(
            messages, edge_dst, dim=0, dim_size=num_nodes, reduce="sum"
        )
        aggregated = aggregated / (self.num_neighbors ** 0.5)

        # Step 3: Channel mixing (shared, cheap)
        aggregated = self.linear_down(aggregated)

        # Self-connection (skip)
        self_feat = self.self_connection(node_features)

        # Gate nonlinearity
        node_features_out = self.gate(aggregated + self_feat)

        return node_features_out


# ============================================================
# Output Head: von Mises-Fisher Direction Head
# ============================================================


class VMFDirectionHead(nn.Module):
    """
    Predicts upright direction as a von Mises-Fisher distribution on S^2.

    Takes equivariant node features, performs global pooling, and outputs:
        mu:    (B, 3) mean direction (unit vector, l=1 equivariant)
        kappa: (B, 1) concentration parameter (l=0 invariant, > 0)

    The l=1 output naturally rotates with the input (equivariance).
    The l=0 output is invariant (confidence doesn't change with rotation).

    Args:
        irreps_in:      input irreps from backbone
        kappa_init:     initial kappa value (controls initial certainty)
    """

    def __init__(self, irreps_in, kappa_init=10.0):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)

        # Linear projection to 1 vector (l=1) + 1 scalar (l=0) for kappa
        self.direction_proj = o3.Linear(irreps_in, o3.Irreps("1x1o"))
        self.kappa_proj = o3.Linear(irreps_in, o3.Irreps("1x0e"))

        # Learnable bias for kappa (initialized to produce kappa_init)
        self.kappa_bias = nn.Parameter(
            torch.tensor([math.log(math.exp(kappa_init) - 1.0)])  # inverse softplus
        )

    def forward(self, node_features, batch):
        """
        Args:
            node_features: (num_nodes, irreps_in.dim) per-node features
            batch:         (num_nodes,) batch index

        Returns:
            mu:    (B, 3) predicted upright direction (unit vector)
            kappa: (B, 1) concentration parameter (> 0)
        """
        # Project to l=1 vector and l=0 scalar per node
        vec_per_node = self.direction_proj(node_features)  # (num_nodes, 3)
        scalar_per_node = self.kappa_proj(node_features)  # (num_nodes, 1)

        # Global pooling (mean over nodes in each graph)
        vec_global = scatter(vec_per_node, batch, dim=0, reduce="mean")  # (B, 3)
        scalar_global = scatter(scalar_per_node, batch, dim=0, reduce="mean")  # (B, 1)

        # Normalize direction to unit vector
        mu = F.normalize(vec_global, dim=-1, eps=1e-8)  # (B, 3)

        # Kappa: softplus to ensure > 0, with learnable bias
        kappa = F.softplus(scalar_global + self.kappa_bias)  # (B, 1)

        return mu, kappa


# ============================================================
# Output Head: Per-point Support Classification (Auxiliary)
# ============================================================


class SupportHead(nn.Module):
    """
    Per-point support classification head (auxiliary task).

    Takes equivariant node features, extracts l=0 (scalar, invariant) components,
    and predicts per-point probability of being a support point.

    Invariance: support labels don't change with rotation, so we only use
    scalar (l=0) features, which are rotation-invariant.

    Args:
        irreps_in: input irreps from backbone
    """

    def __init__(self, irreps_in):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)

        # Extract only scalar components for invariant prediction
        num_scalars = sum(mul for mul, ir in irreps_in if ir.l == 0)

        self.mlp = nn.Sequential(
            nn.Linear(num_scalars, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Store which components are scalars for extraction
        self._scalar_indices = []
        idx = 0
        for mul, ir in irreps_in:
            dim = ir.dim
            if ir.l == 0:
                for m in range(mul):
                    self._scalar_indices.append(idx)
                    idx += dim
            else:
                idx += mul * dim

    def forward(self, node_features):
        """
        Args:
            node_features: (num_nodes, irreps_in.dim) per-node features

        Returns:
            support_prob: (num_nodes,) support probability in [0, 1]
        """
        # Extract scalar (l=0) features only
        scalars = node_features[:, self._scalar_indices]  # (num_nodes, num_scalars)
        logits = self.mlp(scalars).squeeze(-1)  # (num_nodes,)
        return torch.sigmoid(logits)


# ============================================================
# Main Network: Equivariant UprightNet
# ============================================================


class EquivariantUprightNet(nn.Module):
    """
    E(3)-Equivariant Upright Orientation Estimation Network.

    Architecture overview:
        1. Build radius graph from point cloud positions
        2. Compute spherical harmonic edge features Y^l(r_ij / ||r_ij||)
        3. Initialize node features from aggregated edge SH
        4. Apply N equivariant convolution layers (TP + Gate)
        5. Output heads:
           - VMF direction head (global): mu (B,3) + kappa (B,1)
           - Support head (per-node): support_prob (N,)

    Args:
        irreps_hidden:     hidden layer irreps (e.g., "32x0e+16x1o+8x2e")
        lmax:              max spherical harmonic degree for edges
        max_radius:        cutoff radius for graph construction
        num_layers:        number of equivariant convolution layers
        num_radial_basis:  number of radial basis functions
        radial_neurons:    radial MLP hidden size
        num_neighbors:     average neighbor count for normalization
        kappa_init:        initial vMF concentration parameter
    """

    def __init__(
        self,
        irreps_hidden='32x0e+16x1o+8x2e',
        lmax=2,
        max_radius=0.5,
        num_layers=4,
        num_radial_basis=10,
        radial_neurons=64,
        num_neighbors=20.0,
        kappa_init=10.0,
        conv_type='depthwise',
    ):
        super().__init__()
        self.max_radius = max_radius
        self.num_radial_basis = num_radial_basis
        self.num_neighbors = num_neighbors

        # Spherical harmonic irreps for edge features
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        irreps_hidden = o3.Irreps(irreps_hidden)

        # --- Input encoding ---
        # Initial node features from aggregated edge spherical harmonics
        self.input_linear = o3.Linear(self.irreps_sh, irreps_hidden)

        # --- Equivariant backbone ---
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == 'depthwise':
                layer = EfficientConvLayer(
                    irreps_in=irreps_hidden,
                    irreps_out=irreps_hidden,
                    irreps_sh=self.irreps_sh,
                    max_radius=max_radius,
                    num_radial_basis=num_radial_basis,
                    radial_neurons=radial_neurons,
                    num_neighbors=num_neighbors,
                )
            else:  # 'fctp'
                layer = EquivariantConvLayer(
                    irreps_in=irreps_hidden,
                    irreps_out=irreps_hidden,
                    irreps_sh=self.irreps_sh,
                    max_radius=max_radius,
                    num_radial_basis=num_radial_basis,
                    radial_neurons=radial_neurons,
                    num_neighbors=num_neighbors,
                )
            self.conv_layers.append(layer)
            # Update irreps_hidden to actual output irreps (may differ due to Gate)
            irreps_hidden = layer.irreps_out

        self.backbone_out_irreps = irreps_hidden

        # --- Output heads ---
        self.direction_head = VMFDirectionHead(
            irreps_in=irreps_hidden,
            kappa_init=kappa_init,
        )
        self.support_head = SupportHead(irreps_in=irreps_hidden)

    def forward(self, data):
        """
        Args:
            data: PyG Data object with:
                data.pos:   (num_nodes, 3) point positions
                data.batch: (num_nodes,) batch indices

        Returns:
            dict with:
                'direction_mu':    (B, 3) predicted upright direction (unit vector)
                'direction_kappa': (B, 1) vMF concentration (confidence)
                'support_pred':    (num_nodes,) per-point support probability
        """
        pos = data.pos
        batch = data.batch

        # --- Build graph ---
        edge_index = radius_graph(
            pos,
            r=self.max_radius,
            batch=batch,
            max_num_neighbors=64,
        )
        edge_src, edge_dst = edge_index

        # --- Edge features ---
        edge_vec = pos[edge_dst] - pos[edge_src]  # (E, 3)
        edge_length = edge_vec.norm(dim=-1, keepdim=False)  # (E,)

        # Spherical harmonics of edge direction
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh,
            edge_vec,
            normalize=True,
            normalization="component",
        )  # (E, irreps_sh.dim)

        # Radial basis embedding of edge length
        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.num_radial_basis,
            basis="smooth_finite",
            cutoff=True,
        )  # (E, num_radial_basis)
        edge_length_embedded = edge_length_embedded.mul(self.num_radial_basis**0.5)

        # --- Initial node features ---
        # Scatter edge SH to destination nodes → per-node initial feature
        node_features = scatter(
            edge_sh,
            edge_dst,
            dim=0,
            dim_size=pos.shape[0],
            reduce="sum",
        )  # (N, irreps_sh.dim)
        node_features = node_features / (self.num_neighbors**0.5)

        # Project to hidden irreps
        node_features = self.input_linear(node_features)

        # --- Equivariant backbone ---
        for conv_layer in self.conv_layers:
            node_features = conv_layer(
                node_features,
                edge_index,
                edge_sh,
                edge_length_embedded,
            )

        # --- Output heads ---
        mu, kappa = self.direction_head(node_features, batch)
        support_pred = self.support_head(node_features)

        return {
            "direction_mu": mu,  # (B, 3) equivariant
            "direction_kappa": kappa,  # (B, 1) invariant
            "support_pred": support_pred,  # (N,) invariant
        }


def build_equivariant_model(opts):
    """
    Factory function to create EquivariantUprightNet from config options.

    Args:
        opts: argparse namespace with equivariant network parameters

    Returns:
        model: EquivariantUprightNet instance
    """
    model = EquivariantUprightNet(
        irreps_hidden=opts.irreps_hidden,
        lmax=opts.lmax,
        max_radius=opts.max_radius,
        num_layers=opts.equi_layers,
        num_radial_basis=opts.num_radial_basis,
        radial_neurons=opts.radial_neurons,
        num_neighbors=opts.num_neighbors,
        kappa_init=opts.vmf_kappa_init,
        conv_type=getattr(opts, 'conv_type', 'depthwise'),
    )
    return model

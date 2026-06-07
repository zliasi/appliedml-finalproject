"""GNN backends for per-atom magnetic moment prediction.

The node-level backends predict one signed moment per atom: embed the atomic
numbers, pass messages over the radius-cutoff graph, then read out a scalar per
node (no pooling). Their forward returns a tensor of shape [num_atoms].

  cgconv          - crystal graph conv, uses the distance edge feature
  schnetconv      - continuous-filter conv with a radial-basis distance expansion
  graphconv       - simple neighbour-sum conv (connectivity only)
  sageconv        - GraphSAGE mean-aggregation conv (connectivity only)
  gatv2conv       - graph-attention v2, uses the distance edge feature
  gcnconv         - classic GCN (connectivity only)
  transformerconv - graph-transformer attention, uses the distance edge feature
  gineconv        - GIN with edge features, uses the distance edge feature
  nnconv          - edge-conditioned MPNN, uses the distance edge feature
  genconv         - DeeperGCN generalized aggregation (connectivity only)
  gmmconv         - Gaussian-mixture (MoNet) kernels over the distance
  resgatedgraphconv - residual gated graph conv, uses the distance edge feature
  generalconv     - general MPNN with edge features
  pdnconv         - pathfinder discovery network, uses the distance edge feature
  splineconv      - B-spline kernels over the (normalised) distance

dimenet, et, and tensornet are the heavier reference models, all periodic:
  dimenet   - PyG DimeNet++ adapted to per-atom output, made PBC-aware by using
              the graph's periodic edges and minimum-image displacement vectors
              (data.edge_index / data.edge_vec) for distances and angles.
  et        - torchmd-net Equivariant Transformer (the architecture ViSNet
              extends), native PBC via its box-aware neighbour search.
  tensornet - torchmd-net TensorNet, rank-2 Cartesian-tensor equivariant model,
              native PBC.
et and tensornet build their own periodic neighbour graph from positions plus
the per-graph box (data.cell), so they need torchmd-net installed. All three
return per-atom output via a shared readout. Validate against the installed
torchmd-net/PyG versions on the first run (see bench/ smoke test).
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import (
    CGConv, GATv2Conv, GCNConv, GENConv, GeneralConv, GINEConv, GMMConv,
    GraphConv, NNConv, PDNConv, radius_graph, ResGatedGraphConv, SAGEConv,
    SplineConv, TransformerConv,
)
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock

DEFAULT_CONV_LAYERS: int = 3
DEFAULT_CONV_DIM: int = 64
DEFAULT_HIDDEN_LAYERS: int = 0
DEFAULT_NUM_ELEMENTS: int = 118
DEFAULT_ACTIVATION: str = "relu"
DEFAULT_RADIUS_CUTOFF: float = 6.0
EDGE_FEATURE_DIM: int = 1
NUM_GAUSSIANS: int = 50
GMM_KERNEL_SIZE: int = 8
SPLINE_KERNEL_SIZE: int = 8

# DimeNet++ architecture constants
DIMENET_NUM_SPHERICAL: int = 7
DIMENET_NUM_RADIAL: int = 6
DIMENET_BASIS_EMB_SIZE: int = 8
DIMENET_ENVELOPE_EXPONENT: int = 5
DIMENET_NUM_BEFORE_SKIP: int = 1
DIMENET_NUM_AFTER_SKIP: int = 2
# torchmd-net (ET / TensorNet) reference-model constants
TORCHMDNET_NUM_RBF: int = 50
TORCHMDNET_MAX_NEIGHBORS: int = 50

NODE_LEVEL_BACKENDS: list[str] = [
    "cgconv", "schnetconv", "graphconv", "sageconv", "gatv2conv",
    "gcnconv", "transformerconv", "gineconv", "nnconv", "genconv",
    "gmmconv", "resgatedgraphconv", "generalconv", "pdnconv", "splineconv",
]
# heavy reference models, per-atom via custom forward, all periodic. dimenet is
# PyG (made PBC-aware here); et/tensornet are torchmd-net (native PBC).
WRAPPED_BACKENDS: list[str] = ["dimenet", "et", "tensornet"]
CONV_BACKENDS: list[str] = NODE_LEVEL_BACKENDS + WRAPPED_BACKENDS
# node-level backends that consume the scalar distance edge feature, the rest
# use only connectivity
EDGE_FEATURE_BACKENDS: set[str] = {
    "cgconv", "gatv2conv", "transformerconv", "gineconv", "nnconv",
    "gmmconv", "resgatedgraphconv", "generalconv", "pdnconv",
}
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU, "elu": nn.ELU, "sigmoid": nn.Sigmoid,
}


def _build_readout_mlp(
    conv_dim: int, n_hidden_layers: int, activation: nn.Module,
) -> nn.Sequential:
    """Build the per-node readout: maps a node vector to one scalar.

    Args:
        conv_dim: Input and hidden width.
        n_hidden_layers: Hidden layers before the final scalar (>= 0).
        activation: Activation module instance, reused between layers.

    Returns:
        Sequential MLP mapping [num_nodes, conv_dim] -> [num_nodes, 1].
    """
    assert conv_dim > 0, "conv_dim must be positive"
    assert n_hidden_layers >= 0, "n_hidden_layers must be >= 0"
    layers: list[nn.Module] = []
    for _ in range(n_hidden_layers):
        layers.append(nn.Linear(conv_dim, conv_dim))
        layers.append(activation)
    layers.append(nn.Linear(conv_dim, 1))
    return nn.Sequential(*layers)


def _make_conv(
    conv_backend: str, conv_dim: int, radius_cutoff: float,
) -> nn.Module:
    """Build one node-level message-passing layer of the requested backend.

    Args:
        conv_backend: One of NODE_LEVEL_BACKENDS.
        conv_dim: Node feature width.
        radius_cutoff: Cutoff (only used by schnet's interaction block).

    Returns:
        The constructed conv layer.

    Raises:
        ValueError: If conv_backend is not a node-level backend.
    """
    if conv_backend == "cgconv":
        return CGConv(conv_dim, dim=EDGE_FEATURE_DIM, batch_norm=True)
    if conv_backend == "schnetconv":
        return InteractionBlock(
            conv_dim, NUM_GAUSSIANS, conv_dim, radius_cutoff,
        )
    if conv_backend == "graphconv":
        return GraphConv(conv_dim, conv_dim)
    if conv_backend == "sageconv":
        return SAGEConv(conv_dim, conv_dim)
    if conv_backend == "gatv2conv":
        return GATv2Conv(conv_dim, conv_dim, edge_dim=EDGE_FEATURE_DIM)
    if conv_backend == "gcnconv":
        return GCNConv(conv_dim, conv_dim)
    if conv_backend == "transformerconv":
        return TransformerConv(conv_dim, conv_dim, edge_dim=EDGE_FEATURE_DIM)
    if conv_backend == "gineconv":
        gin_mlp = nn.Sequential(
            nn.Linear(conv_dim, conv_dim), nn.ReLU(),
            nn.Linear(conv_dim, conv_dim),
        )
        return GINEConv(gin_mlp, edge_dim=EDGE_FEATURE_DIM)
    if conv_backend == "nnconv":
        # the edge network maps the scalar distance to a [conv_dim, conv_dim]
        # weight matrix per edge
        edge_mlp = nn.Sequential(
            nn.Linear(EDGE_FEATURE_DIM, conv_dim), nn.ReLU(),
            nn.Linear(conv_dim, conv_dim * conv_dim),
        )
        return NNConv(conv_dim, conv_dim, edge_mlp)
    if conv_backend == "genconv":
        return GENConv(conv_dim, conv_dim)
    if conv_backend == "gmmconv":
        return GMMConv(
            conv_dim, conv_dim, dim=EDGE_FEATURE_DIM, kernel_size=GMM_KERNEL_SIZE,
        )
    if conv_backend == "resgatedgraphconv":
        return ResGatedGraphConv(conv_dim, conv_dim, edge_dim=EDGE_FEATURE_DIM)
    if conv_backend == "generalconv":
        return GeneralConv(conv_dim, conv_dim, in_edge_channels=EDGE_FEATURE_DIM)
    if conv_backend == "pdnconv":
        return PDNConv(
            conv_dim, conv_dim, edge_dim=EDGE_FEATURE_DIM, hidden_channels=conv_dim,
        )
    if conv_backend == "splineconv":
        return SplineConv(
            conv_dim, conv_dim, dim=EDGE_FEATURE_DIM, kernel_size=SPLINE_KERNEL_SIZE,
        )
    raise ValueError(f"not a node-level backend: {conv_backend}")


class MagmomGNN(nn.Module):
    """Node-level GNN that predicts one magnetic moment per atom."""

    def __init__(
        self,
        conv_backend: str = "cgconv",
        n_conv_layers: int = DEFAULT_CONV_LAYERS,
        conv_dim: int = DEFAULT_CONV_DIM,
        n_hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
        num_elements: int = DEFAULT_NUM_ELEMENTS,
        activation: str = DEFAULT_ACTIVATION,
        radius_cutoff: float = DEFAULT_RADIUS_CUTOFF,
        element_means: "list[float] | None" = None,
    ) -> None:
        """Build embedding, conv layers, and the per-node readout.

        Args:
            conv_backend: Message-passing backend (one of NODE_LEVEL_BACKENDS).
            n_conv_layers: Number of conv layers (> 0).
            conv_dim: Node feature width (> 0).
            n_hidden_layers: Hidden layers in the readout MLP (>= 0).
            num_elements: Embedding table covers atomic numbers 0..num_elements.
            activation: Activation name (one of ACTIVATIONS).
            radius_cutoff: Edge cutoff, used by schnet's distance expansion.
        """
        super().__init__()
        assert conv_backend in NODE_LEVEL_BACKENDS, (
            f"not a node-level backend: {conv_backend}"
        )
        assert n_conv_layers > 0, "n_conv_layers must be positive"
        assert conv_dim > 0, "conv_dim must be positive"
        assert activation in ACTIVATIONS, f"unknown activation: {activation}"

        self.conv_backend = conv_backend
        self.radius_cutoff = radius_cutoff
        self.embedding = nn.Embedding(num_elements + 1, conv_dim, padding_idx=0)
        self.activation = ACTIVATIONS[activation]()
        self.conv_layers = nn.ModuleList([
            _make_conv(conv_backend, conv_dim, radius_cutoff)
            for _ in range(n_conv_layers)
        ])
        # schnet expands the scalar distance into a radial basis per edge
        self.distance_expansion = (
            GaussianSmearing(0.0, radius_cutoff, NUM_GAUSSIANS)
            if conv_backend == "schnetconv" else None
        )
        self.readout = _build_readout_mlp(
            conv_dim, n_hidden_layers, self.activation,
        )
        # optional fixed per-element bias (residual target): the GNN then only
        # learns the deviation from each element's mean moment
        if element_means is not None:
            self.register_buffer(
                "element_bias",
                torch.tensor(element_means, dtype=torch.float32),
            )
        else:
            self.element_bias = None

    def forward(self, data: Data) -> torch.Tensor:
        """Predict one magnetic moment per atom.

        Args:
            data: PyG batch with x (atomic numbers), edge_index, edge_attr.

        Returns:
            A 1-D tensor of length num_atoms (concatenated across the batch).
        """
        assert hasattr(data, "edge_index"), "Data must have edge_index"
        node_features = self.embedding(data.x.long())

        if self.conv_backend == "schnetconv":
            edge_weight = data.edge_attr.squeeze(-1)
            edge_rbf = self.distance_expansion(edge_weight)
            for block in self.conv_layers:
                node_features = node_features + block(
                    node_features, data.edge_index, edge_weight, edge_rbf,
                )
        elif self.conv_backend == "splineconv":
            # SplineConv needs pseudo-coordinates in [0, 1], scale the distance
            pseudo = (data.edge_attr / self.radius_cutoff).clamp(0.0, 1.0)
            for conv in self.conv_layers:
                node_features = self.activation(
                    conv(node_features, data.edge_index, pseudo)
                )
        else:
            uses_edge_feature = self.conv_backend in EDGE_FEATURE_BACKENDS
            for conv in self.conv_layers:
                if uses_edge_feature:
                    node_features = conv(
                        node_features, data.edge_index, data.edge_attr,
                    )
                else:
                    node_features = conv(node_features, data.edge_index)
                node_features = self.activation(node_features)

        out = self.readout(node_features).squeeze(-1)
        if self.element_bias is not None:
            out = out + self.element_bias[data.x.long().view(-1)]
        return out


class DimeNetBackend(nn.Module):
    """DimeNet++ wrapper, adapted to per-atom output.

    Directional message passing with angular features - the slow, accurate
    angular reference. PyG's DimeNetPlusPlus sums per-atom output blocks into one
    scalar per graph, the forward here returns those per-atom blocks instead.
    Caveat: the neighbour graph is built from positions without periodic
    boundaries.
    """

    def __init__(
        self,
        n_conv_layers: int = DEFAULT_CONV_LAYERS,
        conv_dim: int = DEFAULT_CONV_DIM,
        n_hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
        num_elements: int = DEFAULT_NUM_ELEMENTS,
        activation: str = DEFAULT_ACTIVATION,
        radius_cutoff: float = DEFAULT_RADIUS_CUTOFF,
    ) -> None:
        """Build the DimeNet++ model (see class docstring for caveats)."""
        super().__init__()
        # older DimeNet code paths reference the removed numpy.math alias
        import math
        import numpy as np
        if not hasattr(np, "math"):
            np.math = math
        from torch_geometric.nn import DimeNetPlusPlus

        assert n_conv_layers > 0, "n_conv_layers must be positive"
        assert conv_dim > 0, "conv_dim must be positive"
        self.model = DimeNetPlusPlus(
            hidden_channels=conv_dim,
            out_channels=1,
            num_blocks=n_conv_layers,
            int_emb_size=conv_dim // 2,
            basis_emb_size=DIMENET_BASIS_EMB_SIZE,
            out_emb_channels=conv_dim,
            num_spherical=DIMENET_NUM_SPHERICAL,
            num_radial=DIMENET_NUM_RADIAL,
            cutoff=radius_cutoff,
            envelope_exponent=DIMENET_ENVELOPE_EXPONENT,
            num_before_skip=DIMENET_NUM_BEFORE_SKIP,
            num_after_skip=DIMENET_NUM_AFTER_SKIP,
            num_output_layers=max(1, n_hidden_layers + 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Predict one value per atom (see class docstring)."""
        z = data.x.long().squeeze(-1)
        return self._per_atom(
            z, data.edge_index, data.edge_vec, data.pos.size(0),
        ).squeeze(-1)

    def _per_atom(
        self, z: torch.Tensor, edge_index: torch.Tensor,
        edge_vec: torch.Tensor, num_atoms: int,
    ) -> torch.Tensor:
        """Per-atom DimeNet++ output (the P blocks before the graph reduction).

        Orchestrates the model's own submodules like DimeNetPlusPlus.forward but
        returns the per-atom contributions rather than summing them per graph,
        and is periodic: it consumes the precomputed PBC graph (``edge_index``)
        and the minimum-image displacement vectors (``edge_vec``) instead of
        rebuilding a non-periodic ``radius_graph`` from raw positions. Distances
        and triplet angles are taken from ``edge_vec``, so in-plane periodic
        neighbours are included.
        """
        m = self.model
        # PyG moved triplets from a DimeNet method to a module-level
        # function in newer releases, support both.
        triplets_fn = getattr(m, "triplets", None)
        if triplets_fn is None:
            from torch_geometric.nn.models.dimenet import (
                triplets as triplets_fn,
            )
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets_fn(
            edge_index, num_nodes=z.size(0),
        )
        # edge_vec[m] is the periodic vector src->dst, i.e. pos[col]-pos[row]
        # with the cell offset applied. PyG DimeNet's angle uses
        # pos_ij = pos[i]-pos[j] = edge_vec[idx_ji] and
        # pos_jk = pos[j]-pos[k] = edge_vec[idx_kj] (i=col, j=row).
        dist = edge_vec.norm(dim=-1)
        pos_ij = edge_vec[idx_ji]
        pos_jk = edge_vec[idx_kj]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=-1).norm(dim=-1)
        angle = torch.atan2(b, a)
        rbf = m.rbf(dist)
        sbf = m.sbf(dist, angle, idx_kj)
        x = m.emb(z, rbf, i, j)
        out = m.output_blocks[0](x, rbf, i, num_nodes=num_atoms)
        for interaction, output in zip(
            m.interaction_blocks, m.output_blocks[1:],
        ):
            x = interaction(x, rbf, sbf, idx_kj, idx_ji)
            out = out + output(x, rbf, i, num_nodes=num_atoms)
        return out


class _TorchMDNetBackend(nn.Module):
    """Shared wrapper around a torchmd-net representation model.

    torchmd-net's representation models return per-atom scalar features ``x`` of
    shape [num_atoms, hidden] and have native, box-aware periodic neighbour
    search, so PBC is handled inside the model. A per-node readout maps ``x`` to
    one moment per atom. The per-graph box is ``data.cell`` (shape
    [num_graphs, 3, 3] after batching), passed to the model's forward.
    """

    def __init__(
        self, representation: nn.Module, conv_dim: int,
        n_hidden_layers: int, activation: str,
    ) -> None:
        super().__init__()
        self.model = representation
        self.readout = _build_readout_mlp(
            conv_dim, n_hidden_layers, ACTIVATIONS[activation](),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Predict one moment per atom (PBC handled inside the model)."""
        z = data.x.long().view(-1)
        batch = (
            data.batch if getattr(data, "batch", None) is not None
            else torch.zeros_like(z)
        )
        box = getattr(data, "cell", None)
        # representation models return (x, vec, z, pos, batch); take x.
        x = self.model(z, data.pos, batch, box)[0]
        return self.readout(x).squeeze(-1)


class ETBackend(_TorchMDNetBackend):
    """torchmd-net Equivariant Transformer (native PBC), per-atom output."""

    def __init__(
        self,
        n_conv_layers: int = DEFAULT_CONV_LAYERS,
        conv_dim: int = DEFAULT_CONV_DIM,
        n_hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
        num_elements: int = DEFAULT_NUM_ELEMENTS,
        activation: str = DEFAULT_ACTIVATION,
        radius_cutoff: float = DEFAULT_RADIUS_CUTOFF,
    ) -> None:
        """Build the ET representation (torchmd-net required)."""
        from torchmdnet.models.torchmd_et import TorchMD_ET

        assert n_conv_layers > 0, "n_conv_layers must be positive"
        assert conv_dim > 0, "conv_dim must be positive"
        representation = TorchMD_ET(
            hidden_channels=conv_dim,
            num_layers=n_conv_layers,
            num_rbf=TORCHMDNET_NUM_RBF,
            cutoff_lower=0.0,
            cutoff_upper=radius_cutoff,
            max_z=num_elements + 1,
            max_num_neighbors=TORCHMDNET_MAX_NEIGHBORS,
        )
        super().__init__(representation, conv_dim, n_hidden_layers, activation)


class TensorNetBackend(_TorchMDNetBackend):
    """torchmd-net TensorNet (native PBC), per-atom output."""

    def __init__(
        self,
        n_conv_layers: int = DEFAULT_CONV_LAYERS,
        conv_dim: int = DEFAULT_CONV_DIM,
        n_hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
        num_elements: int = DEFAULT_NUM_ELEMENTS,
        activation: str = DEFAULT_ACTIVATION,
        radius_cutoff: float = DEFAULT_RADIUS_CUTOFF,
    ) -> None:
        """Build the TensorNet representation (torchmd-net required)."""
        from torchmdnet.models.tensornet import TensorNet

        assert n_conv_layers > 0, "n_conv_layers must be positive"
        assert conv_dim > 0, "conv_dim must be positive"
        representation = TensorNet(
            hidden_channels=conv_dim,
            num_layers=n_conv_layers,
            num_rbf=TORCHMDNET_NUM_RBF,
            cutoff_lower=0.0,
            cutoff_upper=radius_cutoff,
            max_z=num_elements + 1,
            max_num_neighbors=TORCHMDNET_MAX_NEIGHBORS,
        )
        super().__init__(representation, conv_dim, n_hidden_layers, activation)


def build_model(config: dict) -> nn.Module:
    """Build a model from a yaml-loaded config dict.

    Args:
        config: Config with ``conv_backend`` and optional hyperparameters.

    Returns:
        A MagmomGNN (node-level) or a dimenet/et/tensornet reference wrapper.

    Raises:
        ValueError: If ``conv_backend`` is not recognised.
    """
    backend = config.get("conv_backend", "cgconv")
    if backend not in CONV_BACKENDS:
        raise ValueError(f"unknown conv_backend: {backend}")

    shared = dict(
        n_conv_layers=config.get("n_conv_layers", DEFAULT_CONV_LAYERS),
        conv_dim=config.get("conv_dim", DEFAULT_CONV_DIM),
        n_hidden_layers=config.get("n_hidden_layers", DEFAULT_HIDDEN_LAYERS),
        num_elements=config.get("num_elements", DEFAULT_NUM_ELEMENTS),
        activation=config.get("activation", DEFAULT_ACTIVATION),
        radius_cutoff=config.get("radius_cutoff", DEFAULT_RADIUS_CUTOFF),
    )
    if backend == "dimenet":
        return DimeNetBackend(**shared)
    if backend == "et":
        return ETBackend(**shared)
    if backend == "tensornet":
        return TensorNetBackend(**shared)
    return MagmomGNN(
        conv_backend=backend,
        element_means=config.get("element_means"),
        **shared,
    )

"""Build PyTorch Geometric graphs from ASE structures.

Radius-cutoff graph builder with embedding mode: node features are
raw atomic numbers (the GNN's embedding layer handles the rest).
Edge features are interatomic distances or unit values (topological).

Supports full-slab graphs (any target) and BFS-based local subgraph
extraction around H (hads target only — set ``n_hops`` in
``build_local_graph``). Target value is passed via the generic
``target`` kwarg, it ends up in ``Data.y``.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ase.atoms import Atoms
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

DEFAULT_RADIUS_CUTOFF: float = 6.0
DEFAULT_MAX_NEIGHBORS: int = 50
MINIMUM_DISTANCE_ANG: float = 1e-6
LOG_PROGRESS_DIVISOR: int = 10
HYDROGEN_ATOMIC_NUMBER: int = 1
EDGE_MODE_DISTANCE: str = "distance"
EDGE_MODE_TOPOLOGICAL: str = "topological"
VALID_EDGE_MODES: list[str] = [
    EDGE_MODE_DISTANCE, EDGE_MODE_TOPOLOGICAL,
]


def _target_to_tensor(target: "float | np.ndarray") -> torch.Tensor:
    """Convert a target into a ``Data.y`` tensor.

    A scalar becomes shape ``[1]`` (graph-level), a per-atom sequence becomes
    shape ``[num_atoms]`` (node-level). PyG concatenates either across a batch,
    so the loss and metrics handle both without changes.

    Args:
        target: Scalar regression target, or a per-atom array/list.

    Returns:
        Float32 tensor of shape ``[1]`` for a scalar or ``[num_atoms]`` for a
        per-atom sequence.
    """
    if isinstance(target, (list, tuple, np.ndarray)):
        values = np.asarray(target, dtype=np.float32).reshape(-1)
        assert values.size > 0, "Per-atom target must not be empty"
        return torch.from_numpy(values)
    return torch.tensor([target], dtype=torch.float32)


class GraphBuilder:
    """Build PyTorch Geometric graphs from ASE structures.

    Attributes:
        radius_cutoff: Cutoff radius for edges (Angstroms)
        max_neighbors: Maximum number of neighbors per atom
        edge_mode: Edge feature mode (distance or topological)
    """

    def __init__(
        self,
        radius_cutoff: float = DEFAULT_RADIUS_CUTOFF,
        max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
        edge_mode: str = EDGE_MODE_DISTANCE,
    ):
        """Initialize GraphBuilder.

        Args:
            radius_cutoff: Cutoff radius for neighbor detection
            max_neighbors: Maximum neighbors per node
            edge_mode: "distance" or "topological"

        Raises:
            ValueError: If edge_mode not recognized
        """
        assert radius_cutoff > 0, "Cutoff must be positive"
        assert max_neighbors > 0, "Max neighbors must be positive"
        if edge_mode not in VALID_EDGE_MODES:
            raise ValueError(
                f"edge_mode must be one of {VALID_EDGE_MODES}"
            )

        self.radius_cutoff = radius_cutoff
        self.max_neighbors = max_neighbors
        self.edge_mode = edge_mode

    def build_graph(
        self,
        atoms: Atoms,
        target: "Optional[float | np.ndarray]" = None,
    ) -> Data:
        """Build PyTorch Geometric graph from ASE structure.

        Args:
            atoms: ASE Atoms object
            target: Scalar (graph-level) or per-atom array (node-level),
                optional

        Returns:
            PyG Data with x=atomic_numbers, edge_index, edge_attr, y
        """
        assert len(atoms) > 0, "Atoms object must not be empty"

        atomic_numbers = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.long,
        )
        positions = torch.tensor(
            atoms.get_positions(), dtype=torch.float32,
        )

        edge_index, edge_attr, edge_vec, cell_offsets = self._build_edges(atoms)

        target_tensor = None
        if target is not None:
            target_tensor = _target_to_tensor(target)

        # cell stored as [1, 3, 3] so a PyG batch collates to [num_graphs, 3, 3],
        # the per-graph box shape the periodic models (torchmd-net) expect.
        cell = torch.tensor(
            np.asarray(atoms.get_cell()), dtype=torch.float32,
        ).unsqueeze(0)

        graph = Data(
            x=atomic_numbers,
            pos=positions,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_vec=edge_vec,
            cell_offsets=cell_offsets,
            cell=cell,
            y=target_tensor,
        )
        assert graph.x.size(0) == len(atoms), "Node count mismatch"
        return graph

    def build_local_graph(
        self,
        atoms: Atoms,
        n_hops: int,
        target: "Optional[float | np.ndarray]" = None,
        h_index: Optional[int] = None,
    ) -> Data:
        """Build local subgraph around H via BFS (hads-only path).

        Args:
            atoms: ASE Atoms with H adsorbate
            n_hops: BFS hops from H atom
            target: Scalar (graph-level), or a per-atom array which is
                subset to the selected subgraph atoms (node-level), optional
            h_index: H atom index (auto-detected if None)

        Returns:
            PyG Data with subgraph nodes and edges
        """
        assert len(atoms) > 0, "Atoms must not be empty"
        if n_hops < 1:
            raise ValueError(
                f"n_hops must be >= 1, got {n_hops}"
            )

        if h_index is None:
            h_index = self._find_hydrogen(atoms)

        selected = self._select_bfs_neighborhood(
            atoms, h_index, n_hops,
        )
        sub_edge_index, sub_edge_attr, _, _ = self._build_edges(
            atoms[selected],
        )
        if target is None:
            target_tensor = None
        elif isinstance(target, (list, tuple, np.ndarray)):
            target_tensor = _target_to_tensor(np.asarray(target)[selected])
        else:
            target_tensor = _target_to_tensor(target)
        atomic_numbers = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.long,
        )
        positions = torch.tensor(
            atoms.get_positions(), dtype=torch.float32,
        )
        graph = Data(
            x=atomic_numbers[selected],
            pos=positions[selected],
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr,
            y=target_tensor,
        )
        assert graph.x.size(0) == len(selected)
        return graph

    def _select_bfs_neighborhood(
        self,
        atoms: Atoms,
        h_index: int,
        n_hops: int,
    ) -> list[int]:
        """Find atom indices within n_hops of H via BFS."""
        full_edge_index, _, _, _ = self._build_edges(atoms)
        adjacency = self._edge_index_to_adjacency(
            full_edge_index, len(atoms),
        )
        selected = self._bfs_neighborhood(
            adjacency, h_index, n_hops,
        )
        assert len(selected) > 0, "BFS returned empty set"
        return selected

    @staticmethod
    def _find_hydrogen(atoms: Atoms) -> int:
        """Find index of H atom in structure.

        Raises:
            ValueError: If no H atom found
        """
        atomic_numbers = atoms.get_atomic_numbers()
        h_indices = [
            i for i, z in enumerate(atomic_numbers)
            if z == HYDROGEN_ATOMIC_NUMBER
        ]
        if len(h_indices) == 0:
            raise ValueError("No hydrogen atom found")
        assert len(h_indices) == 1, (
            f"Expected 1 H atom, found {len(h_indices)}"
        )
        return h_indices[0]

    @staticmethod
    def _edge_index_to_adjacency(
        edge_index: torch.Tensor,
        n_atoms: int,
    ) -> list[list[int]]:
        """Convert edge_index to adjacency list."""
        assert edge_index.dim() == 2, "edge_index must be 2D"
        adjacency: list[list[int]] = [
            [] for _ in range(n_atoms)
        ]
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            adjacency[s].append(d)
        return adjacency

    @staticmethod
    def _bfs_neighborhood(
        adjacency: list[list[int]],
        start: int,
        n_hops: int,
    ) -> list[int]:
        """BFS from start node, return all within n_hops."""
        assert 0 <= start < len(adjacency), (
            "Start index out of range"
        )
        assert n_hops >= 1, "n_hops must be >= 1"

        visited: set[int] = {start}
        queue: deque[tuple[int, int]] = deque([(start, 0)])

        while queue:
            node, depth = queue.popleft()
            if depth >= n_hops:
                continue
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        result = sorted(visited)
        assert start in result, "Start node must be in result"
        return result

    def _build_edges(
        self,
        atoms: Atoms,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create edges from distance cutoff with PBC support.

        Also returns the periodic edge displacement vectors and integer cell
        offsets, so models that need real geometry under PBC (DimeNet's angles)
        can use the minimum-image vectors instead of raw position differences.
        ``edge_index`` and ``edge_attr`` are unchanged from the distance-only
        build, so the message-passing backends are unaffected.

        Returns:
            Tuple of (edge_index, edge_attr, edge_vec, cell_offsets) where
            ``edge_vec[m] = pos[dst] + cell_offsets[m] @ cell - pos[src]`` (the
            periodic vector from src to dst) and ``edge_attr`` is its norm.
        """
        assert len(atoms) > 0, "Atoms must not be empty"

        src, dst, dist, vec, offset = neighbor_list(
            "ijdDS", atoms, cutoff=self.radius_cutoff,
        )

        n_max = self.max_neighbors * len(atoms)
        if len(src) > n_max:
            sorted_idx = np.argsort(dist)[:n_max]
            src = src[sorted_idx]
            dst = dst[sorted_idx]
            dist = dist[sorted_idx]
            vec = vec[sorted_idx]
            offset = offset[sorted_idx]

        edge_index = torch.tensor(
            np.stack([src, dst]), dtype=torch.long,
        )

        if len(src) > 0:
            edge_vec = torch.tensor(vec, dtype=torch.float32)
            cell_offsets = torch.tensor(offset, dtype=torch.long)
            if self.edge_mode == EDGE_MODE_TOPOLOGICAL:
                edge_attr = torch.ones(len(src), 1)
            else:
                edge_attr = torch.tensor(
                    dist, dtype=torch.float32,
                ).unsqueeze(1)
        else:
            edge_vec = torch.zeros(0, 3)
            cell_offsets = torch.zeros(0, 3, dtype=torch.long)
            edge_attr = torch.zeros(0, 1)

        return edge_index, edge_attr.float(), edge_vec, cell_offsets


def build_graphs(
    atoms_list: list[Atoms],
    targets: Optional[list[float]] = None,
    radius_cutoff: float = DEFAULT_RADIUS_CUTOFF,
    edge_mode: str = EDGE_MODE_DISTANCE,
) -> list[Data]:
    """Build graphs for multiple structures (convenience helper).

    Args:
        atoms_list: List of ASE Atoms objects
        targets: Optional list of target values
        radius_cutoff: Cutoff for neighbor detection
        edge_mode: Edge feature mode

    Returns:
        List of PyTorch Geometric Data objects
    """
    assert len(atoms_list) > 0, "Must provide at least one"
    builder = GraphBuilder(
        radius_cutoff=radius_cutoff, edge_mode=edge_mode,
    )
    graphs: list[Data] = []

    for index, atoms in enumerate(atoms_list):
        target = targets[index] if targets is not None else None
        graph = builder.build_graph(atoms, target=target)
        graphs.append(graph)

        step = max(1, len(atoms_list) // LOG_PROGRESS_DIVISOR)
        if (index + 1) % step == 0:
            logger.info(
                "Built %d/%d graphs",
                index + 1, len(atoms_list),
            )

    logger.info("Built %d graphs total", len(graphs))
    assert len(graphs) == len(atoms_list), "Count mismatch"
    return graphs

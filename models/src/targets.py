"""Target-spec interface: how a target plugs into the shared pipeline.

Each prediction target (currently hads = H adsorption energy, wf =
work function) exposes a ``TARGET_SPEC`` instance from
``models/targets/<name>/__init__.py``. Shared scripts dispatch on
``--target NAME`` and look up the spec by name.

The spec carries:
  * static metadata (json filename, value field, checkpoint prefix,
    SOAP H-inclusion flag);
  * callables that handle the parts of the pipeline that legitimately
    differ between targets:
        - ``build_gnn(config) -> nn.Module``
        - ``iter_work_items(json_path, dataset_dir) -> Iterator[WorkItem]``

This keeps the shared 01-build-graphs / 03-train / 06-evaluate
scripts target-agnostic; per-target deltas live entirely under
``models/targets/<name>/``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch.nn as nn


@dataclass
class WorkItem:
    """One graph-building task produced by a target's iter_work_items.

    Attributes:
        traj_path: Path to the ASE .traj file to read, or None when the
            structure is supplied directly via ``atoms``.
        atoms: Optional in-memory ASE Atoms, for targets whose structures
            come from an aggregate file (e.g. h5) rather than a .traj. The
            build-graphs worker uses this when ``traj_path`` is None.
        target_value: Scalar regression target (graph-level), or a per-atom
            array of length num_atoms (node-level).
        group_key: Partitioning unit. Items sharing a group_key go
            into the same train/val/test split (``comp_id`` for wf so
            all variants of one composition stay together; for hads
            ``comp_id/variant`` so all sites of one slab stay together).
        comp_id: Composition id; used by the script to look up
            ``composition_type`` from compositions.csv for the
            stratified split.
        metadata: Free-form extras to attach to the PyG Data object
            (e.g. ``{"variant": ..., "site": ...}``).
    """

    target_value: float | np.ndarray
    group_key: str
    comp_id: str
    traj_path: Optional[Path] = None
    atoms: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetSpec:
    """Static + behavioural contract for one target.

    Attributes:
        name: Short target identifier (``"hads"``, ``"wf"``). Appears
            in run-dir names (``hads-fcc12-v1p1``) and checkpoint stems.
        json_filename: Name of the per-dataset JSON file under
            ``datasets/<dataset>/data/``.
        value_field: Key into a flat record for the regression target.
        include_hydrogen: SOAP species list flag (True for hads
            adsorbed slabs, False for wf bare slabs).
        checkpoint_prefix: Leading token of checkpoint filenames
            (matches ``name`` in current convention).
        build_gnn: Factory taking a yaml-loaded config dict and
            returning an ``nn.Module``. Implemented in
            ``targets/<name>/gnn.py``.
        iter_work_items: Callable that reads the per-dataset JSON +
            dataset root and yields ``WorkItem`` records. Encapsulates
            target-specific record schemas, outlier filters, grouping
            keys, traj-path resolution, and local/full-subgraph choice.
        node_level: If True the target is one value per atom (Data.y has
            shape [num_atoms]); if False, one value per graph.
    """

    name: str
    json_filename: str
    value_field: str
    include_hydrogen: bool
    checkpoint_prefix: str

    build_gnn: Callable[[dict], nn.Module]
    iter_work_items: Callable[[Path, Path], Iterable[WorkItem]]
    node_level: bool = False


def load_target_spec(name: str) -> TargetSpec:
    """Import ``models/targets/<name>/`` and return its TARGET_SPEC.

    Args:
        name: Target identifier (e.g. ``"hads"``, ``"wf"``).

    Returns:
        The ``TARGET_SPEC`` instance exported by the target package.

    Raises:
        ImportError: If the target package or its TARGET_SPEC is
            missing.
    """
    import importlib

    module = importlib.import_module(f"targets.{name}")
    spec = getattr(module, "TARGET_SPEC", None)
    if spec is None:
        raise ImportError(
            f"targets/{name}/__init__.py must export TARGET_SPEC"
        )
    assert isinstance(spec, TargetSpec), (
        f"targets/{name}/TARGET_SPEC must be a TargetSpec, "
        f"got {type(spec)}"
    )
    return spec

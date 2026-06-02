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
from typing import Any, Callable, Iterable

import torch.nn as nn


@dataclass
class WorkItem:
    """One graph-building task produced by a target's iter_work_items.

    Attributes:
        traj_path: Absolute path to the ASE .traj file to read.
        target_value: Scalar regression target (eV).
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

    traj_path: Path
    target_value: float
    group_key: str
    comp_id: str
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
    """

    name: str
    json_filename: str
    value_field: str
    include_hydrogen: bool
    checkpoint_prefix: str

    build_gnn: Callable[[dict], nn.Module]
    iter_work_items: Callable[[Path, Path], Iterable[WorkItem]]


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

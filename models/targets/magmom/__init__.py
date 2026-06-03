"""magmom target: per-atom magnetic moment prediction (node-level).

Exports TARGET_SPEC for the shared pipeline. iter_work_items reads the
atom-indexed aggregate (magmoms.h5) and yields one WorkItem per slab, carrying
the structure in-memory (no .traj) and the per-atom moments as the target.
"""

import logging
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from ase import Atoms

from src.targets import TargetSpec, WorkItem

from .gnn import build_model

logger = logging.getLogger(__name__)

H5_FILENAME: str = "magmoms.h5"


def iter_work_items(h5_path: Path, dataset_dir: Path) -> Iterable[WorkItem]:
    """Yield one WorkItem per slab in magmoms.h5.

    The aggregate is grouped ``/<comp_id>/<variant>/`` with atom-ordered
    datasets (magmoms, atomic_numbers, positions, cell, pbc). The structure is
    rebuilt as an ASE Atoms and carried on the WorkItem; the per-atom moments
    are the (node-level) target. Group key is ``comp_id`` so all variants of one
    composition stay in the same split.

    Args:
        h5_path: Path to magmoms.h5.
        dataset_dir: Dataset root (unused; the structure lives in the h5).

    Yields:
        One WorkItem per slab.
    """
    n_slabs = 0
    with h5py.File(h5_path, "r") as handle:
        for comp_id in handle:
            for variant_name, group in handle[comp_id].items():
                magmoms = np.asarray(group["magmoms"], dtype=np.float32)
                # slabs are periodic in-plane; fall back if the aggregate was
                # written without a pbc dataset
                pbc = (np.asarray(group["pbc"]) if "pbc" in group
                       else [True, True, False])
                atoms = Atoms(
                    numbers=np.asarray(group["atomic_numbers"]),
                    positions=np.asarray(group["positions"]),
                    cell=np.asarray(group["cell"]),
                    pbc=pbc,
                )
                assert len(atoms) == magmoms.shape[0], (
                    f"atom/moment count mismatch in {comp_id}/{variant_name}"
                )
                yield WorkItem(
                    target_value=magmoms,
                    group_key=comp_id,
                    comp_id=comp_id,
                    atoms=atoms,
                    metadata={"variant": variant_name},
                )
                n_slabs += 1
    logger.info("magmom: yielded %d slabs from %s", n_slabs, h5_path)


TARGET_SPEC = TargetSpec(
    name="magmom",
    json_filename=H5_FILENAME,
    value_field="magmoms",
    include_hydrogen=False,
    checkpoint_prefix="magmom",
    build_gnn=build_model,
    iter_work_items=iter_work_items,
    node_level=True,
)

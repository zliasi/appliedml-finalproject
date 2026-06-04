"""magmom target: per-atom magnetic moment prediction (node-level).

Exports TARGET_SPEC for the shared pipeline. iter_work_items reads the
atom-indexed aggregate (magmoms.h5) and yields one WorkItem per slab, carrying
the structure in-memory (no .traj) and the per-atom moments as the target.

The target is the absolute moment |m| by default (the field standard for
magmom-initialisation, e.g. CHGNet, since the DFT sign is largely an artifact
of the SCF spin initialisation). Set ``MAGMOM_SIGNED=1`` to keep the DFT signs.
"""

import logging
import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from ase import Atoms

from src.targets import TargetSpec, WorkItem

from .gnn import build_model

logger = logging.getLogger(__name__)

H5_FILENAME: str = "magmoms.h5"
SIGNED_ENV_VAR: str = "MAGMOM_SIGNED"


def iter_work_items(h5_path: Path, dataset_dir: Path) -> Iterable[WorkItem]:
    """Yield one WorkItem per slab in magmoms.h5.

    The aggregate is grouped ``/<comp_id>/<variant>/`` with atom-ordered
    datasets (magmoms, atomic_numbers, positions, cell, pbc). The structure is
    rebuilt as an ASE Atoms and carried on the WorkItem, the per-atom moments
    are the (node-level) target. The target is the absolute moment |m| unless
    ``MAGMOM_SIGNED=1`` is set. Group key is ``comp_id`` so all variants of one
    composition stay in the same split.

    Args:
        h5_path: Path to magmoms.h5.
        dataset_dir: Dataset root (unused, the structure lives in the h5).

    Yields:
        One WorkItem per slab.
    """
    signed = os.environ.get(SIGNED_ENV_VAR, "").strip().lower() in {
        "1", "true", "yes",
    }
    target_mode = "signed" if signed else "absolute"
    logger.info(
        "magmom: target is %s%s",
        "signed moments" if signed else "absolute moments |m|",
        "" if signed else f" (set {SIGNED_ENV_VAR}=1 for signed)",
    )
    n_slabs = 0
    with h5py.File(h5_path, "r") as handle:
        for comp_id in handle:
            for variant_name, group in handle[comp_id].items():
                magmoms = np.asarray(group["magmoms"], dtype=np.float32)
                if not signed:
                    magmoms = np.abs(magmoms)
                # slabs are periodic in-plane, fall back if the aggregate was
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
                    metadata={
                        "variant": variant_name,
                        "magmom_target": target_mode,
                    },
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

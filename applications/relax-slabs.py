"""Example: warm-start a GPAW slab relaxation with predicted magnetic moments.

Reads a slab, predicts its initial magnetic moments with a trained GNN, sets
them on the atoms, and runs a short GPAW relaxation. The point is the warm start:
seeding the SCF with reasonable per-atom moments converges faster (and to the
right magnetic state) than GPAW's default guess.

This is an example, the GPAW settings here are minimal placeholders.

    python applications/relax-slabs.py --checkpoint best.pt slab.traj
"""

import argparse
import importlib.util
from pathlib import Path
from typing import Callable

from ase.io import read
from ase.optimize import LBFGS

HERE = Path(__file__).resolve().parent

GPAW_PW_CUTOFF: int = 350
GPAW_KPTS_DENSITY: float = 3.5
DEFAULT_FMAX: float = 0.05
DEFAULT_MAX_STEPS: int = 50


def _load_predict_magmoms() -> Callable:
    """Import ``predict_magmoms`` from the hyphenated infer-magmoments.py.

    The sibling module's filename has a hyphen, so it cannot be imported with a
    normal ``import`` statement. Load it by path instead.
    """
    spec = importlib.util.spec_from_file_location(
        "infer_magmoments", HERE / "infer-magmoments.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.predict_magmoms


def main() -> None:
    """Predict initial moments for a slab and run a short GPAW relaxation."""
    parser = argparse.ArgumentParser(description="warm-started GPAW relaxation")
    parser.add_argument("structure", type=Path, help="slab .traj file")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fmax", type=float, default=DEFAULT_FMAX)
    parser.add_argument("--steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--txt", type=Path, default=Path("relax.txt"))
    args = parser.parse_args()

    predict_magmoms = _load_predict_magmoms()
    atoms = read(args.structure)

    # the warm start: predicted per-atom moments become the SCF's initial guess
    moments = predict_magmoms(atoms, args.checkpoint)
    atoms.set_initial_magnetic_moments(moments)
    print(f"seeded {len(moments)} initial moments (muB)")

    # GPAW imported here so the rest of the module loads without a DFT install
    from gpaw import GPAW, PW
    atoms.calc = GPAW(
        mode=PW(GPAW_PW_CUTOFF),
        xc="RPBE",
        kpts={"density": GPAW_KPTS_DENSITY},
        spinpol=True,
        txt=str(args.txt),
    )
    LBFGS(atoms).run(fmax=args.fmax, steps=args.steps)
    print(f"relaxed energy: {atoms.get_potential_energy():.4f} eV")


if __name__ == "__main__":
    main()

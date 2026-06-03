"""Predict per-atom magnetic moments for a slab from a trained checkpoint.

``predict_magmoms(atoms, checkpoint)`` returns one signed moment per atom.
Loading the checkpoint is the expensive part, so ``load_predictor`` returns a
reusable ``predict(atoms)`` callable when predicting many slabs.

CLI:
    python applications/infer-magmoments.py --checkpoint best.pt slab.traj
"""

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from ase import Atoms
from ase.io import read

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

from torch_geometric.data import Batch  # noqa: E402

from src.graph_builder import GraphBuilder  # noqa: E402
from targets.magmom.gnn import build_model  # noqa: E402

DEFAULT_RADIUS_CUTOFF: float = 6.0


def load_predictor(
    checkpoint: Path, device: str = "cpu",
) -> Callable[[Atoms], np.ndarray]:
    """Load a checkpoint once; return ``predict(atoms) -> per-atom moments``.

    Args:
        checkpoint: Path to a trained .pt checkpoint ({model_state_dict,
            config, ...}).
        device: Device to run inference on (cpu for the warm-start use case).

    Returns:
        A callable mapping an ASE Atoms to a float array of per-atom moments.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    cutoff = float(config.get("radius_cutoff", DEFAULT_RADIUS_CUTOFF))
    builder = GraphBuilder(radius_cutoff=cutoff)

    def predict(atoms: Atoms) -> np.ndarray:
        """Predict per-atom magnetic moments for one slab."""
        assert len(atoms) > 0, "structure has no atoms"
        graph = builder.build_graph(atoms)
        batch = Batch.from_data_list([graph]).to(device)
        with torch.no_grad():
            moments = model(batch)
        return moments.cpu().numpy()

    return predict


def predict_magmoms(
    atoms: Atoms, checkpoint: Path, device: str = "cpu",
) -> np.ndarray:
    """Predict per-atom magnetic moments for one slab (loads the checkpoint).

    For many slabs, use ``load_predictor`` to load the checkpoint once.

    Args:
        atoms: ASE Atoms for the slab.
        checkpoint: Path to a trained .pt checkpoint.
        device: Inference device.

    Returns:
        Float array of one signed moment per atom.
    """
    return load_predictor(checkpoint, device)(atoms)


def main() -> None:
    """Predict and print/save the moments for one structure file."""
    parser = argparse.ArgumentParser(description="predict per-atom magmoms")
    parser.add_argument("structure", type=Path, help="slab .traj file")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out", type=Path, default=None, help="save moments as .npy",
    )
    args = parser.parse_args()

    atoms = read(args.structure)
    moments = predict_magmoms(atoms, args.checkpoint, args.device)
    print(f"{len(moments)} atoms; predicted moments (muB):")
    print(np.array2string(moments, precision=3, max_line_width=100))
    if args.out is not None:
        np.save(args.out, moments)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

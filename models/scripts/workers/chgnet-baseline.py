"""CHGNet zero-shot baseline for the magmom target.

Runs the pretrained CHGNet (Deng et al. 2023), which predicts site magnetic
moments out of the box, on the same test split the GNNs use, and scores its
predictions against the DFT |m|. CHGNet is trained on absolute moments from
~1.5M Materials Project structures, so this is a strong "free" reference and a
direct check of whether the from-scratch GNNs beat a large pretrained model.
Writes baselines-chgnet.json (+ parity/error plots) into the run's eval/ dir,
which 05-submit-evaluate.sh --all merges into the ranking.

    python scripts/workers/chgnet-baseline.py --target magmom --dataset magmom21-v0p1
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(MODELS_ROOT))
REPO_ROOT = MODELS_ROOT.parent

from src.metrics import (  # noqa: E402
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from src.plots import plot_error_distribution, plot_parity  # noqa: E402
from src.splits import (  # noqa: E402
    load_composition_index,
    parse_dataset_token,
    stratified_split,
)
from src.targets import load_target_spec  # noqa: E402

PROP_LABEL: str = "magnetic moment"
UNIT: str = "muB"
PROGRESS_EVERY: int = 200


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE/RMSE/R2 over all atoms."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def build_test_items(spec, dataset_dir: Path, json_path: Path) -> list:
    """Materialise the deterministic test split (same atoms as the GNNs)."""
    all_items = list(spec.iter_work_items(json_path, dataset_dir))
    comp_types, _ = load_composition_index(dataset_dir)
    by_group: dict[str, list] = defaultdict(list)
    for wi in all_items:
        by_group[wi.group_key].append(wi)
    _, _, test_keys = stratified_split(sorted(by_group.keys()), comp_types)
    return [wi for k in test_keys for wi in by_group[k]]


def by_element(z: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-element metrics, keyed by chemical symbol."""
    from ase.data import chemical_symbols
    out: dict[str, dict] = {}
    for elem in sorted(set(int(v) for v in z)):
        mask = z == elem
        out[chemical_symbols[elem]] = score(y_true[mask], y_pred[mask])
    return out


def main() -> None:
    """Run CHGNet on the test split and write metrics + plots."""
    parser = argparse.ArgumentParser(description="CHGNet magmom baseline")
    parser.add_argument("--target", default="magmom")
    parser.add_argument("--dataset", required=True, help="e.g. magmom21-v0p1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--wandb", action="store_true",
        help="log the chgnet baseline to Weights & Biases",
    )
    args = parser.parse_args()

    spec = load_target_spec(args.target)
    dataset_name, _ = parse_dataset_token(args.dataset)
    dataset_dir = REPO_ROOT / "datasets" / dataset_name
    json_path = dataset_dir / "data" / spec.json_filename
    items = build_test_items(spec, dataset_dir, json_path)
    print(f"test slabs: {len(items)}")

    from chgnet.model import CHGNet
    from pymatgen.io.ase import AseAtomsAdaptor

    chgnet = CHGNet.load(use_device=args.device)
    structures = [AseAtomsAdaptor.get_structure(wi.atoms) for wi in items]
    preds = chgnet.predict_structure(structures, batch_size=args.batch_size)

    y_pred_parts, y_true_parts, z_parts = [], [], []
    for wi, pred in zip(items, preds):
        m = np.abs(np.asarray(pred["m"], dtype=np.float32).reshape(-1))
        y = np.asarray(wi.target_value, dtype=np.float32).reshape(-1)
        assert m.shape == y.shape, "CHGNet/target atom count mismatch"
        y_pred_parts.append(m)
        y_true_parts.append(y)
        z_parts.append(np.asarray(wi.atoms.get_atomic_numbers()))
    y_pred = np.concatenate(y_pred_parts)
    y_true = np.concatenate(y_true_parts)
    z = np.concatenate(z_parts)

    overall = score(y_true, y_pred)
    run_dir = REPO_ROOT / "models" / "runs" / f"{spec.name}-{args.dataset}"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "target": args.target,
        "dataset": args.dataset,
        "n_test_atoms": int(y_true.shape[0]),
        "metrics": {"chgnet": overall},
        "by_element": by_element(z, y_true, y_pred),
    }
    (eval_dir / "baselines-chgnet.json").write_text(json.dumps(payload, indent=2))

    metrics = {**overall, "n_samples": int(y_true.shape[0])}
    plot_parity(
        y_pred, y_true, eval_dir / "parity-baseline-chgnet.png",
        metrics, PROP_LABEL, UNIT,
    )
    plot_error_distribution(
        y_pred, y_true, eval_dir / "error-dist-baseline-chgnet.png",
        PROP_LABEL, UNIT,
    )
    if args.wandb:
        import os
        import wandb
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=run_dir.name, name="baseline-chgnet",
            config={"baseline": "chgnet", "dataset": args.dataset},
        )
        wandb.log({
            "test_mae": overall["mae"],
            "test_rmse": overall["rmse"],
            "test_r2": overall["r2"],
        })
        wandb.finish()
    print(
        f"chgnet (test, {y_true.shape[0]} atoms): "
        f"MAE={overall['mae']:.4f} RMSE={overall['rmse']:.4f} R2={overall['r2']:.4f}"
    )
    print(f"wrote {eval_dir / 'baselines-chgnet.json'}")


if __name__ == "__main__":
    main()
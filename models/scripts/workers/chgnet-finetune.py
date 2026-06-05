"""Fine-tune CHGNet on magmom21, then score it on the held-out test split.

Zero-shot CHGNet is out-of-domain (bulk Materials Project crystals vs metal
alloy slabs). This fine-tunes its magmom head on the magmom21 train+val slabs
and evaluates on the SAME (seeded) test split the GNNs use, so the result is
directly comparable. Writes baselines-chgnet-ft.json (+ parity/error plots),
which 05-submit-evaluate.sh merges as ``baseline-chgnet-ft``.

UNVALIDATED against your installed chgnet version: check the StructureData /
Trainer / predict_structure calls on the first run and adjust if the API differs.

    python scripts/workers/chgnet-finetune.py --target magmom --dataset magmom21-v0p1
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


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE/RMSE/R2 over all atoms."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def build_splits(spec, dataset_dir: Path, json_path: Path):
    """Return (train, val, test) WorkItem lists from the seeded split."""
    all_items = list(spec.iter_work_items(json_path, dataset_dir))
    comp_types, _ = load_composition_index(dataset_dir)
    by_group: dict[str, list] = defaultdict(list)
    for wi in all_items:
        by_group[wi.group_key].append(wi)
    train_k, val_k, test_k = stratified_split(sorted(by_group.keys()), comp_types)
    pick = lambda keys: [wi for k in keys for wi in by_group[k]]
    return pick(train_k), pick(val_k), pick(test_k)


def main() -> None:
    """Fine-tune CHGNet on magmom and evaluate on the test split."""
    parser = argparse.ArgumentParser(description="fine-tune CHGNet on magmom")
    parser.add_argument("--target", default="magmom")
    parser.add_argument("--dataset", required=True, help="e.g. magmom21-v0p1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    spec = load_target_spec(args.target)
    dataset_name, _ = parse_dataset_token(args.dataset)
    dataset_dir = REPO_ROOT / "datasets" / dataset_name
    json_path = dataset_dir / "data" / spec.json_filename
    train_items, val_items, test_items = build_splits(spec, dataset_dir, json_path)
    pool = train_items + val_items
    print(f"fine-tune pool={len(pool)}  test={len(test_items)}")

    from chgnet.data.dataset import StructureData, get_train_val_test_loader
    from chgnet.model import CHGNet
    from chgnet.trainer import Trainer
    from pymatgen.io.ase import AseAtomsAdaptor

    # magmom-only fine-tuning: real magmoms, dummy energies/forces (not in loss)
    structures = [AseAtomsAdaptor.get_structure(wi.atoms) for wi in pool]
    magmoms = [
        np.abs(np.asarray(wi.target_value, dtype=float)).tolist() for wi in pool
    ]
    energies = [0.0] * len(pool)
    forces = [np.zeros((len(s), 3)).tolist() for s in structures]
    dataset = StructureData(
        structures=structures, energies=energies, forces=forces, magmoms=magmoms,
    )
    train_loader, val_loader, _ = get_train_val_test_loader(
        dataset, batch_size=args.batch_size, train_ratio=0.9, val_ratio=0.1,
    )

    chgnet = CHGNet.load()
    # Train magmom only, but keep energy in the target set with zero weight:
    # chgnet's Trainer.save_checkpoint hard-codes the energy history, so a
    # magmom-only ("m") run crashes with KeyError 'e'. Zero energy/force/stress
    # weight means only magmom gradients update the model.
    trainer = Trainer(
        model=chgnet, targets="em",
        energy_loss_ratio=0.0, force_loss_ratio=0.0,
        stress_loss_ratio=0.0, mag_loss_ratio=1.0,
        optimizer="Adam", criterion="MSE",
        epochs=args.epochs, learning_rate=args.lr, use_device=args.device,
    )
    trainer.train(train_loader, val_loader)
    model = trainer.model

    test_structs = [AseAtomsAdaptor.get_structure(wi.atoms) for wi in test_items]
    preds = model.predict_structure(test_structs, batch_size=args.batch_size)
    yp_parts, yt_parts = [], []
    for wi, pred in zip(test_items, preds):
        m = np.abs(np.asarray(pred["m"], dtype=np.float32).reshape(-1))
        y = np.asarray(wi.target_value, dtype=np.float32).reshape(-1)
        assert m.shape == y.shape, "CHGNet/target atom count mismatch"
        yp_parts.append(m)
        yt_parts.append(y)
    y_pred = np.concatenate(yp_parts)
    y_true = np.concatenate(yt_parts)
    overall = score(y_true, y_pred)

    run_dir = REPO_ROOT / "models" / "runs" / f"{spec.name}-{args.dataset}"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": args.target, "dataset": args.dataset,
        "n_test_atoms": int(y_true.shape[0]),
        "metrics": {"chgnet-ft": overall},
    }
    (eval_dir / "baselines-chgnet-ft.json").write_text(json.dumps(payload, indent=2))

    metrics = {**overall, "n_samples": int(y_true.shape[0])}
    plot_parity(
        y_pred, y_true, eval_dir / "parity-baseline-chgnet-ft.png",
        metrics, PROP_LABEL, UNIT,
    )
    plot_error_distribution(
        y_pred, y_true, eval_dir / "error-dist-baseline-chgnet-ft.png",
        PROP_LABEL, UNIT,
    )
    if args.wandb:
        import os
        import wandb
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=run_dir.name, name="baseline-chgnet-ft",
            config={"baseline": "chgnet-ft", "dataset": args.dataset,
                    "epochs": args.epochs, "lr": args.lr},
        )
        wandb.log({"test_mae": overall["mae"], "test_rmse": overall["rmse"],
                   "test_r2": overall["r2"]})
        wandb.finish()
    print(
        f"chgnet-ft (test, {y_true.shape[0]} atoms): "
        f"MAE={overall['mae']:.4f} RMSE={overall['rmse']:.4f} R2={overall['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
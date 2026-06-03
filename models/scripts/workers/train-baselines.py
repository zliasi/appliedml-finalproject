"""Train and evaluate non-GNN baselines for one target+dataset+version.

Uses the same stratified split as the GNN (by group_key, stratified
by composition type). All composition-only baselines plus SOAP+KRR
plus a stacked ensemble. Writes one checkpoint per baseline to
``models/runs/<target>-<dataset>-<version>/checkpoints/`` and a
``baseline-metrics.json`` to ``.../eval/``.

Usage:
    python scripts/workers/train-baselines.py \\
        --target hads --dataset fcc12-v1p1
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(MODELS_ROOT))
REPO_ROOT = MODELS_ROOT.parent

from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.baselines import (  # noqa: E402
    BASELINE_NAMES,
    SUPPORTED_ELEMENTS,
    build_baseline,
    build_lincomb_model,
    get_soap_config,
    save_baseline_checkpoint,
)
from src.splits import (  # noqa: E402
    items_to_features,
    load_composition_index,
    parse_dataset_token,
    stratified_split,
)
from src.targets import WorkItem, load_target_spec  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SOAP_MAX_TRAIN: int = 10000
SOAP_MAX_TEST: int = 5000


def evaluate(
    name: str, y_true: np.ndarray, y_pred: np.ndarray,
) -> dict:
    """Standard metrics dict for one baseline."""
    return {
        "model": name,
        "mae_ev": float(mean_absolute_error(y_true, y_pred)),
        "rmse_ev": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "r2": float(r2_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }


def compute_pure_metal_values(
    all_items: list[WorkItem],
) -> dict[str, float]:
    """Mean target value for each pure-metal ``<elem>100`` composition.

    Lincomb reference: average ``target_value`` over all WorkItems
    whose comp_id is ``ag100``, ``au100``, etc.
    """
    by_elem: dict[str, list[float]] = defaultdict(list)
    elem_for_id = {
        f"{e.lower()}100": e for e in SUPPORTED_ELEMENTS
    }
    for wi in all_items:
        elem = elem_for_id.get(wi.comp_id)
        if elem is not None:
            by_elem[elem].append(wi.target_value)
    out: dict[str, float] = {}
    for elem in sorted(by_elem):
        out[elem] = float(np.mean(by_elem[elem]))
        logger.info(
            "Pure %s: %.4f eV (n=%d)",
            elem, out[elem], len(by_elem[elem]),
        )
    return out


def compute_soap_descriptors(
    items: list[WorkItem],
    include_hydrogen: bool,
) -> tuple[np.ndarray, list[int]]:
    """SOAP descriptors for atoms read from each item's traj path.

    Returns the descriptor matrix and the indices of items that
    actually had a traj file present (so callers can subset the
    target array accordingly).
    """
    from ase.io import read as ase_read
    from dscribe.descriptors import SOAP

    soap = SOAP(**get_soap_config(
        SUPPORTED_ELEMENTS, include_hydrogen=include_hydrogen,
    ))

    descriptors = []
    valid_indices = []
    for i, wi in enumerate(items):
        if not wi.traj_path.exists():
            continue
        atoms = ase_read(wi.traj_path)
        desc = soap.create(atoms)
        descriptors.append(desc.flatten())
        valid_indices.append(i)

    assert len(descriptors) > 0, "No SOAP descriptors computed"
    logger.info(
        "SOAP: %d/%d structures",
        len(descriptors), len(items),
    )
    return np.array(descriptors, dtype=np.float32), valid_indices


def train_stacked_ensemble(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict:
    """Composition-only stacked ensemble (5-fold CV → Ridge meta)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    comp_baselines = [
        n for n in BASELINE_NAMES
        if n not in ("soap-krr", "stacked-ensemble")
    ]
    n_train = len(train_y)
    n_test = len(test_y)
    n_models = len(comp_baselines)

    train_meta = np.zeros((n_train, n_models))
    test_meta = np.zeros((n_test, n_models))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for col, name in enumerate(comp_baselines):
        logger.info("Stack CV: %s", name)
        test_model = build_baseline(name)
        test_model.fit(train_x, train_y)
        test_meta[:, col] = test_model.predict(test_x)
        for train_idx, val_idx in kf.split(train_x):
            fold = build_baseline(name)
            fold.fit(train_x[train_idx], train_y[train_idx])
            train_meta[val_idx, col] = fold.predict(train_x[val_idx])

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(train_meta, train_y)
    return evaluate(
        "stacked-ensemble", test_y, meta_model.predict(test_meta),
    )


def parse_args() -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--skip-soap", action="store_true",
        help="Skip SOAP+KRR (requires dscribe)",
    )
    p.add_argument(
        "--baselines", type=str, nargs="*", default=None,
        help="Subset of baselines (default: all comp-only)",
    )
    return p.parse_args()


def main() -> None:
    """Train all baselines + stacked ensemble; save metrics + ckpts."""
    args = parse_args()
    spec = load_target_spec(args.target)
    dataset_name, version = parse_dataset_token(args.dataset)

    dataset_dir = REPO_ROOT / "datasets" / dataset_name
    json_path = dataset_dir / "data" / spec.json_filename
    run_dir = (
        REPO_ROOT / "models" / "runs"
        / f"{spec.name}-{args.dataset}"
    )
    eval_dir = run_dir / "eval"
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    assert json_path.exists(), f"Missing: {json_path}"

    all_items = list(spec.iter_work_items(json_path, dataset_dir))
    logger.info("Loaded %d WorkItems", len(all_items))

    comp_types, compositions = load_composition_index(dataset_dir)
    items_by_group: dict[str, list[WorkItem]] = defaultdict(list)
    for wi in all_items:
        items_by_group[wi.group_key].append(wi)
    group_keys = sorted(items_by_group.keys())

    train_keys, val_keys, test_keys = stratified_split(
        group_keys, comp_types,
    )
    train_items = [
        wi for k in train_keys for wi in items_by_group[k]
    ]
    test_items = [
        wi for k in test_keys for wi in items_by_group[k]
    ]
    logger.info(
        "Split: train=%d test=%d groups (%d/%d items)",
        len(train_keys), len(test_keys),
        len(train_items), len(test_items),
    )

    train_x, train_y = items_to_features(train_items, compositions)
    test_x, test_y = items_to_features(test_items, compositions)

    baselines_to_run = args.baselines or [
        n for n in BASELINE_NAMES if n != "soap-krr"
    ]
    all_results: list[dict] = []

    # --- lincomb -------------------------------------------------
    pure_values = compute_pure_metal_values(all_items)
    if pure_values:
        lincomb_model = build_lincomb_model(
            pure_values, SUPPORTED_ELEMENTS,
        )
        lc_pred = lincomb_model.predict(test_x)
        lc_metrics = evaluate("lincomb", test_y, lc_pred)
        all_results.append(lc_metrics)
        logger.info(
            "lincomb: MAE=%.4f RMSE=%.4f R2=%.4f",
            lc_metrics["mae_ev"], lc_metrics["rmse_ev"],
            lc_metrics["r2"],
        )
        ckpt = checkpoint_dir / (
            f"{spec.checkpoint_prefix}-lincomb-{args.dataset}.pt"
        )
        save_baseline_checkpoint(
            path=ckpt, model=lincomb_model, kind="lincomb",
            feature_kind="composition",
            feature_config={"pure_values": pure_values},
            elements=SUPPORTED_ELEMENTS, dataset=args.dataset,
            metadata={
                "test_mae_ev": lc_metrics["mae_ev"],
                "test_rmse_ev": lc_metrics["rmse_ev"],
                "test_r2": lc_metrics["r2"],
                "n_train": int(len(train_x)),
                "n_test": int(len(test_x)),
            },
        )

    # --- composition-only baselines ------------------------------
    for name in baselines_to_run:
        if name == "soap-krr":
            continue
        logger.info("Training: %s", name)
        model = build_baseline(name)
        model.fit(train_x, train_y)
        metrics = evaluate(name, test_y, model.predict(test_x))
        all_results.append(metrics)
        logger.info(
            "%s: MAE=%.4f RMSE=%.4f R2=%.4f",
            name, metrics["mae_ev"], metrics["rmse_ev"],
            metrics["r2"],
        )
        ckpt = checkpoint_dir / (
            f"{spec.checkpoint_prefix}-{name}-{args.dataset}.pt"
        )
        save_baseline_checkpoint(
            path=ckpt, model=model, kind=name,
            feature_kind="composition", feature_config={},
            elements=SUPPORTED_ELEMENTS, dataset=args.dataset,
            metadata={
                "test_mae_ev": metrics["mae_ev"],
                "test_rmse_ev": metrics["rmse_ev"],
                "test_r2": metrics["r2"],
                "n_train": int(len(train_x)),
                "n_test": int(len(test_x)),
            },
        )

    # --- soap-krr ------------------------------------------------
    if not args.skip_soap and (
        args.baselines is None or "soap-krr" in baselines_to_run
    ):
        soap_train = list(train_items)
        soap_test = list(test_items)
        if len(soap_train) > SOAP_MAX_TRAIN:
            rng = np.random.default_rng(42)
            idx = rng.choice(
                len(soap_train), SOAP_MAX_TRAIN, replace=False,
            )
            soap_train = [soap_train[i] for i in idx]
            logger.info(
                "SOAP subsample train: -> %d", SOAP_MAX_TRAIN,
            )
        if len(soap_test) > SOAP_MAX_TEST:
            rng = np.random.default_rng(43)
            idx = rng.choice(
                len(soap_test), SOAP_MAX_TEST, replace=False,
            )
            soap_test = [soap_test[i] for i in idx]
            logger.info(
                "SOAP subsample test: -> %d", SOAP_MAX_TEST,
            )

        train_soap, train_valid = compute_soap_descriptors(
            soap_train, spec.include_hydrogen,
        )
        test_soap, test_valid = compute_soap_descriptors(
            soap_test, spec.include_hydrogen,
        )
        soap_train_y = np.array(
            [soap_train[i].target_value for i in train_valid]
        )
        soap_test_y = np.array(
            [soap_test[i].target_value for i in test_valid]
        )

        model = build_baseline("soap-krr")
        model.fit(train_soap, soap_train_y)
        metrics = evaluate(
            "soap-krr", soap_test_y, model.predict(test_soap),
        )
        all_results.append(metrics)
        logger.info(
            "soap-krr: MAE=%.4f RMSE=%.4f R2=%.4f",
            metrics["mae_ev"], metrics["rmse_ev"], metrics["r2"],
        )
        ckpt = checkpoint_dir / (
            f"{spec.checkpoint_prefix}-soap-krr-{args.dataset}.pt"
        )
        save_baseline_checkpoint(
            path=ckpt, model=model, kind="soap-krr",
            feature_kind="soap",
            feature_config=get_soap_config(
                SUPPORTED_ELEMENTS,
                include_hydrogen=spec.include_hydrogen,
            ),
            elements=SUPPORTED_ELEMENTS, dataset=args.dataset,
            metadata={
                "test_mae_ev": metrics["mae_ev"],
                "test_rmse_ev": metrics["rmse_ev"],
                "test_r2": metrics["r2"],
                "n_train": int(len(soap_train_y)),
                "n_test": int(len(soap_test_y)),
            },
        )

    # --- stacked ensemble ---------------------------------------
    logger.info("Training stacked ensemble...")
    stack = train_stacked_ensemble(train_x, train_y, test_x, test_y)
    all_results.append(stack)
    logger.info(
        "stacked-ensemble: MAE=%.4f RMSE=%.4f R2=%.4f",
        stack["mae_ev"], stack["rmse_ev"], stack["r2"],
    )

    output_path = eval_dir / "baseline-metrics.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results: %s", output_path)


if __name__ == "__main__":
    main()

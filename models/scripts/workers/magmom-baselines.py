"""Non-GNN per-atom baselines for the magmom target.

Trains simple references the GNNs must beat: a per-element-mean predictor (the
floor), a linear regression, and a GPU gradient-boosted-tree model (XGBoost)
on per-atom local features. Reads the SAME built graph caches the GNNs train on,
so the comparison is on identical data and the identical test split, and scores
per-atom MAE/RMSE/R2 like the GNN eval.

Run via scripts/03-submit-baselines.sh, or directly:
    python scripts/workers/magmom-baselines.py \
        --target magmom --dataset magmom21-v0p1 --cutoff 6
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

MODELS_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(MODELS_ROOT))

from src.dataset import load_graphs  # noqa: E402
from src.metrics import (  # noqa: E402
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)

DEFAULT_CUTOFF: int = 6
DEFAULT_DEVICE: str = "cuda"  # XGBoost device, use "cpu" on a GPU-less node
XGB_N_ESTIMATORS: int = 500
XGB_MAX_DEPTH: int = 6
XGB_LEARNING_RATE: float = 0.1
RANDOM_SEED: int = 0
N_DISTANCE_FEATURES: int = 4  # degree, mean, min, max neighbour distance


def build_element_lut(dataset: Any) -> tuple[np.ndarray, list[int]]:
    """Map atomic numbers to dense indices from the training graphs.

    Args:
        dataset: An indexable dataset of PyG ``Data`` slabs.

    Returns:
        A tuple ``(lut, elements)`` where ``lut[z]`` is the dense index of
        atomic number ``z`` (or -1 if unseen) and ``elements`` is the sorted
        list of atomic numbers.
    """
    seen: set[int] = set()
    for i in range(len(dataset)):
        seen.update(int(z) for z in dataset[i].x.view(-1).tolist())
    elements = sorted(seen)
    assert elements, "no atoms found in the training graphs"
    lut = np.full(max(elements) + 1, -1, dtype=np.int64)
    for idx, z in enumerate(elements):
        lut[z] = idx
    return lut, elements


def graph_features(
    graph: Any, lut: np.ndarray, n_elem: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-atom features, targets, and self-element index for one slab.

    Features per atom: self-element one-hot, neighbour-element counts, and the
    coordination number plus mean/min/max neighbour distance.

    Args:
        graph: One PyG ``Data`` slab (x, edge_index, edge_attr, y).
        lut: Atomic-number to dense-index lookup (-1 for unseen).
        n_elem: Number of distinct elements (one-hot width).

    Returns:
        ``(features, targets, self_idx)`` for every atom in the slab.
    """
    z = graph.x.view(-1).numpy().astype(np.int64)
    n_atoms = z.shape[0]
    self_idx = np.where(z < lut.shape[0], lut[np.clip(z, 0, lut.shape[0] - 1)], -1)

    self_oh = np.zeros((n_atoms, n_elem), dtype=np.float32)
    rows = np.arange(n_atoms)[self_idx >= 0]
    self_oh[rows, self_idx[self_idx >= 0]] = 1.0

    edge_index = graph.edge_index.numpy()
    n_edges = edge_index.shape[1]
    src, dst = edge_index[0], edge_index[1]
    if graph.edge_attr is not None and n_edges > 0:
        dist = graph.edge_attr.numpy().reshape(n_edges, -1)[:, 0].astype(np.float32)
    else:
        dist = np.zeros(n_edges, dtype=np.float32)

    degree = np.zeros(n_atoms, dtype=np.float32)
    dist_sum = np.zeros(n_atoms, dtype=np.float32)
    dist_max = np.zeros(n_atoms, dtype=np.float32)
    dist_min = np.full(n_atoms, np.inf, dtype=np.float32)
    nbr_counts = np.zeros((n_atoms, n_elem), dtype=np.float32)
    if n_edges > 0:
        np.add.at(degree, dst, 1.0)
        np.add.at(dist_sum, dst, dist)
        np.maximum.at(dist_max, dst, dist)
        np.minimum.at(dist_min, dst, dist)
        nb_idx = np.where(
            z[src] < lut.shape[0], lut[np.clip(z[src], 0, lut.shape[0] - 1)], -1,
        )
        ok = nb_idx >= 0
        np.add.at(nbr_counts, (dst[ok], nb_idx[ok]), 1.0)

    dist_mean = np.where(degree > 0, dist_sum / np.maximum(degree, 1.0), 0.0)
    dist_min = np.where(np.isfinite(dist_min), dist_min, 0.0)
    feats = np.concatenate(
        [
            self_oh,
            nbr_counts,
            degree[:, None],
            dist_mean[:, None],
            dist_min[:, None],
            dist_max[:, None],
        ],
        axis=1,
    ).astype(np.float32)

    y = graph.y.view(-1).numpy().astype(np.float32)
    assert y.shape[0] == n_atoms, "atom/target count mismatch in baseline features"
    return feats, y, self_idx


def featurise_split(
    dataset: Any, lut: np.ndarray, n_elem: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-atom features/targets/self-index over every slab in a split."""
    feats_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    idx_list: list[np.ndarray] = []
    for i in range(len(dataset)):
        feats, y, self_idx = graph_features(dataset[i], lut, n_elem)
        feats_list.append(feats)
        y_list.append(y)
        idx_list.append(self_idx)
    return (
        np.concatenate(feats_list, axis=0),
        np.concatenate(y_list, axis=0),
        np.concatenate(idx_list, axis=0),
    )


def per_element_mean_predict(
    y_train: np.ndarray, idx_train: np.ndarray, idx_test: np.ndarray, n_elem: int,
) -> np.ndarray:
    """Predict each test atom's moment as the train mean for its element."""
    global_mean = float(y_train.mean())
    means = np.full(n_elem, global_mean, dtype=np.float32)
    for c in range(n_elem):
        mask = idx_train == c
        if mask.any():
            means[c] = float(y_train[mask].mean())
    out = np.where(idx_test >= 0, means[np.clip(idx_test, 0, n_elem - 1)], global_mean)
    return out.astype(np.float32)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE/RMSE/R2 over all atoms."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def split_path(graphs_dir: Path, cutoff: int, dataset: str, split: str) -> Path:
    """Cache path for one split, matching the build-graphs naming."""
    return graphs_dir / f"graphs-r{cutoff}-{dataset}-{split}.lmdb"


def run_baselines(
    target: str, dataset: str, cutoff: int, out_path: Optional[Path],
    device: str = DEFAULT_DEVICE,
) -> dict[str, Any]:
    """Train and evaluate the node-level baselines, write metrics JSON."""
    run_dir = MODELS_ROOT / "runs" / f"{target}-{dataset}"
    graphs_dir = run_dir / "graphs"
    train_path = split_path(graphs_dir, cutoff, dataset, "train")
    test_path = split_path(graphs_dir, cutoff, dataset, "test")
    assert train_path.exists(), f"missing train cache: {train_path}"
    assert test_path.exists(), f"missing test cache: {test_path}"

    train_ds = load_graphs(train_path)
    test_ds = load_graphs(test_path)
    lut, elements = build_element_lut(train_ds)
    n_elem = len(elements)
    logger.info("elements=%d cutoff=r%d", n_elem, cutoff)

    x_train, y_train, idx_train = featurise_split(train_ds, lut, n_elem)
    x_test, y_test, idx_test = featurise_split(test_ds, lut, n_elem)
    logger.info("train atoms=%d test atoms=%d", y_train.shape[0], y_test.shape[0])

    results: dict[str, Any] = {}

    pred = per_element_mean_predict(y_train, idx_train, idx_test, n_elem)
    results["per_element_mean"] = score(y_test, pred)

    linear = LinearRegression()
    linear.fit(x_train, y_train)
    results["linear_regression"] = score(y_test, linear.predict(x_test))

    xgb = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE, tree_method="hist",
        device=device, random_state=RANDOM_SEED,
    )
    xgb.fit(x_train, y_train)
    results["xgboost"] = score(y_test, xgb.predict(x_test))

    payload = {
        "target": target,
        "dataset": dataset,
        "cutoff": cutoff,
        "n_elements": n_elem,
        "n_train_atoms": int(y_train.shape[0]),
        "n_test_atoms": int(y_test.shape[0]),
        "metrics": results,
    }

    out = out_path or (run_dir / "eval" / f"baselines-magmom-r{cutoff}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", out)

    print(f"\nnode-level baselines (test, r{cutoff}, {y_test.shape[0]} atoms):")
    for name, m in results.items():
        print(f"  {name:24s} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}")
    print(f"\nwrote {out}")
    return payload


def main() -> None:
    """Parse args and run the node-level baselines."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="per-atom magmom baselines")
    parser.add_argument("--target", default="magmom")
    parser.add_argument("--dataset", required=True, help="e.g. magmom21-v0p1")
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE,
        help="XGBoost device: cuda (GPU) or cpu",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run_baselines(
        args.target, args.dataset, args.cutoff, args.out, args.device,
    )


if __name__ == "__main__":
    main()
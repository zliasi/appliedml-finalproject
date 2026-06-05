"""Evaluate trained GNN or baseline checkpoints on test set.

Loads each checkpoint, runs inference on the corresponding test
split, writes per-checkpoint metrics + parity + error-distribution
plots into ``runs/<target>-<dataset>/eval/``, and prints a ranked
summary across all evaluated checkpoints.

Usage:
    # one checkpoint
    python scripts/04-evaluate.py --target hads --dataset fcc12-v1p1 \\
        --checkpoint runs/hads-fcc12-v1p1/checkpoints/hads-schnet-r8-...-3k.pt

    # everything under runs/<target>-<dataset>/checkpoints/
    python scripts/04-evaluate.py --target hads --dataset fcc12-v1p1 --all
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np  # noqa: E402

MODELS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODELS_ROOT))
REPO_ROOT = MODELS_ROOT.parent

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from src.baselines import (  # noqa: E402
    SUPPORTED_ELEMENTS,
    get_soap_config,
)
from src.dataset import load_graphs  # noqa: E402
from src.metrics import (  # noqa: E402
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from src.plots import (  # noqa: E402
    plot_error_distribution,
    plot_parity,
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



GNN_ARCH_PATTERN: re.Pattern = re.compile(r"c\d+l\d+h\d+")
PROPERTY_LABEL: dict[str, str] = {
    "hads": "adsorption energy",
    "wf": "work function",
    "magmom": "magnetic moment",
}
PROPERTY_UNIT: dict[str, str] = {
    "hads": "eV",
    "wf": "eV",
    "magmom": "muB",
}
COMP_TYPE_MAP: dict[int, str] = {
    1: "pure", 2: "binary", 3: "ternary",
    4: "quaternary", 5: "quinary",
    6: "senary", 7: "septenary",
}


# ----------------------------------------------------------------
# GNN evaluation
# ----------------------------------------------------------------


def _is_gnn_checkpoint(path: Path) -> bool:
    """A GNN stem carries an architecture token like ``c128l3h1``."""
    return bool(GNN_ARCH_PATTERN.search(path.stem))


def _find_arch_index(parts: list[str]) -> int:
    """Index of the architecture segment within stem-split parts."""
    for i, part in enumerate(parts):
        if GNN_ARCH_PATTERN.match(part):
            return i
    raise ValueError(
        f"No architecture segment in: {'-'.join(parts)}"
    )


def derive_test_graph_path(
    checkpoint_path: Path,
    graphs_dir: Path,
) -> Path:
    """Map a GNN checkpoint name back to its test lmdb.

    Checkpoint shape:
      <prefix>-<backend>-[sub<H>-]r<C>-<arch>-<dataset>-<size>.pt
      <prefix>-ggconv-sub<H>-<arch>-<dataset>-<size>.pt
    """
    stem = checkpoint_path.stem
    parts = stem.split("-")
    assert len(parts) >= 5, f"Bad checkpoint name: {stem}"

    backend = parts[1]
    arch_index = _find_arch_index(parts)
    between = parts[2:arch_index]
    # Dataset token may be one or more hyphenated parts (e.g.
    # ``fcc12-v1p1``), everything between arch_index+1 and the trailing
    # size segment.
    dataset_tag = "-".join(parts[arch_index + 1:-1])

    if backend == "ggconv":
        n_hops = next(
            (p for p in between if p.startswith("sub")), "sub2",
        )
        graph_tag = f"ggconv-{n_hops}"
    else:
        sub = [p for p in between if p.startswith("sub")]
        r = [p for p in between if p.startswith("r")]
        graph_tag = (
            f"{sub[0]}-{r[0]}" if sub else f"{r[0]}"
        )
    if "topo" in between:
        graph_tag += "-topo"

    return graphs_dir / (
        f"graphs-{graph_tag}-{dataset_tag}-test.lmdb"
    )


def _parse_checkpoint_name(
    stem: str, base_config: dict,
) -> dict:
    """Extract hyperparams from filename when config is missing."""
    config = dict(base_config)
    parts = stem.split("-")
    assert len(parts) >= 3, f"Unexpected name: {stem}"

    config["conv_backend"] = parts[1]
    arch_match = re.match(r"c(\d+)l(\d+)h(\d+)", parts[2])
    if arch_match is not None:
        graph_type = "full"
    else:
        arch_match = re.match(
            r"c(\d+)l(\d+)h(\d+)",
            parts[3] if len(parts) > 3 else "",
        )
        graph_type = parts[2]
    config["graph_type"] = graph_type
    assert arch_match is not None, (
        f"Cannot parse arch: {stem}"
    )
    config["conv_dim"] = int(arch_match.group(1))
    config["n_conv_layers"] = int(arch_match.group(2))
    config["n_hidden_layers"] = int(arch_match.group(3))
    return config


def load_gnn_model(
    checkpoint_path: Path, device: str, build_gnn,
) -> nn.Module:
    """Rebuild the GNN from its checkpoint config and load weights."""
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False,
    )
    model_config = checkpoint.get("config", {})
    if "conv_dim" not in model_config:
        model_config = _parse_checkpoint_name(
            checkpoint_path.stem, model_config,
        )
    model = build_gnn(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def evaluate_gnn_split(
    model: nn.Module, loader: DataLoader, device: str,
    node_level: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Run inference on one DataLoader.

    Returns preds, targets, comp_ids, and (node-level only) the atomic number
    of every prediction so the metrics can be split per element. ``elements``
    is empty for a graph-level target.
    """
    model.eval()
    preds: list[float] = []
    targets: list[float] = []
    comp_ids: list[str] = []
    elements: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            p = model(batch)
            preds.extend(p.cpu().numpy().tolist())
            targets.extend(batch.y.cpu().numpy().tolist())
            if node_level:
                elements.extend(batch.x.view(-1).cpu().numpy().tolist())
            if hasattr(batch, "comp_id"):
                comp_ids.extend(batch.comp_id)
            else:
                comp_ids.extend(["unknown"] * len(p))
    return (
        np.array(preds), np.array(targets), comp_ids,
        np.array(elements),
    )


# ----------------------------------------------------------------
# Baseline evaluation
# ----------------------------------------------------------------


def _build_baseline_test_set(
    spec, dataset_dir: Path, json_path: Path,
) -> tuple[list[WorkItem], dict[str, dict[str, float]]]:
    """Materialise the deterministic baseline test split."""
    all_items = list(spec.iter_work_items(json_path, dataset_dir))
    comp_types, compositions = load_composition_index(dataset_dir)
    items_by_group: dict[str, list[WorkItem]] = defaultdict(list)
    for wi in all_items:
        items_by_group[wi.group_key].append(wi)
    _, _, test_keys = stratified_split(
        sorted(items_by_group.keys()), comp_types,
    )
    test_items = [
        wi for k in test_keys for wi in items_by_group[k]
    ]
    return test_items, compositions


def evaluate_baseline_checkpoint(
    checkpoint_path: Path,
    test_items: list[WorkItem],
    compositions: dict[str, dict[str, float]],
    include_hydrogen: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Featurise + predict for a baseline-envelope checkpoint."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )
    feature_kind = checkpoint["feature_kind"]
    model = checkpoint["model"]

    if feature_kind == "composition":
        test_x, test_y = items_to_features(
            test_items, compositions,
        )
        y_pred = np.asarray(model.predict(test_x))
        comp_ids = [wi.comp_id for wi in test_items]
        return y_pred, test_y, comp_ids

    if feature_kind == "soap":
        from ase.io import read as ase_read
        from dscribe.descriptors import SOAP

        soap = SOAP(**get_soap_config(
            SUPPORTED_ELEMENTS, include_hydrogen=include_hydrogen,
        ))
        descriptors, valid_idx = [], []
        for i, wi in enumerate(test_items):
            if not wi.traj_path.exists():
                continue
            atoms = ase_read(wi.traj_path)
            descriptors.append(soap.create(atoms).flatten())
            valid_idx.append(i)
        x = np.array(descriptors)
        y_pred = np.asarray(model.predict(x))
        y_true = np.array(
            [test_items[i].target_value for i in valid_idx]
        )
        comp_ids = [test_items[i].comp_id for i in valid_idx]
        return y_pred, y_true, comp_ids

    raise ValueError(
        f"Unknown feature_kind '{feature_kind}' in {checkpoint_path}"
    )


# ----------------------------------------------------------------
# Metrics + plotting
# ----------------------------------------------------------------


def compute_metrics(
    y_pred: np.ndarray, y_true: np.ndarray,
) -> dict[str, float | int]:
    """MAE/RMSE/R2/max_error/N."""
    errors = y_pred - y_true
    return {
        "mae": round(
            mean_absolute_error(y_true, y_pred), 4,
        ),
        "rmse": round(
            root_mean_squared_error(y_true, y_pred), 4,
        ),
        "r2": round(r2_score(y_true, y_pred), 4),
        "max_error": round(float(np.max(np.abs(errors))), 4),
        "n_samples": int(len(y_true)),
    }


def infer_composition_type(comp_id: str) -> str:
    """Map ``ag050-pd050`` -> ``binary`` etc."""
    return COMP_TYPE_MAP.get(len(comp_id.split("-")), "unknown")


def metrics_by_type(
    comp_ids: list[str],
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Same metrics but bucketed by composition_type."""
    groups: dict[
        str, tuple[list[float], list[float]]
    ] = defaultdict(lambda: ([], []))
    for i, comp_id in enumerate(comp_ids):
        t = infer_composition_type(comp_id)
        groups[t][0].append(y_pred[i])
        groups[t][1].append(y_true[i])
    out: dict[str, dict[str, float | int]] = {}
    for t, (preds, targets) in sorted(groups.items()):
        out[t] = compute_metrics(
            np.array(preds), np.array(targets),
        )
    return out


def metrics_by_element(
    elements: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Same metrics but bucketed by chemical element (node-level only)."""
    from ase.data import chemical_symbols
    groups: dict[
        int, tuple[list[float], list[float]]
    ] = defaultdict(lambda: ([], []))
    for i, z in enumerate(elements):
        groups[int(z)][0].append(y_pred[i])
        groups[int(z)][1].append(y_true[i])
    out: dict[str, dict[str, float | int]] = {}
    for z, (preds, targets) in sorted(groups.items()):
        symbol = (
            chemical_symbols[z]
            if 0 <= z < len(chemical_symbols) else str(z)
        )
        out[symbol] = compute_metrics(
            np.array(preds), np.array(targets),
        )
    return out




def save_eval_outputs(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    comp_ids: list[str],
    model_tag: str,
    output_dir: Path,
    prop_label: str,
    unit: str,
    elements: "np.ndarray | None" = None,
) -> dict[str, float | int]:
    """Write metrics JSON + parity + error plots, return overall.

    For a node-level target, ``elements`` carries the atomic number of every
    prediction, and a per-element metrics breakdown is added to the JSON.
    """
    overall = compute_metrics(y_pred, y_true)
    by_type = metrics_by_type(comp_ids, y_pred, y_true)
    payload: dict[str, Any] = {"overall": overall, "by_type": by_type}
    if elements is not None and len(elements) == len(y_pred):
        payload["by_element"] = metrics_by_element(
            elements, y_pred, y_true,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"metrics-{model_tag}.json"
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    plot_parity(
        y_pred, y_true,
        output_dir / f"parity-{model_tag}.png", overall, prop_label, unit,
    )
    plot_error_distribution(
        y_pred, y_true,
        output_dir / f"error-dist-{model_tag}.png", prop_label, unit,
    )
    logger.info("Wrote %s + parity/error plots", metrics_path)
    return overall


# ----------------------------------------------------------------
# CLI + driver
# ----------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--all", action="store_true",
        help="Evaluate every checkpoint in runs/.../checkpoints/",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Evaluate just this checkpoint (default: pick latest)",
    )
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--signed", action="store_true",
        help="Evaluate the signed-target run dir (default: absolute |m|).",
    )
    return p.parse_args()


def evaluate_one(
    checkpoint_path: Path,
    spec,
    run_dir: Path,
    eval_dir: Path,
    test_items: list[WorkItem] | None,
    compositions: dict[str, dict[str, float]] | None,
    device: str,
    prop_label: str,
    unit: str,
) -> dict[str, Any]:
    """Evaluate one checkpoint (GNN or baseline)."""
    if _is_gnn_checkpoint(checkpoint_path):
        test_path = derive_test_graph_path(
            checkpoint_path, run_dir / "graphs",
        )
        assert test_path.exists(), f"Missing test lmdb: {test_path}"
        model = load_gnn_model(
            checkpoint_path, device, spec.build_gnn,
        )
        loader = DataLoader(
            load_graphs(test_path), batch_size=64, shuffle=False,
        )
        y_pred, y_true, comp_ids, elements = evaluate_gnn_split(
            model, loader, device, node_level=spec.node_level,
        )
    else:
        assert test_items is not None
        assert compositions is not None
        y_pred, y_true, comp_ids = evaluate_baseline_checkpoint(
            checkpoint_path, test_items,
            compositions, spec.include_hydrogen,
        )
        elements = np.array([])

    metrics = save_eval_outputs(
        y_pred, y_true, comp_ids,
        checkpoint_path.stem, eval_dir, prop_label, unit,
        elements=elements,
    )
    metrics["model"] = checkpoint_path.stem
    return metrics


def load_node_baseline_metrics(eval_dir: Path) -> list[dict[str, Any]]:
    """Collect node-level baseline results written by the baseline worker.

    Reads ``baselines-*-r*.json`` (one per cutoff) and turns each baseline into a
    summary row so it ranks alongside the GNN checkpoints. Empty if absent.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_dir.glob("baselines-*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        cutoff = payload.get("cutoff")
        suffix = f"-r{cutoff}" if cutoff is not None else ""
        n_test = payload.get("n_test_atoms")
        for name, m in payload.get("metrics", {}).items():
            if not {"mae", "rmse", "r2"} <= set(m):
                continue
            rows.append({
                "model": f"baseline-{name}{suffix}",
                "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"],
                "n_samples": n_test,
            })
    return rows


def print_summary(
    metrics_list: list[dict[str, Any]],
    eval_dir: Path,
    unit: str = "",
) -> None:
    """Print a MAE-ranked table and save it as JSON."""
    ranked = sorted(metrics_list, key=lambda m: m["mae"])
    logger.info("ALL MODELS RANKED BY MAE:")
    logger.info(
        "%-40s %10s %10s %10s %8s",
        "Model", f"MAE({unit})", f"RMSE({unit})", "R2", "N",
    )
    for m in ranked:
        logger.info(
            "%-40s %10.4f %10.4f %10.4f %8s",
            m["model"], m["mae"], m["rmse"],
            m["r2"], m.get("n_samples", ""),
        )
    summary_path = eval_dir / "all-models-summary.json"
    with open(summary_path, "w") as f:
        json.dump(ranked, f, indent=2)
    logger.info("Summary: %s", summary_path)


def main() -> None:
    """Evaluate one or many checkpoints, print summary."""
    args = parse_args()
    spec = load_target_spec(args.target)
    prop_label = PROPERTY_LABEL.get(spec.name, spec.name)
    unit = PROPERTY_UNIT.get(spec.name, "")
    dataset_name, _ = parse_dataset_token(args.dataset)
    dataset_dir = REPO_ROOT / "datasets" / dataset_name
    json_path = dataset_dir / "data" / spec.json_filename
    suffix = "-signed" if args.signed else ""
    run_dir = (
        REPO_ROOT / "models" / "runs"
        / f"{spec.name}-{args.dataset}{suffix}"
    )
    eval_dir = run_dir / "eval"
    checkpoint_dir = run_dir / "checkpoints"
    assert run_dir.exists(), f"Missing run dir: {run_dir}"

    # Lazy: only materialise the baseline test split if we need it.
    baseline_test = None
    baseline_compositions = None

    def _ensure_baseline_test():
        nonlocal baseline_test, baseline_compositions
        if baseline_test is None:
            baseline_test, baseline_compositions = (
                _build_baseline_test_set(spec, dataset_dir, json_path)
            )

    if args.all:
        all_ckpts = sorted(
            checkpoint_dir.glob(f"{spec.checkpoint_prefix}-*.pt")
        )
        assert all_ckpts, f"No checkpoints in {checkpoint_dir}"
        if any(not _is_gnn_checkpoint(c) for c in all_ckpts):
            _ensure_baseline_test()
        metrics_list: list[dict[str, Any]] = []
        for ckpt in all_ckpts:
            try:
                metrics_list.append(evaluate_one(
                    ckpt, spec, run_dir, eval_dir,
                    baseline_test, baseline_compositions,
                    args.device, prop_label, unit,
                ))
            except Exception as e:
                logger.warning(
                    "Failed %s: %s", ckpt.name, e,
                )
        metrics_list.extend(load_node_baseline_metrics(eval_dir))
        print_summary(metrics_list, eval_dir, unit)
    else:
        if args.checkpoint is None:
            cands = sorted(
                checkpoint_dir.glob(
                    f"{spec.checkpoint_prefix}-*.pt",
                ),
                key=lambda p: p.stat().st_mtime,
            )
            assert cands, f"No checkpoints in {checkpoint_dir}"
            args.checkpoint = cands[-1]
            logger.info("Using latest: %s", args.checkpoint)
        if not _is_gnn_checkpoint(args.checkpoint):
            _ensure_baseline_test()
        evaluate_one(
            args.checkpoint, spec, run_dir, eval_dir,
            baseline_test, baseline_compositions,
            args.device, prop_label, unit,
        )


if __name__ == "__main__":
    main()

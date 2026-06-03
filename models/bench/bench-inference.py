"""Inference + load benchmark for the warm-start use case.

For each trained checkpoint, measures the real per-job cost: cold load
(torch.load + build_model + load_state_dict, exactly what the warm-start app
does) and warm single-slab CPU inference latency, plus the parameter count and
the checkpoint's recorded val MAE. The resulting table is what picks a backend,
where fast load and fast inference matter as much as accuracy.

Point it at the run's checkpoints directory. If no checkpoints exist yet (before
training), it falls back to building each backend from its config with random
weights - useful only to smoke-test that the backends build and run.

Usage:
    python bench/bench-inference.py --checkpoints runs/magmom-magmom21-v0p1/checkpoints
    python bench/bench-inference.py            # smoke mode if no checkpoints found
"""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
import yaml
from ase.build import fcc111

MODELS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODELS_ROOT))

from torch_geometric.data import Batch  # noqa: E402

from src.graph_builder import GraphBuilder  # noqa: E402
from targets.magmom.gnn import build_model  # noqa: E402

# atomic numbers of the 21 magmom21 elements, to populate a representative slab
ELEMENT_NUMBERS: list[int] = [
    13, 22, 23, 24, 25, 26, 27, 28, 29, 30, 42,
    44, 45, 46, 47, 74, 75, 76, 77, 78, 79,
]
SLAB_SIZE: tuple[int, int, int] = (4, 4, 4)  # ~64-atom fcc(111) slab
SLAB_VACUUM_ANG: float = 10.0
WARMUP_REPS: int = 5
DEFAULT_REPS: int = 50


def make_slab_batch(cutoff: float, seed: int = 0) -> Batch:
    """Build one synthetic ~64-atom slab as a single-graph PyG batch.

    Args:
        cutoff: Radius cutoff for the graph edges (Angstroms).
        seed: Seed for the random element assignment (reproducible).

    Returns:
        A PyG Batch holding one slab graph (with the batch vector set).
    """
    atoms = fcc111("Pt", size=SLAB_SIZE, vacuum=SLAB_VACUUM_ANG)
    rng = np.random.default_rng(seed)
    atoms.numbers = rng.choice(ELEMENT_NUMBERS, size=len(atoms))
    graph = GraphBuilder(radius_cutoff=cutoff).build_graph(atoms)
    return Batch.from_data_list([graph])


def time_inference(model: torch.nn.Module, batch: Batch, reps: int) -> float:
    """Median single-slab forward latency in milliseconds.

    Args:
        model: Model in eval mode.
        batch: One-slab batch on the target device.
        reps: Timed repetitions after warm-up.

    Returns:
        Median forward time in milliseconds.
    """
    assert reps > 0, "reps must be positive"
    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP_REPS):
            model(batch)
        times = []
        for _ in range(reps):
            start = time.perf_counter()
            model(batch)
            times.append((time.perf_counter() - start) * 1000.0)
    return median(times)


def _timed_result(
    model: torch.nn.Module, config: dict, name: str, reps: int, device: str,
    load_ms: float, val_mae: float | None,
) -> dict[str, Any]:
    """Run the slab inference timing and assemble the result record."""
    cutoff = float(config.get("radius_cutoff", 6.0))
    batch = make_slab_batch(cutoff).to(device)
    return {
        "backend": config.get("conv_backend", "?"),
        "cutoff": cutoff,
        "name": name,
        "n_params": sum(p.numel() for p in model.parameters()),
        "load_ms": round(load_ms, 2),
        "infer_ms": round(time_inference(model, batch, reps), 3),
        "val_mae": val_mae,
        "ok": True,
    }


def benchmark_checkpoint(
    ckpt_path: Path, reps: int, device: str,
) -> dict[str, Any]:
    """Benchmark one trained checkpoint (real load + inference cost).

    Args:
        ckpt_path: Path to a .pt checkpoint ({model_state_dict, config, ...}).
        reps: Timed inference repetitions.
        device: Target device string.

    Returns:
        A result record (ok or error).
    """
    try:
        start = time.perf_counter()
        checkpoint = torch.load(
            ckpt_path, map_location=device, weights_only=False,
        )
        config = checkpoint["config"]
        if "conv_backend" not in config:
            return {"name": ckpt_path.stem, "ok": False,
                    "error": "not a GNN checkpoint (no conv_backend)"}
        model = build_model(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        load_ms = (time.perf_counter() - start) * 1000.0
        return _timed_result(
            model, config, ckpt_path.stem, reps, device,
            load_ms, checkpoint.get("val_mae"),
        )
    except Exception as exc:  # record any checkpoint that fails to load/run
        return {"name": ckpt_path.stem, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}


def benchmark_config(
    config_path: Path, reps: int, device: str,
) -> dict[str, Any]:
    """Smoke benchmark one config with random weights (no checkpoint).

    Args:
        config_path: Path to a backend yaml config.
        reps: Timed inference repetitions.
        device: Target device string.

    Returns:
        A result record (ok or error); load_ms is omitted (no real checkpoint).
    """
    config = yaml.safe_load(config_path.read_text())
    try:
        model = build_model(config).to(device).eval()
        record = _timed_result(
            model, config, config_path.stem, reps, device, 0.0, None,
        )
        record.pop("load_ms")
        return record
    except Exception as exc:  # record any backend that fails to build/run
        return {"backend": config.get("conv_backend", "?"),
                "name": config_path.stem, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}


def _print_record(record: dict[str, Any]) -> None:
    """Print one result row."""
    if not record["ok"]:
        print(f"{record['name']:40s}  FAILED: {record['error']}")
        return
    load = f"load={record['load_ms']:8.1f}ms  " if "load_ms" in record else ""
    mae = "" if record["val_mae"] is None else f"  valMAE={record['val_mae']:.4f}"
    print(
        f"{record['backend']:18s} r{int(record['cutoff'])}  "
        f"params={record['n_params']:>9d}  {load}"
        f"infer={record['infer_ms']:8.2f}ms{mae}"
    )


def main() -> None:
    """Benchmark trained checkpoints (or configs as a smoke fallback)."""
    parser = argparse.ArgumentParser(description="magmom inference benchmark")
    parser.add_argument(
        "--checkpoints", type=Path, default=MODELS_ROOT / "runs",
        help="Directory searched recursively for .pt checkpoints.",
    )
    parser.add_argument(
        "--configs", type=Path,
        default=MODELS_ROOT / "targets" / "magmom" / "configs",
        help="Config dir used only when no checkpoints are found.",
    )
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "inference-results.json",
    )
    args = parser.parse_args()

    checkpoints = sorted(args.checkpoints.rglob("*.pt"))
    if checkpoints:
        print(f"benchmarking {len(checkpoints)} checkpoints from {args.checkpoints}\n")
        results = [benchmark_checkpoint(c, args.reps, args.device) for c in checkpoints]
    else:
        configs = sorted(args.configs.rglob("*.yaml"))
        assert configs, f"no checkpoints and no configs under {args.configs}"
        print("no checkpoints found; smoke-testing configs with random weights\n")
        results = [benchmark_config(c, args.reps, args.device) for c in configs]

    for record in results:
        _print_record(record)

    args.out.write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for record in results if record["ok"])
    print(f"\n{n_ok}/{len(results)} benchmarked -> {args.out}")


if __name__ == "__main__":
    main()

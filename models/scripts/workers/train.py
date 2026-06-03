"""Train one GNN model for target+dataset+version.

Loads target-specific GNN via TARGET_SPEC.build_gnn, fetches the
pre-built graph lmdb caches from
``models/runs/<target>-<dataset>-<version>/graphs/``, and writes
the best checkpoint to ``.../checkpoints/``.

Loader / precision defaults come from the ``bench/`` results:
7 DataLoader workers + pin_memory + persistent_workers (~2-3x on
loader-bound archs, ~1.4x on compute-bound) and TF32 matmul/conv
(modest extra win, no observed accuracy damage). Override via
``--workers`` / ``--no-tf32``. BF16 / torch.compile not enabled
(BF16 was a wash on speed; compile breaks ``torch_cluster.radius``
in dimenet / visnet).

Optional wandb logging via --wandb. Project name = ``run_dir.name``
(``<target>-<dataset>``); augmented datasets get a new ``<dataset>``
slug, which naturally spins up a fresh wandb project.

Usage:
    python scripts/workers/train.py --target hads --dataset fcc12-v1p1 \\
        --config targets/hads/configs/schnet/c128l3h1-r8.yaml --wandb
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import yaml

MODELS_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(MODELS_ROOT))
REPO_ROOT = MODELS_ROOT.parent

from torch_geometric.loader import DataLoader  # noqa: E402

from src.dataset import load_graphs  # noqa: E402
from src.targets import load_target_spec  # noqa: E402
from src.trainer import train  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

WANDB_ENTITY: Optional[str] = os.environ.get("WANDB_ENTITY")


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), f"Bad config: {path}"
    return cfg


def derive_graph_tag(config: dict[str, Any]) -> str:
    """Reproduce the tag scheme used by build-graphs.py."""
    backend = config.get("conv_backend", "cgconv")
    graph_type = config.get("graph_type", "full")
    edge_mode = config.get("edge_mode", "distance")
    cutoff = int(config.get("radius_cutoff", 6.0))

    if backend == "ggconv":
        n_hops = config.get("n_hops", 2)
        return f"ggconv-sub{n_hops}"
    if graph_type.startswith("sub"):
        tag = f"{graph_type}-r{cutoff}"
    else:
        tag = f"r{cutoff}"
    if edge_mode == "topological":
        tag += "-topo"
    return tag


def setup_data(
    config: dict[str, Any],
    run_dir: Path,
    dataset_token: str,
    workers: int,
) -> tuple[DataLoader, DataLoader, int]:
    """Open train/val lmdb caches; build DataLoaders."""
    tag = derive_graph_tag(config)
    graphs_dir = run_dir / "graphs"
    train_path = graphs_dir / (
        f"graphs-{tag}-{dataset_token}-train.lmdb"
    )
    val_path = graphs_dir / (
        f"graphs-{tag}-{dataset_token}-val.lmdb"
    )
    assert train_path.exists(), f"Missing: {train_path}"
    assert val_path.exists(), f"Missing: {val_path}"

    train_data = load_graphs(train_path)
    val_data = load_graphs(val_path)
    bs = config.get("batch_size", 64)

    loader_kwargs: dict[str, Any] = dict(
        num_workers=workers,
        pin_memory=workers > 0,
        persistent_workers=workers > 0,
    )
    train_loader = DataLoader(
        train_data, batch_size=bs, shuffle=True, drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_data, batch_size=bs, shuffle=False,
        **loader_kwargs,
    )
    logger.info(
        "Data: train=%d val=%d batch_size=%d workers=%d",
        len(train_data), len(val_data), bs, workers,
    )
    return train_loader, val_loader, len(train_data) + len(val_data)


def setup_precision(use_tf32: bool) -> None:
    """Configure float precision once at process start.

    TF32 routes FP32 matmul / convolution through the tensor cores
    at 10-bit mantissa; gives ~5% on matmul-heavy archs with no
    observed accuracy damage on these GNNs (see ``bench/``).
    """
    if torch.cuda.is_available() and use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled (matmul + cudnn)")


def build_checkpoint_name(
    config: dict[str, Any],
    target_prefix: str,
    dataset_token: str,
    n_samples: int,
) -> str:
    """Compose the ``.pt`` filename stem."""
    assert n_samples > 0, "Must have at least one sample"
    size_tag = (
        f"{n_samples // 1000}k"
        if n_samples >= 1000
        else str(n_samples)
    )

    conv_backend = config.get("conv_backend", "cgconv")
    graph_type = config.get("graph_type", "full")
    edge_mode = config.get("edge_mode", "distance")
    cutoff = int(config.get("radius_cutoff", 6.0))

    parts = [f"{target_prefix}-{conv_backend}"]
    if graph_type.startswith("sub"):
        parts.append(graph_type)
    if conv_backend != "ggconv":
        parts.append(f"r{cutoff}")
    if edge_mode == "topological":
        parts.append("topo")
    parts.append(
        f"c{config.get('conv_dim', 64)}"
        f"l{config.get('n_conv_layers', 3)}"
        f"h{config.get('n_hidden_layers', 0)}"
    )
    parts.append(dataset_token)
    parts.append(size_tag)
    return "-".join(parts)


def _make_wandb_callback() -> Callable:
    """Per-epoch wandb logger."""
    import wandb

    def callback(epoch, train_loss, val_metrics):
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        })
    return callback


def setup_wandb(
    config: dict[str, Any],
    train_config: dict[str, Any],
    model: torch.nn.Module,
    args: argparse.Namespace,
    project: str,
) -> Optional[Callable]:
    """Initialise wandb run; return per-epoch callback or None."""
    if not args.wandb:
        return None
    import wandb

    n_params = sum(p.numel() for p in model.parameters())
    run_name = (
        args.wandb_run_name or train_config["checkpoint_name"]
    )
    wandb.init(
        entity=WANDB_ENTITY, project=project, name=run_name,
        config={**config, **train_config, "n_params": n_params},
    )
    wandb.watch(model, log="gradients", log_freq=50)
    return _make_wandb_callback()


def parse_args() -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--workers", type=int, default=7,
        help="DataLoader workers (default 7; bench shows ~2-3x on "
             "loader-bound archs, ~1.4x on compute-bound).",
    )
    p.add_argument(
        "--no-tf32", action="store_true",
        help="Disable TF32 matmul/conv (default: on).",
    )
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-run-name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    """Train one GNN end to end."""
    args = parse_args()
    spec = load_target_spec(args.target)
    run_dir = (
        REPO_ROOT / "models" / "runs"
        / f"{spec.name}-{args.dataset}"
    )
    assert run_dir.exists(), f"Missing run dir: {run_dir}"

    setup_precision(use_tf32=not args.no_tf32)

    config = load_config(args.config)
    train_loader, val_loader, n_samples = setup_data(
        config, run_dir, args.dataset, args.workers,
    )

    model = spec.build_gnn(config)
    logger.info(
        "Model params: %d",
        sum(p.numel() for p in model.parameters()),
    )

    checkpoint_name = build_checkpoint_name(
        config, spec.checkpoint_prefix, args.dataset, n_samples,
    )

    train_config = {
        **config,
        "device": args.device,
        "lr": config.get("lr", 1e-3),
        "weight_decay": config.get("weight_decay", 0.01),
        "max_epochs": config.get("max_epochs", 1000),
        "patience": config.get("patience", 100),
        "output_dir": str(run_dir / "checkpoints"),
        "checkpoint_name": checkpoint_name,
    }

    wandb_project = run_dir.name
    logger.info("Wandb project: %s", wandb_project)
    epoch_callback = setup_wandb(
        config, train_config, model, args, wandb_project,
    )

    best_path = train(
        model, train_loader, val_loader, train_config,
        epoch_callback=epoch_callback,
    )
    logger.info("Best model: %s", best_path)

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()

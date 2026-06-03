"""Build PyG graphs for one target+dataset+version.

Reads the target's JSON via TARGET_SPEC.iter_work_items, groups by
the target's group_key (slab-key for hads, comp_id for wf),
stratified-splits by composition type, and writes lmdb train/val/test
graph caches into ``models/runs/<target>-<dataset>-<version>/graphs/``.

Parallel build uses pickle-bytes return through the IPC pipe to avoid
``vm.max_map_count`` exhaustion from torch.multiprocessing's
per-tensor shared-memory reductions.

Usage:
    python scripts/workers/build-graphs.py \\
        --target hads --dataset fcc12-v1p1 --cutoff 8
    python scripts/workers/build-graphs.py \\
        --target hads --dataset fcc4-v1p1 \\
        --representation ggconv --n-hops 2
"""

import argparse
import csv
import logging
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch.multiprocessing as _torch_mp

# sys.path: models/ on it so we can import src.* and targets.*
MODELS_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(MODELS_ROOT))
REPO_ROOT = MODELS_ROOT.parent

from ase.io import read  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from src.dataset import save_graphs  # noqa: E402
from src.graph_builder import (  # noqa: E402
    GraphBuilder,
    DEFAULT_RADIUS_CUTOFF,
)
from src.splits import load_composition_index  # noqa: E402
from src.targets import WorkItem, load_target_spec  # noqa: E402

# Use file-system sharing strategy as belt-and-braces; the pickle-
# bytes pipe is the primary fix for vm.max_map_count exhaustion.
_torch_mp.set_sharing_strategy("file_system")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CUTOFF_SCALING: float = 1.2
TRAIN_RATIO: float = 0.80
VAL_RATIO: float = 0.10
TEST_RATIO: float = 0.10
SPLIT_SEED: int = 42
VALID_REPRESENTATIONS: tuple[str, ...] = ("atomic", "ggconv")
VALID_N_HOPS: tuple[int, ...] = (1, 2, 3)

DATASET_TOKEN_RE = re.compile(r"^(?P<name>.+?)-(?P<version>v\d+p\d+)$")


def parse_dataset_token(token: str) -> tuple[str, str]:
    """Split ``<name>-v<M>p<N>`` into (name, version).

    Raises:
        ValueError: If the token does not match the expected shape.
    """
    match = DATASET_TOKEN_RE.match(token)
    if not match:
        raise ValueError(
            f"Bad --dataset {token!r}: expected <name>-v<M>p<N>, "
            f"e.g. fcc12-v1p1"
        )
    return match["name"], match["version"]


def load_composition_types(
    dataset_dir: Path,
) -> dict[str, str]:
    """Build comp_id -> composition_type via the shared helper.

    The build pipeline only needs the type map; ``load_composition_index``
    returns (comp_types, compositions) so we discard the second element.
    """
    comp_types, _ = load_composition_index(dataset_dir)
    return comp_types


def stratified_split(
    group_keys: list[str],
    comp_types: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """Split group_keys into train/val/test stratified by comp type."""
    assert len(group_keys) > 0, "Must have at least one group"
    rng = np.random.default_rng(SPLIT_SEED)

    by_type: dict[str, list[str]] = defaultdict(list)
    for key in group_keys:
        comp_id = key.split("/")[0]
        composition_type = comp_types.get(comp_id, "unknown")
        by_type[composition_type].append(key)

    train_keys: list[str] = []
    val_keys: list[str] = []
    test_keys: list[str] = []

    for composition_type, keys in sorted(by_type.items()):
        shuffled = list(keys)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_val = max(1, int(round(n_total * VAL_RATIO)))
        n_test = max(1, int(round(n_total * TEST_RATIO)))
        n_train = n_total - n_val - n_test
        assert n_train > 0, (
            f"Not enough groups for {composition_type}"
        )

        train_keys.extend(shuffled[:n_train])
        val_keys.extend(shuffled[n_train:n_train + n_val])
        test_keys.extend(shuffled[n_train + n_val:])

        logger.info(
            "Split %s: train=%d val=%d test=%d",
            composition_type, n_train, n_val,
            len(shuffled) - n_train - n_val,
        )

    return train_keys, val_keys, test_keys


def resolve_graph_tag(
    representation: str,
    n_hops: int | None,
    edge_mode: str,
    cutoff: float = DEFAULT_RADIUS_CUTOFF,
) -> str:
    """Filename tag for the output lmdb."""
    assert representation in VALID_REPRESENTATIONS, (
        f"Invalid representation: {representation}"
    )
    if representation == "ggconv":
        assert n_hops is not None, "ggconv requires --n-hops"
        return f"ggconv-sub{n_hops}"
    if n_hops is not None:
        tag = f"sub{n_hops}-r{int(cutoff)}"
    else:
        tag = f"r{int(cutoff)}"
    if edge_mode == "topological":
        tag += "-topo"
    return tag


def create_builder(
    representation: str,
    cutoff: float,
    cutoff_scaling: float,
    edge_mode: str,
) -> object:
    """GraphBuilder (atomic) or GGConvGraphBuilder (ggconv).

    The ggconv builder is only available for targets that ship it
    (currently hads only).
    """
    if representation == "ggconv":
        try:
            from targets.hads.ggconv_graph_builder import (
                GGConvGraphBuilder,
                SUPPORTED_ELEMENTS as GGCONV_ELEMENTS,
            )
        except ImportError as e:
            raise ImportError(
                "ggconv representation requires "
                "targets/hads/ggconv_graph_builder.py"
            ) from e
        return GGConvGraphBuilder(
            elements=GGCONV_ELEMENTS,
            cutoff_scaling=cutoff_scaling,
        )

    return GraphBuilder(
        radius_cutoff=cutoff, edge_mode=edge_mode,
    )


# ---------------------------------------------------------------
# Parallel build path (bytes-through-pipe to dodge vm.max_map_count)
# ---------------------------------------------------------------

_worker_builder = None
_worker_n_hops: int | None = None


def _init_worker(
    representation: str,
    cutoff: float,
    cutoff_scaling: float,
    edge_mode: str,
    n_hops: int | None,
) -> None:
    """Per-process initializer: build the graph builder once."""
    global _worker_builder, _worker_n_hops
    _worker_builder = create_builder(
        representation, cutoff, cutoff_scaling, edge_mode,
    )
    _worker_n_hops = n_hops


def _build_one(work_item: WorkItem) -> bytes | None:
    """Build one graph in a worker; return pickled bytes.

    Returning bytes (not Data) avoids torch.multiprocessing's
    per-tensor shared-memory reductions, which otherwise create
    one VM mapping per tensor and exhaust ``vm.max_map_count`` on
    large datasets.
    """
    if work_item.atoms is not None:
        atoms = work_item.atoms
    elif work_item.traj_path is not None and work_item.traj_path.exists():
        atoms = read(work_item.traj_path)
    else:
        return None
    if _worker_n_hops is not None:
        graph = _worker_builder.build_local_graph(
            atoms, n_hops=_worker_n_hops,
            target=work_item.target_value,
        )
    else:
        graph = _worker_builder.build_graph(
            atoms, target=work_item.target_value,
        )
    for key, value in work_item.metadata.items():
        setattr(graph, key, value)
    return pickle.dumps(graph, protocol=-1)


def build_split_graphs(
    items_by_group: dict[str, list[WorkItem]],
    selected: list[str],
    builder: object,
    builder_args: tuple,
    n_hops: int | None,
    n_workers: int,
) -> list[Data]:
    """Build the graphs for one split (train / val / test).

    Args:
        items_by_group: map group_key -> WorkItems.
        selected: group_keys included in this split.
        builder: serial-path builder (used when n_workers <= 1).
        builder_args: forwarded to ``_init_worker`` in each pool
            worker (representation, cutoff, cutoff_scaling, edge_mode).
        n_hops: BFS hop count, or ``None`` for full-slab graphs.
        n_workers: process-pool size; serial if <= 1.
    """
    assert len(selected) > 0, "Empty split"

    work: list[WorkItem] = []
    for key in selected:
        work.extend(items_by_group[key])

    graphs: list[Data] = []
    missing = 0

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        logger.info(
            "Parallel build: %d items, %d workers",
            len(work), n_workers,
        )
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(*builder_args, n_hops),
        ) as pool:
            for blob in pool.map(_build_one, work, chunksize=64):
                if blob is None:
                    missing += 1
                else:
                    graphs.append(pickle.loads(blob))
    else:
        for wi in work:
            if wi.atoms is not None:
                atoms = wi.atoms
            elif wi.traj_path is not None and wi.traj_path.exists():
                atoms = read(wi.traj_path)
            else:
                logger.warning("Missing structure: %s", wi.group_key)
                missing += 1
                continue
            if n_hops is not None:
                graph = builder.build_local_graph(
                    atoms, n_hops=n_hops,
                    target=wi.target_value,
                )
            else:
                graph = builder.build_graph(
                    atoms, target=wi.target_value,
                )
            for key, value in wi.metadata.items():
                setattr(graph, key, value)
            graphs.append(graph)

    if missing:
        logger.warning(
            "Missing traj files: %d (of %d)", missing, len(work),
        )
    assert len(graphs) > 0, "No graphs built"
    return graphs


# ---------------------------------------------------------------
# Pipeline + CLI
# ---------------------------------------------------------------


def get_save_function(representation: str):
    """Pick the right ``save_*_graphs`` for the representation."""
    if representation == "ggconv":
        from targets.hads.ggconv_graph_builder import (
            save_ggconv_graphs,
        )
        return save_ggconv_graphs
    return save_graphs


def log_split_stats(
    split: str, graphs: list[Data], output_path: Path,
) -> None:
    """Per-split summary line."""
    node_counts = [g.x.size(0) for g in graphs]
    logger.info(
        "%s: %d graphs -> %s (nodes min=%d mean=%.1f max=%d)",
        split, len(graphs), output_path,
        min(node_counts),
        sum(node_counts) / len(node_counts),
        max(node_counts),
    )


def parse_args() -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(
        description="Build PyG graphs for target+dataset+version.",
    )
    p.add_argument(
        "--target", required=True,
        help="hads | wf (or any registered target)",
    )
    p.add_argument(
        "--dataset", required=True,
        help="Dataset token: <name>-v<M>p<N>, e.g. fcc12-v1p1",
    )
    p.add_argument(
        "-r", "--representation", type=str, default="atomic",
        choices=list(VALID_REPRESENTATIONS),
    )
    p.add_argument(
        "--cutoff", type=float, default=DEFAULT_RADIUS_CUTOFF,
    )
    p.add_argument(
        "--cutoff-scaling", type=float,
        default=DEFAULT_CUTOFF_SCALING,
    )
    p.add_argument(
        "--n-hops", type=int, default=None, choices=list(VALID_N_HOPS),
    )
    p.add_argument(
        "--edge-mode", type=str, default="distance",
        choices=["distance", "topological"],
    )
    p.add_argument(
        "--n-workers", type=int, default=1,
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output dir (default: "
             "models/runs/<target>-<dataset>/graphs/). "
             "Used by the bench so its graphs live under "
             "models/bench/graphs/ rather than under runs/.",
    )
    return p.parse_args()


def main() -> None:
    """Build graphs for one target/dataset/version."""
    args = parse_args()

    dataset_name, version = parse_dataset_token(args.dataset)
    spec = load_target_spec(args.target)

    dataset_dir = REPO_ROOT / "datasets" / dataset_name
    json_path = dataset_dir / "data" / spec.json_filename
    run_dir = (
        REPO_ROOT / "models" / "runs"
        / f"{spec.name}-{args.dataset}"
    )
    output_dir = args.output_dir or run_dir / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    assert json_path.exists(), f"Missing: {json_path}"
    logger.info(
        "target=%s dataset=%s version=%s",
        spec.name, dataset_name, version,
    )
    logger.info("json=%s out=%s", json_path, output_dir)

    if args.representation == "ggconv":
        assert args.n_hops is not None, (
            "ggconv requires --n-hops"
        )

    work_items = list(spec.iter_work_items(json_path, dataset_dir))
    assert len(work_items) > 0, "No work items emitted"

    items_by_group: dict[str, list[WorkItem]] = defaultdict(list)
    for wi in work_items:
        items_by_group[wi.group_key].append(wi)
    group_keys = sorted(items_by_group.keys())
    logger.info(
        "Total: %d items, %d unique groups, repr=%s",
        len(work_items), len(group_keys), args.representation,
    )

    comp_types = load_composition_types(dataset_dir)
    train_keys, val_keys, test_keys = stratified_split(
        group_keys, comp_types,
    )

    builder_args = (
        args.representation,
        args.cutoff,
        args.cutoff_scaling,
        args.edge_mode,
    )
    builder = (
        None if args.n_workers > 1
        else create_builder(*builder_args)
    )
    save_function = get_save_function(args.representation)
    tag = resolve_graph_tag(
        args.representation, args.n_hops,
        args.edge_mode, args.cutoff,
    )

    for split_name, keys in (
        ("train", train_keys),
        ("val", val_keys),
        ("test", test_keys),
    ):
        split_graphs = build_split_graphs(
            items_by_group, keys, builder, builder_args,
            args.n_hops, args.n_workers,
        )
        filename = f"graphs-{tag}-{args.dataset}-{split_name}.lmdb"
        out_path = output_dir / filename
        save_function(split_graphs, out_path)
        log_split_stats(split_name, split_graphs, out_path)

    logger.info("GRAPH BUILDING COMPLETE")


if __name__ == "__main__":
    main()

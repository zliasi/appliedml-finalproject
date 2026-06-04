"""Generate an HP-sweep grid of training configs for one backend.

Reads the backend's base config as a template, then writes one YAML per grid
point (conv_dim x n_conv_layers x n_hidden_layers x lr) into
``targets/<target>/sweeps/<backend>/``. The learning rate is folded into the
checkpoint name through ``arch_suffix`` so points that differ only in lr stay
distinct. Submit them with 06-submit-sweep.sh, then rank everything with
04-evaluate.py --all.

    python scripts/05-generate-sweep-configs.py --backend cgconv --cutoff 6
    python scripts/05-generate-sweep-configs.py --backend schnetconv --cutoff 8 --max-configs 12
"""

import argparse
import itertools
import random
from pathlib import Path
from typing import Any

import yaml

MODELS_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONV_DIMS: tuple[int, ...] = (64, 128, 256)
DEFAULT_N_CONV: tuple[int, ...] = (3, 4, 5)
DEFAULT_N_HIDDEN: tuple[int, ...] = (1, 2)
DEFAULT_LRS: tuple[float, ...] = (1.0e-3, 3.0e-4)
SAMPLE_SEED: int = 0


def lr_tag(lr: float) -> str:
    """Compact, filename-safe tag for a learning rate (1e-3 -> ``lr1em3``)."""
    sci = f"{lr:.0e}"  # e.g. "1e-03"
    sci = sci.replace("e-0", "em").replace("e-", "em")
    sci = sci.replace("e+0", "ep").replace("e+", "ep")
    return "lr" + sci


def load_base_config(target: str, backend: str, cutoff: int) -> dict[str, Any]:
    """Load the backend's base config as a template of the fixed fields."""
    path = (
        MODELS_ROOT / "targets" / target / "configs" / backend
        / f"c128l3h1-r{cutoff}.yaml"
    )
    assert path.exists(), f"missing base config: {path}"
    return yaml.safe_load(path.read_text())


def grid_configs(
    base: dict[str, Any],
    cutoff: int,
    conv_dims: tuple[int, ...],
    n_conv: tuple[int, ...],
    n_hidden: tuple[int, ...],
    lrs: tuple[float, ...],
) -> list[tuple[str, dict[str, Any]]]:
    """Build ``(filename, config)`` pairs for the full grid."""
    out: list[tuple[str, dict[str, Any]]] = []
    for dim, nc, nh, lr in itertools.product(conv_dims, n_conv, n_hidden, lrs):
        cfg = dict(base)
        cfg["conv_dim"] = dim
        cfg["n_conv_layers"] = nc
        cfg["n_hidden_layers"] = nh
        cfg["lr"] = lr
        cfg["radius_cutoff"] = float(cutoff)
        cfg["arch_suffix"] = lr_tag(lr)
        name = f"c{dim}l{nc}h{nh}{lr_tag(lr)}-r{cutoff}.yaml"
        out.append((name, cfg))
    return out


def parse_args() -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description="generate HP-sweep configs")
    p.add_argument("--target", default="magmom")
    p.add_argument("--backend", required=True, help="config subdir name")
    p.add_argument("--cutoff", type=int, default=6)
    p.add_argument(
        "--conv-dims", type=int, nargs="+", default=list(DEFAULT_CONV_DIMS),
    )
    p.add_argument(
        "--n-conv", type=int, nargs="+", default=list(DEFAULT_N_CONV),
    )
    p.add_argument(
        "--n-hidden", type=int, nargs="+", default=list(DEFAULT_N_HIDDEN),
    )
    p.add_argument(
        "--lrs", type=float, nargs="+", default=list(DEFAULT_LRS),
    )
    p.add_argument(
        "--max-configs", type=int, default=None,
        help="randomly subsample the grid to this many (reproducible seed)",
    )
    return p.parse_args()


def main() -> None:
    """Write the sweep configs for one backend."""
    args = parse_args()
    base = load_base_config(args.target, args.backend, args.cutoff)
    configs = grid_configs(
        base, args.cutoff,
        tuple(args.conv_dims), tuple(args.n_conv),
        tuple(args.n_hidden), tuple(args.lrs),
    )
    if args.max_configs is not None and args.max_configs < len(configs):
        configs = random.Random(SAMPLE_SEED).sample(configs, args.max_configs)

    out_dir = MODELS_ROOT / "targets" / args.target / "sweeps" / args.backend
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in configs:
        (out_dir / name).write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"wrote {len(configs)} configs to {out_dir}")
    print("submit with:")
    print(
        f"  ./scripts/06-submit-sweep.sh --target {args.target} "
        f"--dataset <DATASET> --backend {args.backend}"
    )


if __name__ == "__main__":
    main()
"""Strip optimizer state from a trained checkpoint for faster warm-start loading.

The trainer saves the full optimizer state in every ``.pt``, which ``torch.load``
deserialises on each warm-start even though inference never uses it. This writes
a slim copy keeping only what the warm-start app needs (model weights, config,
val MAE), which loads faster and is much smaller. Run it on the final chosen
model(s) before shipping. The bench (bench/bench-inference.py) quantifies the
load-time saving.

    python scripts/slim-checkpoint.py runs/magmom-magmom21-v0p1/checkpoints/best.pt
    python scripts/slim-checkpoint.py --all runs/magmom-magmom21-v0p1/checkpoints
"""

import argparse
from pathlib import Path

import torch

KEEP_KEYS: tuple[str, ...] = ("model_state_dict", "config", "val_mae", "epoch")
SLIM_SUFFIX: str = "-slim"


def slim_one(path: Path, out_dir: "Path | None" = None) -> Path:
    """Write a slim copy of one checkpoint, return its path.

    Keeps only the inference-relevant keys (``KEEP_KEYS``) and drops the
    optimizer state. Asserts the result is still loadable by the warm-start app.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    slim = {k: ckpt[k] for k in KEEP_KEYS if k in ckpt}
    assert "model_state_dict" in slim and "config" in slim, (
        f"{path.name} lacks model_state_dict/config, not a GNN checkpoint"
    )
    out = (out_dir or path.parent) / f"{path.stem}{SLIM_SUFFIX}.pt"
    torch.save(slim, out)
    before = path.stat().st_size / 1e6
    after = out.stat().st_size / 1e6
    print(f"{path.name}: {before:.1f} MB -> {after:.1f} MB  ({out.name})")
    return out


def main() -> None:
    """Slim one checkpoint, or every checkpoint in a directory with --all."""
    parser = argparse.ArgumentParser(
        description="strip optimizer state from trained checkpoints",
    )
    parser.add_argument(
        "target", type=Path,
        help="a .pt checkpoint, or a directory when --all is set",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="treat target as a directory and slim every .pt in it",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="write slim copies here (default: alongside the originals)",
    )
    args = parser.parse_args()

    if args.all:
        assert args.target.is_dir(), f"not a directory: {args.target}"
        paths = sorted(
            p for p in args.target.glob("*.pt")
            if not p.stem.endswith(SLIM_SUFFIX)
        )
        assert paths, f"no .pt checkpoints in {args.target}"
    else:
        assert args.target.is_file(), f"not a file: {args.target}"
        paths = [args.target]

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        slim_one(path, args.out_dir)
    print(f"\nslimmed {len(paths)} checkpoint(s)")


if __name__ == "__main__":
    main()
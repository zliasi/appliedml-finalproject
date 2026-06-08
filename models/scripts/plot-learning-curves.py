"""Learning-curve figures for the magmom models, from the wandb history.

Same idea as the tutorial's training-curve plot: train loss (left axis) and
validation MAE (right axis) versus epoch. The per-epoch history is already
logged to wandb by the training run (the --wandb callback logs epoch,
train_loss, val_mae, ...), so this just pulls it and draws it in the report
style. Self-contained: imports nothing from the pipeline, so it is safe to run
while training is still in flight.

Run from models/ (in the magmom21 env, login node):

    # one dual-axis curve per matched model:
    python scripts/plot-learning-curves.py --models dimenet-r8 schnetconv-r6 cgconv-r3
    # plus an overlay of val MAE for several models on one figure:
    python scripts/plot-learning-curves.py --models dimenet-r8 schnetconv-r6 cgconv-r3 nnconv-r6 --overlay

PNGs go to runs/<project>/eval/curves/ by default (override with --out).
"""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT = "cheac/magmom-magmom21-v0p1"
# default output: alongside the run's other eval figures
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "runs" / PROJECT.split("/")[-1] / "eval" / "curves"
)
UNIT = "muB"
UNIT_MATHTEXT = {"muB": r"$\mu_\mathrm{B}$"}
COLOR_TRAIN = "#1065ab"   # blue, matches src/plots.py
COLOR_VAL = "#b2182b"     # red
FONT_SIZE = 7.0
LINEWIDTH = 1.0
FIGSIZE = (3.6, 2.6)
DPI = 400


def style() -> None:
    """Project figure rcParams (mirrors src/plots.py)."""
    mpl.rcParams.update({
        "mathtext.fontset": "custom",
        "font.size": FONT_SIZE, "axes.linewidth": LINEWIDTH,
        "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE - 1,
        "savefig.dpi": DPI, "figure.dpi": DPI,
    })


def unit_label() -> str:
    """muB rendered as a Bohr-magneton symbol."""
    return UNIT_MATHTEXT.get(UNIT, UNIT)


def fetch_history(run) -> tuple[list, list, list]:
    """Pull (epoch, train_loss, val_mae) from a wandb run, sorted by epoch."""
    rows = []
    for row in run.scan_history(keys=["epoch", "train_loss", "val_mae"]):
        if row.get("epoch") is None:
            continue
        rows.append((row["epoch"], row.get("train_loss"), row.get("val_mae")))
    rows.sort(key=lambda r: r[0])
    epochs = [r[0] for r in rows]
    train = [r[1] for r in rows]
    val = [r[2] for r in rows]
    return epochs, train, val


def plot_dual_axis(epochs, train, val, out_path: Path, label: str) -> None:
    """Tutorial-style dual-axis curve: train loss (left) + val MAE (right)."""
    style()
    fig, ax_l = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax_l.plot(epochs, train, color=COLOR_TRAIN, linewidth=LINEWIDTH)
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Train loss (Huber)", color=COLOR_TRAIN)
    ax_l.tick_params(axis="y", labelcolor=COLOR_TRAIN, direction="in")
    ax_l.tick_params(axis="x", direction="in", top=True)
    ax_r = ax_l.twinx()
    ax_r.plot(epochs, val, color=COLOR_VAL, linewidth=LINEWIDTH)
    ax_r.set_ylabel(f"Validation MAE ({unit_label()})", color=COLOR_VAL)
    ax_r.tick_params(axis="y", labelcolor=COLOR_VAL, direction="in")
    ax_l.set_title(label, fontsize=FONT_SIZE)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_overlay(curves: list, out_path: Path) -> None:
    """Overlay validation-MAE curves for several models on one figure."""
    style()
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for label, epochs, val in curves:
        ax.plot(epochs, val, linewidth=LINEWIDTH, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Validation MAE ({unit_label()})")
    ax.legend(frameon=False)
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def short(name: str) -> str:
    """Trim a run name to backend-cutoff (drop the dataset/size suffix)."""
    return name.replace("magmom-", "").replace(
        "-c128l3h1-magmom21-v0p1-17k", "",
    )


def main() -> None:
    """Pull matched runs from wandb and draw learning-curve figures."""
    parser = argparse.ArgumentParser(description="magmom learning curves")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="substrings matched against run names, e.g. schnetconv-r8 dimenet-r6",
    )
    parser.add_argument(
        "--overlay", action="store_true",
        help="also draw one figure overlaying the matched models' val MAE",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import wandb
    api = wandb.Api()
    runs = list(api.runs(args.project))

    overlay = []
    for pattern in args.models:
        matched = [r for r in runs if pattern in r.name]
        if not matched:
            print(f"no run matching '{pattern}'")
            continue
        run = matched[0]
        epochs, train, val = fetch_history(run)
        if not epochs:
            print(f"{run.name}: no epoch history yet (still starting?)")
            continue
        label = short(run.name)
        out = args.out / f"curve-{label}.png"
        plot_dual_axis(epochs, train, val, out, label)
        best = min(v for v in val if v is not None)
        print(f"{label:24} {len(epochs)} epochs, best val MAE {best:.4f}  -> {out}")
        overlay.append((label, epochs, val))

    if args.overlay and overlay:
        out = args.out / "curve-overlay-valmae.png"
        plot_overlay(overlay, out)
        print(f"overlay -> {out}")


if __name__ == "__main__":
    main()

"""Shared figure style and parity/error plots.

Used by 05-submit-evaluate.sh (GNN checkpoints) and the baseline worker, so both produce
identically styled parity scatters and signed-error histograms.
"""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIGURE_SIZE: tuple[float, float] = (3.25, 2.17)
FIGURE_SIZE_SQUARE: tuple[float, float] = (3.25, 3.25)
FIGURE_DPI: int = 400
FONT_SIZE: float = 7.0
LINEWIDTH: float = 1.0
MARKER_SIZE: float = 10.0
MARKER_EDGE_WIDTH: float = 0.3
COLOR_FILL: str = "#1065ab"
COLOR_EDGE: str = "black"
# Units that need mathtext rendering (e.g. Bohr magneton as a proper symbol).
UNIT_MATHTEXT: dict[str, str] = {"muB": r"$\mu_\mathrm{B}$"}


def _format_unit(unit: str) -> str:
    """Render a unit string for axis labels, mathtext where needed."""
    return UNIT_MATHTEXT.get(unit, unit)


def apply_figure_style() -> None:
    """Set rcParams to the project figure style."""
    mpl.rcParams.update({
        "mathtext.fontset": "custom",
        "font.size": FONT_SIZE,
        "axes.linewidth": LINEWIDTH,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "savefig.dpi": FIGURE_DPI,
        "figure.dpi": FIGURE_DPI,
    })


def style_axes(ax: "mpl.axes.Axes") -> None:
    """Inward ticks on all four sides, no grid."""
    ax.tick_params(
        axis="both", top=True, right=True, direction="in",
    )


def _annotate_parity(
    ax: "mpl.axes.Axes", metrics: dict, prop_label: str, unit: str,
) -> None:
    """Common axes config for parity plots."""
    unit = _format_unit(unit)
    ax.set_xlabel(f"True {prop_label} ({unit})")
    ax.set_ylabel(f"Predicted {prop_label} ({unit})")
    text = (
        f"MAE = {metrics['mae']:.3f} {unit}\n"
        f"RMSE = {metrics['rmse']:.3f} {unit}\n"
        f"R$^2$ = {metrics['r2']:.3f}\n"
        f"N = {metrics['n_samples']}"
    )
    ax.annotate(
        text,
        xy=(0.05, 0.95), xycoords="axes fraction",
        verticalalignment="top", fontsize=FONT_SIZE,
        bbox=dict(
            boxstyle="round", facecolor="white",
            edgecolor="none", alpha=0.8,
        ),
    )
    style_axes(ax)


def plot_parity(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    out_path: Path,
    metrics: dict,
    prop_label: str,
    unit: str,
) -> None:
    """Predicted-vs-true scatter with y=x reference and metric box."""
    apply_figure_style()
    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE_SQUARE, dpi=FIGURE_DPI,
    )
    ax.scatter(
        y_true, y_pred,
        s=MARKER_SIZE, color=COLOR_FILL, edgecolors=COLOR_EDGE,
        linewidths=MARKER_EDGE_WIDTH, alpha=1.0, zorder=3,
    )
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    margin = (hi - lo) * 0.05
    lims = [lo - margin, hi + margin]
    ax.plot(lims, lims, "k--", linewidth=LINEWIDTH, zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_box_aspect(1)
    _annotate_parity(ax, metrics, prop_label, unit)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_error_distribution(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    out_path: Path,
    prop_label: str,
    unit: str,
) -> None:
    """Histogram of signed errors."""
    apply_figure_style()
    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE, dpi=FIGURE_DPI,
    )
    ax.hist(
        y_pred - y_true, bins=40,
        facecolor=COLOR_FILL, edgecolor=COLOR_EDGE,
        linewidth=LINEWIDTH, alpha=1.0,
    )
    ax.set_xlabel(f"Predicted - true {prop_label} ({_format_unit(unit)})")
    ax.set_ylabel("Count")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
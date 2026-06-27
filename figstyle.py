"""Shared manuscript-figure style — imported by every make_figN.py.

One source of truth for fonts, sizes, spines, the Okabe-Ito colour-blind-safe
palette, and the panel-label helper, so all six figures are visually consistent.

Fixed colour ROLES (use the role, never a raw hex, in figure code):
    STEP    = '#0072B2'  blue        (stepping / 2-state jump)
    RAMP    = '#D55E00'  vermillion  (ramping / drift-diffusion)
    NEUTRAL = '#555555'  grey        (reference / machine-precision band)
    ACCENT  = '#009E73'  green       (callouts / reference lines)
"""
from __future__ import annotations

import matplotlib as mpl

# --- Okabe-Ito palette, fixed role -> colour ---------------------------------
PALETTE = {
    "STEP": "#0072B2",      # blue
    "RAMP": "#D55E00",      # vermillion
    "NEUTRAL": "#555555",   # grey
    "ACCENT": "#009E73",    # bluish green
}

# sequential, colour-blind-safe colormap for heatmaps / density panels (added for Fig 2)
SEQ_CMAP = "cividis"


def apply_style() -> None:
    """Install the manuscript rcParams (idempotent; called on import)."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,        # axis labels 8pt
        "axes.titlesize": 9,
        "xtick.labelsize": 7,       # ticks 7pt
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,     # 1.2 linewidth
        "axes.linewidth": 0.8,
        "axes.spines.top": False,   # top/right spines off
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.autolayout": False,           # constrained_layout (below) manages spacing
        "figure.constrained_layout.use": True,  # auto anti-overlap layout for all figures
        "savefig.dpi": 300,                   # 300 dpi raster
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,                   # editable vector text in the PDF
        "ps.fonttype": 42,
        # readable, semi-opaque legend frame everywhere
        "legend.framealpha": 0.9,
        "legend.facecolor": "white",
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": True,
    })


def panel_label(ax, letter: str, dx: float = -0.07, dy: float = 1.03) -> None:
    """Place a bold lowercase panel letter at the top-left, just outside the axes."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="right", clip_on=False)


# --- anti-overlap readability helpers (shared by all figures) ----------------
def safe_text(ax, x, y, s, **kw):
    """ax.text with a default white rounded halo so labels stay legible over
    lines, markers and heatmap cells. Any kw (incl. bbox) overrides the default."""
    kw.setdefault("bbox", dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    return ax.text(x, y, s, **kw)


def safe_legend(ax, outside=False, **kw):
    """Legend with the white semi-opaque manuscript frame. If outside=True, park it
    just outside the top-right of the axes (clear of the data)."""
    kw.setdefault("framealpha", 0.9)
    kw.setdefault("facecolor", "white")
    kw.setdefault("edgecolor", "#cccccc")
    kw.setdefault("fancybox", True)
    if outside:
        kw.setdefault("bbox_to_anchor", (1.02, 1.0))
        kw.setdefault("loc", "upper left")
    return ax.legend(**kw)


# install on import so `import figstyle` is enough
apply_style()

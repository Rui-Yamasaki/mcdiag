"""Figure 4 - the decision signal at the boundary, not below it. Values read live from results/.

  (a) per-region movement-controlled choice decode AUC under the valid (linear) control versus the
      over-correcting (expanded) control, both datasets, with 95% CIs, against the 0.57 bar
        -> results/referee_response/corrected_results/corrected_decode_by_region.csv
  (b) the cost of over-correction: movement-independent survivor counts by control, showing the
      failure runs both ways (expanded over-removes, pca under-removes, linear is the valid middle)
        -> results/referee_response/corrected_results/overcorrection_contrast.csv

Run:  python make_fig4.py   -> figures/fig4.{png,pdf}
Reads cached results/ only; no refits, no downloads. Double-column width (180 mm).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import figstyle  # noqa: E402
from figstyle import PALETTE, panel_label, safe_text  # noqa: E402
import figcheck  # noqa: E402

ROOT = Path(__file__).resolve().parent
REF = ROOT / "results" / "referee_response" / "corrected_results"
CONTRAST = REF / "overcorrection_contrast.csv"
FIGDIR = ROOT / "figures"
MM = 1 / 25.4
BAR, CHANCE = 0.57, 0.50
VALID = "#0072B2"        # linear (valid) control, Okabe-Ito blue
BROKEN = "#D55E00"       # expanded (over-correcting) control, vermillion

# regions to show (top to bottom); IBL SNr excluded (only 1 decodable session)
ROWS = [("IBL", "MRN"), ("IBL", "SCm"), ("IBL", "IRN"), ("IBL", "GRN"),
        ("Steinmetz", "MRN"), ("Steinmetz", "SC"), ("Steinmetz", "SNr")]


# ============================================================ (a) decode forest, valid vs broken
def panel_a(ax):
    d = pd.read_csv(REF / "corrected_decode_by_region.csv")

    def get(ds, reg, ctl):
        r = d[(d.dataset == ds) & (d.region == reg) & (d.control == ctl)].iloc[0]
        return float(r.decode_auc), float(r.decode_lo), float(r.decode_hi), int(r.n_sessions)

    n = len(ROWS)
    yc = np.arange(n)[::-1].astype(float)                  # row centres, top to bottom
    off = 0.17
    ax.axvline(CHANCE, ls=":", lw=1.0, color="#333333", zorder=1)
    ax.axvline(BAR, ls="--", lw=1.1, color="#333333", zorder=1)
    for (ds, reg), y in zip(ROWS, yc):
        for ctl, col, mk, dy, fill in (("linear", VALID, "o", off, True),
                                       ("expanded", BROKEN, "s", -off, False)):
            a, lo, hi, _ = get(ds, reg, ctl)
            ax.errorbar(a, y + dy, xerr=[[a - lo], [hi - a]], fmt=mk, ms=5.5, color=col,
                        mfc=col if fill else "white", mec=col, mew=1.2, capsize=2.5,
                        elinewidth=1.0, zorder=4)
    ax.set_yticks(yc)
    ax.set_yticklabels([reg for _, reg in ROWS])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0.40, 0.76)
    ax.set_xlabel("movement-controlled choice decode AUC")
    ax.set_title("decode by region: valid vs over-correcting control", fontsize=8.5, loc="left")
    # dataset section brackets on the far left
    ds_groups = {}
    for (ds, _), y in zip(ROWS, yc):
        ds_groups.setdefault(ds, []).append(y)
    for ds, ys in ds_groups.items():
        ax.text(-0.135, np.mean(ys), ds, transform=ax.get_yaxis_transform(), rotation=90,
                ha="center", va="center", fontsize=7.5, fontweight="bold", color="#333333",
                clip_on=False)
        ax.plot([-0.115, -0.115], [min(ys) - 0.32, max(ys) + 0.32], transform=ax.get_yaxis_transform(),
                color="#333333", lw=1.0, clip_on=False, zorder=2)
    safe_text(ax, BAR + 0.004, -0.5, "0.57 bar", fontsize=6.2, va="center", ha="left",
              color="#333333")
    safe_text(ax, CHANCE - 0.004, -0.5, "0.50 chance", fontsize=6.2, va="center", ha="right",
              color="#333333")
    ax.legend(handles=[Line2D([0], [0], marker="o", color=VALID, mfc=VALID, ms=5.5, lw=0,
                              label="linear (valid)"),
                       Line2D([0], [0], marker="s", color=BROKEN, mfc="white", mec=BROKEN, ms=5.5,
                              lw=0, label="expanded (over-corrects)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=6.8,
              frameon=True, framealpha=0.9, edgecolor="#cccccc", columnspacing=1.5)
    return d


def panel_b(ax):
    c = pd.read_csv(CONTRAST)
    al = c[(c.dataset == "IBL") & (c.region == "ALL")].iloc[0]
    bars = [("expanded\n(over-removes)", int(al.movesurv_expanded), BROKEN, "//"),
            ("linear\n(valid)", int(al.movesurv_linear), PALETTE["ACCENT"], ""),
            ("pca\n(under-removes)", int(al.movesurv_pca_expanded), "#E69F00", "..")]
    x = np.arange(len(bars))
    for xi, (lab, val, col, hatch) in zip(x, bars):
        ax.bar(xi, val, 0.64, color=col, edgecolor="white", linewidth=0.6, hatch=hatch, zorder=3)
        ax.text(xi, val + 4, str(val), ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.axhline(bars[1][1], ls=(0, (4, 3)), lw=0.9, color=PALETTE["ACCENT"], alpha=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars])
    ax.get_xticklabels()[1].set_fontweight("bold")
    ax.set_ylim(0, 222)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("movement-independent\nchoice cells (of 301)")
    ax.set_title("cost of over-correction", fontsize=8.5, loc="left")
    return dict(expanded=int(al.movesurv_expanded), linear=int(al.movesurv_linear),
                pca=int(al.movesurv_pca_expanded))


# ============================================================ assemble
def main():
    fig = plt.figure(figsize=(180 * MM, 88 * MM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.5, 0.75])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_a(ax_a)
    info = panel_b(ax_b)
    panel_label(ax_a, "a", dx=-0.10, dy=1.04)
    panel_label(ax_b, "b", dx=-0.28, dy=1.04)
    fig.suptitle("Figure 4  ·  the decision signal at the boundary, not below it",
                 fontsize=10, fontweight="bold", x=0.01, ha="left")

    fig.canvas.draw()
    clean = figcheck.report(fig, "Figure 4")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig4.png")
    fig.savefig(FIGDIR / "fig4.pdf")
    plt.close(fig)
    print(f"saved -> {FIGDIR/'fig4.png'} (+ .pdf)   [{'CLEAN' if clean else 'HAS OVERLAPS'}]  "
          f"(survivors expanded {info['expanded']} / linear {info['linear']} / pca {info['pca']})")


if __name__ == "__main__":
    main()

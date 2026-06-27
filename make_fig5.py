"""Figure 5 - a small movement-independent choice code (under the validated control).

  (a) the cascade under the valid (linear) movement control with within-region FDR: choice-selective,
      movement-independent, stimulus/prior-independent, and triple-coded survivor counts
      -> results/referee_response/corrected_results/corrected_cascade.csv
  (b) the surviving superior colliculus (SCm) cells: per-cell choice decode AUC before versus after
      the valid movement control -> exemplar_cells.csv + phase2_sel_cells_full.csv (raw AUC)
  (c) per-cell choice decode AUC across the population under the valid control
      -> results/referee_response/corrected_results/percell_pmove.csv

Run:  python make_fig5.py   -> figures/fig5.{png,pdf}
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

import figstyle  # noqa: E402
from figstyle import PALETTE, panel_label, safe_text  # noqa: E402
import figcheck  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REF = RESULTS / "referee_response" / "corrected_results"
FIGDIR = ROOT / "figures"
MM = 1 / 25.4

SCM = "#0072B2"          # superior colliculus survivors (Okabe-Ito blue)
OTHER = "#bbbbbb"        # other regions / non-selective bulk (grey)
BAR = 0.57               # recoverability bar
CHANCE = 0.50

SCM_CELLS = ["478de1ce-d7e7-4221-9365-2abdc6e88fb6:410",
             "53ecbf4f-e0d8-4fe6-a852-8b934a37a1c2:600",
             "d5e5311c-8beb-4f8f-b798-3e9bfa6bcdd8:275"]


# ============================================================ (a) the cascade
def panel_a(ax):
    c = pd.read_csv(REF / "corrected_cascade.csv")
    sub = c[(c.dataset == "IBL") & (c.scheme == "within-region BH")
            & (c.control == "published(linear)")]
    allr = sub[sub.region == "ALL"].iloc[0]
    scm = sub[sub.region == "SCm"].iloc[0]
    stages = [("choice-\nselective", int(allr.choice_fdr), int(scm.choice_fdr)),
              ("movement-\nindependent", int(allr.move_indep_fdr), int(scm.move_indep_fdr)),
              ("stim / prior-\nindependent", int(allr.stim_indep_fdr), int(scm.stim_indep_fdr)),
              ("triple-\ncoded", int(allr.triple_fdr), int(scm.triple_fdr))]
    y = np.arange(len(stages))[::-1]           # top to bottom
    for yi, (lab, tot, sc) in zip(y, stages):
        ax.barh(yi, tot, height=0.6, color=OTHER, edgecolor="white", linewidth=0.6, zorder=3)
        ax.barh(yi, sc, height=0.6, color=SCM, edgecolor="white", linewidth=0.6, zorder=4)
        txt = f"{tot}" if sc == tot else f"{tot}  (SCm {sc})"
        ax.text(tot + 2.5, yi, txt, va="center", ha="left", fontsize=7, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in stages])
    ax.set_xlim(0, 132)
    ax.set_ylim(-0.6, len(stages) - 0.4)
    ax.set_xlabel("cells (within-region FDR, q < 0.05)")
    ax.set_title("cascade, valid control", fontsize=8.5, loc="left")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SCM),
                       plt.Rectangle((0, 0), 1, 1, color=OTHER)],
              labels=["superior colliculus", "other regions"], loc="lower right",
              fontsize=6.6, frameon=True, framealpha=0.9, edgecolor="#cccccc")
    safe_text(ax, 0.97, 0.52, "movement and stim filters\nare parallel (not nested)",
              transform=ax.transAxes, ha="right", va="center", fontsize=6.0,
              color=PALETTE["NEUTRAL"])


# ============================================================ (b) SCm exemplar cells
def panel_b(ax):
    raw = pd.read_csv(RESULTS / "phase2_sel_cells_full.csv").set_index("cell")
    ex = pd.read_csv(REF / "exemplar_cells.csv").set_index("cell")
    before = [float(raw.loc[c, "auc_delib"]) for c in SCM_CELLS]
    after = [float(ex.loc[c, "auc_linear"]) for c in SCM_CELLS]
    frs = [float(ex.loc[c, "fr_window"]) for c in SCM_CELLS]
    x0, x1 = 0.0, 1.0
    for b, a, fr in zip(before, after, frs):
        ax.plot([x0, x1], [b, a], "-", color=SCM, lw=1.6, marker="o", ms=6,
                mfc=SCM, mec="white", mew=0.8, zorder=4)
    ax.axhline(BAR, ls="--", lw=1.1, color="#333333", zorder=2)
    ax.axhline(CHANCE, ls=":", lw=1.0, color=PALETTE["NEUTRAL"], zorder=2)
    safe_text(ax, 0.5, BAR + 0.006, "0.57 recoverability bar", fontsize=6.4, va="bottom",
              ha="center", color="#333333")
    safe_text(ax, 0.5, CHANCE + 0.006, "chance", fontsize=6.4, va="bottom", ha="center",
              color=PALETTE["NEUTRAL"])
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["before\ncontrol", "after\nlinear control"])
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(0.47, 0.86)
    ax.set_ylabel("per-cell choice decode AUC")
    ax.set_title("SCm survivors (n = 3)", fontsize=8.5, loc="left")


# ============================================================ (c) per-cell AUC distribution
def panel_c(ax):
    p = pd.read_csv(REF / "percell_pmove.csv")
    a = p[p.dataset == "IBL"]["auc_move_linear"].dropna().to_numpy()
    med = float(np.median(a))
    hi = float((a > 0.65).mean())
    lo = float((a < 0.35).mean())
    bins = np.arange(0.20, 0.815, 0.025)
    sel = (a > 0.65) | (a < 0.35)
    ax.hist(a[~sel], bins=bins, color=OTHER, edgecolor="white", linewidth=0.3, zorder=3,
            label="near chance")
    ax.hist(a[sel], bins=bins, color=SCM, edgecolor="white", linewidth=0.3, zorder=4,
            label="selective")
    ax.axvline(CHANCE, ls=":", lw=1.1, color="#333333", zorder=5)
    for t in (0.35, 0.65):
        ax.axvline(t, ls="--", lw=0.9, color=PALETTE["NEUTRAL"], zorder=5)
    ax.set_xlim(0.20, 0.82)
    ymax = ax.get_ylim()[1] * 1.30
    ax.set_ylim(0, ymax)
    ax.set_xlabel("choice decode AUC\n(after linear control)")
    ax.set_ylabel("number of cells")
    ax.set_title("per-cell AUC distribution", fontsize=8.5, loc="left")
    ax.legend(loc="upper right", fontsize=6.4, frameon=True, framealpha=0.9, edgecolor="#cccccc",
              handlelength=1.2, borderpad=0.4)
    safe_text(ax, 0.215, ymax * 0.96, f"median {med:.2f}", ha="left", va="top",
              fontsize=6.6, color="#333333")
    ax.annotate(f"{hi:.1%} > 0.65", xy=(0.685, ymax * 0.07),
                xytext=(0.735, ymax * 0.52), fontsize=6.4, color=SCM, ha="center",
                arrowprops=dict(arrowstyle="->", color=SCM, lw=0.8))
    return med, hi, lo


# ============================================================ assemble
def main():
    fig = plt.figure(figsize=(180 * MM, 72 * MM))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 0.8, 1.15])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a(ax_a)
    panel_b(ax_b)
    med, hi, lo = panel_c(ax_c)
    panel_label(ax_a, "a", dx=-0.34, dy=1.06)
    panel_label(ax_b, "b", dx=-0.30, dy=1.06)
    panel_label(ax_c, "c", dx=-0.20, dy=1.06)
    fig.suptitle("Figure 5  ·  a small movement-independent choice code",
                 fontsize=10, fontweight="bold", x=0.01, ha="left")

    fig.canvas.draw()
    clean = figcheck.report(fig, "Figure 5")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig5.png")
    fig.savefig(FIGDIR / "fig5.pdf")
    plt.close(fig)
    print(f"saved -> {FIGDIR/'fig5.png'} (+ .pdf)   [{'CLEAN' if clean else 'HAS OVERLAPS'}]  "
          f"(c: median {med:.3f}, {hi:.1%}>0.65, {lo:.1%}<0.35)")


if __name__ == "__main__":
    main()

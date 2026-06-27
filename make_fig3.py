"""Figure 3 - movement controls distort the decision-signal estimate (the calibration centerpiece).

  (a) the three control-validity tests per candidate control (no-signal, preserve a movement-
      orthogonal signal, remove a pure-movement signal) on the synthetic calibration
      -> results/referee_response/proper_control/calibration.csv
  (b) controlled decode AUC versus injected per-cell effect size, one line per control, at the
      measured choice-to-movement correlation -> same calibration grid (rho = 0.493 rows)

Two-panel layout (a stacked over b). The former panel (c) real-data survivor-count comparison was
removed because it duplicates Fig 4b; its quantitative content lives in Extended Data Table 1.

Run:  python make_fig3.py   -> figures/fig3.{png,pdf}
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
RESULTS = ROOT / "results" / "referee_response"
FIGDIR = ROOT / "figures"
MM = 1 / 25.4

# control display names and a fixed colour + marker per control (Okabe-Ito, colour-blind safe)
CTRL = {                          # csv name -> (label, colour, marker)
    "none":              ("none",      "#999999", "o"),
    "linear":            ("linear",    "#0072B2", "s"),   # the validated control
    "ridge_expanded":    ("ridge",     "#56B4E9", "^"),
    "pca_expanded":      ("pca",       "#E69F00", "D"),
    "expanded":          ("expanded",  "#D55E00", "v"),
    "crossfit_expanded": ("crossfit",  "#CC79A7", "X"),
}
ORDER = ["none", "linear", "ridge_expanded", "pca_expanded", "expanded", "crossfit_expanded"]
BAR = 0.57          # recoverability bar
CHANCE = 0.50


def load(name):
    return pd.read_csv(RESULTS / name)


# ============================================================ (a) the three validity tests
def panel_a(ax, cal):
    """Grouped bars: per control, the no-signal / preserve / remove test AUCs. Valid pattern is
    no-signal and remove at chance, preserve high (near the uncontrolled level)."""
    def row(d, rho, ctl):
        r = cal[(np.isclose(cal.d, d)) & (np.isclose(cal.rho, rho)) & (cal.control == ctl)].iloc[0]
        return float(r.auc), float(r.lo), float(r.hi)

    def at(d, rho, ctl):
        return row(d, rho, ctl)[0]

    tests = [("no-signal", 0.0, 0.0, PALETTE["NEUTRAL"], "...."),
             ("preserve",  0.24, 0.0, PALETTE["STEP"], ""),
             ("remove",    0.24, 1.0, PALETTE["RAMP"], "//")]
    x = np.arange(len(ORDER))
    w = 0.26
    for k, (name, d, rho, col, hatch) in enumerate(tests):
        vals = np.array([row(d, rho, c) for c in ORDER])
        err = np.abs(vals[:, 1:].T - vals[:, 0])      # 95% CI as asymmetric error bars
        ax.bar(x + (k - 1) * w, vals[:, 0], w, color=col, edgecolor="white", linewidth=0.5,
               hatch=hatch, label=name, zorder=3,
               yerr=err, error_kw=dict(elinewidth=0.6, ecolor="#444444", capsize=0))
    unc_preserve = at(0.24, 0.0, "none")
    ax.axhline(CHANCE, ls=":", lw=1.0, color="#333333", zorder=2)
    ax.axhline(unc_preserve, ls=(0, (4, 3)), lw=0.9, color=PALETTE["STEP"], alpha=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([CTRL[c][0] for c in ORDER])
    ax.get_xticklabels()[ORDER.index("linear")].set_fontweight("bold")
    ax.set_ylim(0.43, 0.73)
    ax.set_xlim(-0.6, len(ORDER) + 0.55)
    ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ax.set_ylabel("decode AUC")
    ax.set_title("control-validity tests on the synthetic calibration", fontsize=8.5, loc="left")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=6.8,
              columnspacing=1.1, handlelength=1.4, frameon=True, framealpha=0.9,
              edgecolor="#cccccc")
    safe_text(ax, len(ORDER) + 0.5, CHANCE, "chance", fontsize=6.2, va="center", ha="right",
              color="#333333")
    safe_text(ax, len(ORDER) + 0.5, unc_preserve, "uncontrolled\npreserve", fontsize=6.0,
              va="center", ha="right", color=PALETTE["STEP"])
    # mark the two failures with a small star above the relevant bar (caption explains)
    star = dict(fontsize=9, ha="center", va="bottom", fontweight="bold")
    ax.text(ORDER.index("expanded"), at(0.24, 0.0, "expanded") + 0.004, "*",
            color=PALETTE["STEP"], **star)            # preserve fails (over-corrects)
    ax.text(ORDER.index("pca_expanded") + w, at(0.24, 1.0, "pca_expanded") + 0.004, "*",
            color=PALETTE["RAMP"], **star)            # remove leaves movement (under-removes)


# ============================================================ (b) recovery vs injected size
def panel_b(ax, cal):
    sub = cal[np.isclose(cal.rho, 0.493)].sort_values("d")
    for c in ORDER:
        s = sub[sub.control == c]
        lab, col, mk = CTRL[c]
        lw = 2.2 if c == "linear" else 1.3
        z = 6 if c == "linear" else 4
        ax.fill_between(s.d, s.lo, s.hi, color=col, alpha=0.12, lw=0, zorder=z - 1)
        ax.plot(s.d, s.auc, marker=mk, color=col, lw=lw, ms=4.5, label=lab, zorder=z)
    ax.axhline(BAR, ls="--", lw=1.1, color="#333333", zorder=2)
    ax.axhline(CHANCE, ls=":", lw=1.0, color=PALETTE["NEUTRAL"], zorder=2)
    ax.axvline(0.24, ls=(0, (1, 2)), lw=0.8, color="#888888", zorder=1)
    safe_text(ax, 0.004, BAR + 0.003, "0.57 recoverability bar", fontsize=6.2, va="bottom",
              ha="left", color="#333333")
    safe_text(ax, 0.004, CHANCE + 0.003, "chance", fontsize=6.2, va="bottom", ha="left",
              color=PALETTE["NEUTRAL"])
    safe_text(ax, 0.243, 0.467, "threshold\n0.24 SD", fontsize=6.0, va="bottom", ha="left",
              color="#888888")
    ax.set_xlabel("injected per-cell effect size (SD)")
    ax.set_ylabel("movement-controlled decode AUC")
    ax.set_xlim(-0.012, 0.315)
    ax.set_ylim(0.46, 0.65)
    ax.set_xticks([0, 0.06, 0.12, 0.18, 0.24, 0.30])
    ax.set_yticks([0.50, 0.55, 0.60, 0.65])
    ax.set_title("recovery vs injected signal (realistic rho = 0.49)", fontsize=8.5, loc="left")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.30), fontsize=6.4,
              columnspacing=1.1, handlelength=1.6, frameon=True, framealpha=0.9,
              edgecolor="#cccccc")


# ============================================================ assemble
def main():
    cal = load("proper_control/calibration.csv")

    # Two panels, stacked: (a) the grouped control-validity bars want the full 180 mm width;
    # (b) the dose-response sits below it. 180 mm double-column.
    fig = plt.figure(figsize=(176 * MM, 150 * MM))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.16])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    panel_a(ax_a, cal)
    panel_b(ax_b, cal)
    panel_label(ax_a, "a", dx=-0.045, dy=1.04)
    panel_label(ax_b, "b", dx=-0.045, dy=1.05)
    fig.suptitle("Figure 3  ·  movement controls distort the decision-signal estimate",
                 fontsize=10, fontweight="bold", x=0.01, ha="left")

    fig.canvas.draw()
    clean = figcheck.report(fig, "Figure 3")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig3.png")
    fig.savefig(FIGDIR / "fig3.pdf")
    plt.close(fig)
    print(f"saved -> {FIGDIR/'fig3.png'} (+ .pdf)   [{'CLEAN' if clean else 'HAS OVERLAPS'}]")


if __name__ == "__main__":
    main()

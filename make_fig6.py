"""Figure 6 - single-trial dynamics recovery is out of reach at current yields, and the at-boundary
decode replicates. Values read live from results/.

  (a) full-cohort simultaneity census: per-region maximum simultaneous good-QC unit count against the
      120-cell population-recovery requirement
        -> results/referee_response/full_census.csv
  (b) replication: IBL MRN vs Steinmetz MRN movement-controlled (linear) decode AUC, both at the 0.57
      boundary -> results/referee_response/corrected_results/corrected_decode_by_region.csv
  (c) single-cell route: step-vs-ramp recovery vs pooled trial count N at 20 Hz, with the IBL
      operating point -> phase1_recovery_nb_grid_R40.csv

Run:  python make_fig6.py   -> figures/fig6.{png,pdf}
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
REF = RESULTS / "referee_response"
FIGDIR = ROOT / "figures"
MM = 1 / 25.4
BAR, CHANCE, TARGET = 0.57, 0.50, 0.80
REQ_CELLS = 120
HILITE = "#B2182B"
DATACOL = {"IBL": "#785EF0", "Steinmetz": "#DC267F"}    # colour-blind-safe, not STEP/RAMP hues
# region colours, qualitative colour-blind-safe (shared with the recovery-wall convention)
REGCOL = {"MRN": "#009E73", "SCm": "#E69F00", "GRN": "#555555", "IRN": "#CC79A7", "SNr": "#56B4E9"}


# ============================================================ (a) full-cohort census
def panel_a(ax):
    d = pd.read_csv(REF / "full_census.csv").sort_values("max_simul_goodQC", ascending=False)
    regions = list(d.region)
    x = np.arange(len(regions))
    for xi, (_, r) in zip(x, d.iterrows()):
        ax.bar(xi, r.max_simul_goodQC, 0.66, color=REGCOL.get(r.region, "#888"),
               edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(xi, r.max_simul_goodQC + 2.5, int(r.max_simul_goodQC), ha="center", va="bottom",
                fontsize=7.5, fontweight="bold")
    ax.axhline(REQ_CELLS, ls="--", lw=1.3, color=HILITE, zorder=4)
    safe_text(ax, len(regions) - 1.5, REQ_CELLS - 3, f"{REQ_CELLS}-cell requirement",
              ha="center", va="top", fontsize=6.6, color=HILITE, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(regions)
    ax.set_ylim(0, 150)
    ax.set_xlim(-0.6, len(regions) - 0.4)
    ax.set_ylabel("max simultaneous\ngood-QC units")
    ax.set_title("full-cohort simultaneity census", fontsize=8.5, loc="left")
    total_sessions = int(d.region_sessions.sum())
    safe_text(ax, 0.5, 0.97, f"0 of {total_sessions} region-sessions reach {REQ_CELLS}\n"
              f"(full cohort: 205 insertions, 176 sessions)", transform=ax.transAxes,
              ha="center", va="top", fontsize=6.0, color=PALETTE["NEUTRAL"])
    return dict(maxc=int(d.max_simul_goodQC.max()), nsess=total_sessions)


# ============================================================ (b) replication
def panel_b(ax):
    d = pd.read_csv(REF / "corrected_results" / "corrected_decode_by_region.csv")

    def get(ds):
        r = d[(d.dataset == ds) & (d.region == "MRN") & (d.control == "linear")].iloc[0]
        return float(r.decode_auc), float(r.decode_lo), float(r.decode_hi)

    ax.axhline(BAR, ls="--", lw=1.1, color="#333333", zorder=1)
    ax.axhline(CHANCE, ls=":", lw=1.0, color=PALETTE["NEUTRAL"], zorder=1)
    out = {}
    for x, ds in enumerate(["IBL", "Steinmetz"]):
        a, lo, hi = get(ds); out[ds] = (a, lo, hi)
        ax.errorbar(x, a, yerr=[[a - lo], [hi - a]], fmt="o", ms=8, color=DATACOL[ds],
                    capsize=4, elinewidth=1.4, zorder=4)
        ax.text(x, hi + 0.008, f"{a:.3f}", ha="center", va="bottom", fontsize=7,
                color=DATACOL[ds], fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["IBL\nMRN", "Steinmetz\nMRN"])
    ax.set_xlim(-0.6, 1.95)
    ax.set_ylim(0.46, 0.76)
    ax.set_ylabel("movement-controlled\ndecode AUC (linear)")
    ax.set_title("replication (MRN)", fontsize=8.5, loc="left")
    safe_text(ax, 1.62, BAR, "0.57 bar", ha="left", va="center", fontsize=6.2, color="#333333")
    safe_text(ax, 1.62, CHANCE, "0.50\nchance", ha="left", va="center", fontsize=6.2,
              color=PALETTE["NEUTRAL"])
    return out


# ============================================================ (c) single-cell recovery vs N
def panel_c(ax):
    nb = pd.read_csv(RESULTS / "phase1_recovery_nb_grid_R40.csv")
    fr20 = nb[(nb.fano >= 1.5) & (nb.fr == 20)].groupby("N")["recovery"].mean()
    Ns, rec = fr20.index.to_numpy(float), fr20.to_numpy(float)
    ax.plot(Ns, rec, "o-", color="#2b2b2b", lw=1.8, ms=5, zorder=4)
    ax.axhline(TARGET, ls="--", lw=1.0, color=PALETTE["NEUTRAL"], zorder=1)
    ax.axhline(CHANCE, ls=":", lw=1.0, color=PALETTE["NEUTRAL"], zorder=1)
    # IBL single-unit operating point (median session ~53 trials)
    ax.plot(53, 0.56, marker="*", ms=15, color=REGCOL["MRN"], mec="white", mew=0.7, zorder=6)
    ax.annotate("IBL single unit, ~53 tr/session\nper-cell AUC 0.53 (sub-bar)", xy=(53, 0.555),
                xytext=(78, 0.55), ha="left", va="top", fontsize=6.2, color=REGCOL["MRN"],
                arrowprops=dict(arrowstyle="->", color=REGCOL["MRN"], lw=0.8))
    ax.set_xlim(0, 345)
    ax.set_ylim(0.48, 1.0)
    ax.set_xlabel("pooled trial count  N")
    ax.set_ylabel("step-vs-ramp recovery rate")
    ax.set_title("single-cell route", fontsize=8.5, loc="left")
    safe_text(ax, 16, TARGET + 0.008, "0.80 recovery target", ha="left", va="bottom",
              fontsize=6.2, color=PALETTE["NEUTRAL"])
    return dict(Ns=Ns.tolist(), rec=[round(r, 3) for r in rec])


# ============================================================ assemble
def main():
    fig = plt.figure(figsize=(180 * MM, 70 * MM))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 0.72, 1.15])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    a = panel_a(ax_a); b = panel_b(ax_b); c = panel_c(ax_c)
    panel_label(ax_a, "a", dx=-0.22, dy=1.05)
    panel_label(ax_b, "b", dx=-0.34, dy=1.05)
    panel_label(ax_c, "c", dx=-0.20, dy=1.05)
    fig.suptitle("Figure 6  ·  single-trial dynamics recovery is out of reach at current yields",
                 fontsize=9.5, fontweight="bold", x=0.01, ha="left")

    fig.canvas.draw()
    clean = figcheck.report(fig, "Figure 6")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig6.png")
    fig.savefig(FIGDIR / "fig6.pdf")
    plt.close(fig)
    print(f"saved -> {FIGDIR/'fig6.png'} (+ .pdf)   [{'CLEAN' if clean else 'HAS OVERLAPS'}]")
    print(f"  (a) census max good-QC {a['maxc']} < {REQ_CELLS}; {a['nsess']} region-sessions")
    print(f"  (b) replication MRN linear: IBL {b['IBL']}  Steinmetz {b['Steinmetz']}")


if __name__ == "__main__":
    main()

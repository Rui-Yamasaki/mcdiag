"""Figure 2 — "the identifiability map" (constructive). All values aggregated live from results/.

  (a) HEATMAP    REALISTIC recovery over (pooled N) x (firing rate): NB emission, overdispersed
                 (Fano>=1.5), sigma=0.4, R=40  -> phase1_recovery_nb_grid_R40.csv
  (b) POISSON/NB same overdispersed data, two fitters (misspecified Poisson vs matched NB) vs N, R=40
                 -> phase1_recovery_poisson_misspec_R40.csv + phase1_recovery_nb_grid_R40.csv (FR20 row)
  (c) SIGMA      recovery vs ramp diffusion sigma (FR=20, K=50)  -> ramp_validation_robustness.csv
  (d) INTERMEDIACY  P(step model selected) vs generator stepiness -> phase1_recovery_stepiness.csv

The clean-Poisson FR x N map is rendered separately as Fig S1 (figS1.{png,pdf}) — the lenient
best case, not the operating regime.
Missing backing file -> that panel is a labelled "data not found" placeholder (never invented).

Run:  python make_fig2.py   -> figures/fig2.{png,pdf} + figures/figS1.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

import figstyle  # noqa: E402  (installs rcParams incl. constrained_layout)
from figstyle import PALETTE, SEQ_CMAP, panel_label, safe_text, safe_legend  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures"


# ---------------------------------------------------------------- shared reference lines
def draw_chance(ax):
    """0.5 chance line — identical everywhere (dotted, NEUTRAL)."""
    ax.axhline(0.5, ls=":", lw=1.0, color=PALETTE["NEUTRAL"], zorder=1)


def draw_target(ax):
    """0.80 recovery target line — identical everywhere (dashed, NEUTRAL)."""
    ax.axhline(0.80, ls="--", lw=1.0, color=PALETTE["NEUTRAL"], zorder=1)


def placeholder(ax, msg):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linestyle((0, (3, 3))); s.set_color(PALETTE["NEUTRAL"])
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color=PALETTE["NEUTRAL"])


def load(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------- shared heatmap drawer
def draw_recovery_heatmap(ax, fig, Z, FRs, Ns, emission_label, title):
    """Cividis FR x N recovery heatmap with the 0.80 contour, IBL operating point + pooling
    callout (thin arrow), and an emission/regime label. Returns {ibl_rec, target_N}."""
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=SEQ_CMAP, vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels([int(n) for n in Ns])
    ax.set_yticks(range(len(FRs))); ax.set_yticklabels([int(f) for f in FRs])
    ax.set_xlabel("pooled trial count  N"); ax.set_ylabel("mean firing rate (Hz)")
    ax.set_title(title, fontsize=8.5, loc="left")
    X, Y = np.meshgrid(range(len(Ns)), range(len(FRs)))
    cs = ax.contour(X, Y, Z, levels=[0.80], colors="white", linewidths=1.4, linestyles="--")
    ax.clabel(cs, fmt="0.80", fontsize=6.5, inline=True)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("recovery rate", fontsize=7.5); cb.ax.tick_params(labelsize=6.5)

    info = {"ibl_rec": None, "target_N": None}
    if 20.0 in FRs and 40 in Ns:
        iy, ix = FRs.index(20.0), Ns.index(40)
        info["ibl_rec"] = float(Z[iy, ix])
        # tol: the FR20/N320 cell is 4.8/6 = 0.8 exactly but represents as 0.7999999999999999 in
        # float64, so the 1e-9 tolerance lets the contour test register the true 0.80 crossing.
        clears = [n for n in Ns if Z[iy, Ns.index(n)] >= 0.80 - 1e-9]
        info["target_N"] = int(min(clears)) if clears else None
        ax.plot(ix, iy, marker="o", ms=8, mfc="none", mec="white", mew=1.8, zorder=6)
        tx, ty = ix + 1.45, iy - 1.95                       # callout in the low-recovery cells,
        ax.annotate("", xy=(ix, iy), xytext=(tx, ty),       # clear of the 0.80 contour (upper-right)
                    arrowprops=dict(arrowstyle="->", color="white", lw=0.9), zorder=6)
        tgt = f"pool to N≥{info['target_N']}" if info["target_N"] else "never clears 0.80"
        safe_text(ax, tx, ty, f"IBL single unit\n~40 tr · 20 Hz: {info['ibl_rec']:.0%}\n{tgt}",
                  fontsize=6.2, ha="center", va="center")
    safe_text(ax, 0.03, 0.97, emission_label, transform=ax.transAxes, ha="left", va="top",
              fontsize=6.2)
    return info


def nb_recovery_grid(csv="phase1_recovery_nb_grid.csv"):
    """(Z, FRs, Ns) for NB emission, Fano>=1.5, both generators; None if incomplete/missing.
    Completeness is checked against the CANONICAL FR x N axes (from the Poisson sweep), so a
    partially-computed grid stays a placeholder rather than a degenerate sub-grid.
    `csv` selects the grid file (default = R=20; panel_a passes the R=40 grid)."""
    d = load(csv)
    sweep = load("phase1_recovery_sweep.csv")
    if d is None or sweep is None:
        return None
    FRs = sorted(sweep["fr"].unique()); Ns = sorted(sweep["N"].unique())   # full canonical axes
    over = d[d["fano"] >= 1.5]
    Z = (over.groupby(["fr", "N"])["recovery"].mean()
         .reindex(pd.MultiIndex.from_product([FRs, Ns], names=["fr", "N"]))
         .unstack("N").to_numpy())
    if not np.isfinite(Z).all():
        return None
    return Z, FRs, Ns


# ============================================================ (a) NB FR x N recovery heatmap
def panel_a(ax, fig):
    grid = nb_recovery_grid("phase1_recovery_nb_grid_R40.csv")     # R=40
    if grid is None:
        placeholder(ax, "(a) NB grid not ready\nphase1_recovery_nb_grid_R40.csv\n— rerun nb_grid")
        return None
    Z, FRs, Ns = grid
    return draw_recovery_heatmap(
        ax, fig, Z, FRs, Ns,
        emission_label="NB emission\noverdispersed (Fano≥1.5)\nσ=0.4 · R=40",
        title="recovery map (realistic, NB)")


# ============================================================ (b) Poisson vs NB vs N (R=40)
def panel_b(ax):
    """Same overdispersed data (Fano>=1.5, FR=20), two fitters, R=40 — seed-paired by construction.
    grey  = misspecified Poisson fit  (phase1_recovery_poisson_misspec_R40.csv)
    green = matched NB fit            (phase1_recovery_nb_grid_R40.csv, FR=20 row)"""
    pois = load("phase1_recovery_poisson_misspec_R40.csv")
    nb = load("phase1_recovery_nb_grid_R40.csv")
    if pois is None or nb is None:
        placeholder(ax, "(b) data not found\npoisson_misspec_R40 / nb_grid_R40\n— regenerate"); return None
    Ns = [40, 160, 320]
    po = pois[pois["fano"] >= 1.5]
    nbo = nb[(nb["fr"] == 20) & (nb["fano"] >= 1.5)]
    y_p = [float(po[po.N == n]["recovery"].mean()) for n in Ns]
    y_n = [float(nbo[nbo.N == n]["recovery"].mean()) for n in Ns]
    ax.plot(Ns, y_p, marker="s", color=PALETTE["NEUTRAL"], lw=1.8, ms=6,
            label="Poisson fit (misspecified)", zorder=4)
    ax.plot(Ns, y_n, marker="o", color=PALETTE["ACCENT"], lw=1.8, ms=6,
            label="matched NB fit", zorder=4)
    draw_chance(ax); draw_target(ax)
    ax.set_xticks(Ns); ax.set_xlim(min(Ns) - 25, max(Ns) + 25)
    ax.set_ylim(0.4, 0.9); ax.set_xlabel("pooled trial count  N")
    ax.set_ylabel("recovery rate")
    ax.set_title("overdispersion needs NB emission", fontsize=8.5, loc="left")
    safe_text(ax, 0.97, 0.06, "overdispersed (Fano ≥ 1.5) · R=40", transform=ax.transAxes,
              ha="right", va="bottom", fontsize=6.2, color=PALETTE["NEUTRAL"])
    safe_legend(ax, loc="center right", fontsize=6.6)
    return dict(Ns=Ns, poisson=y_p, nb=y_n)


# ============================================================ (c) recovery vs sigma
def panel_c(ax):
    d = load("ramp_validation_robustness.csv")
    if d is None:
        placeholder(ax, "(c) data not found\nramp_validation_robustness.csv\n— regenerate"); return
    sub = d[(d.fr == 20) & (d.K == 50)]
    sig = sorted(sub["sigma"].unique())
    series = [
        ("both", 320, "N=320 (both)", PALETTE["NEUTRAL"], "-", "o"),
        ("both", 160, "N=160 (both)", "#222222", "-", "s"),
        ("ramp", 160, "N=160 (true ramp)", PALETTE["RAMP"], "--", "^"),
    ]
    for kind, N, lab, col, ls, mk in series:
        s = sub[sub.N == N] if kind == "both" else sub[(sub.N == N) & (sub.true == "ramp")]
        y = [s[s.sigma == sg]["correct"].mean() for sg in sig]
        ax.plot(sig, y, marker=mk, color=col, ls=ls, lw=1.8, ms=5.5, label=lab, zorder=4)
    draw_chance(ax); draw_target(ax)
    ax.axvline(0.4, ls=(0, (1, 2)), lw=0.9, color=PALETTE["ACCENT"], zorder=1)
    safe_text(ax, 0.4, 0.62, "canonical σ", color=PALETTE["ACCENT"], fontsize=6.2,
              ha="center", va="center", rotation=90)
    ax.set_xlabel("ramp diffusion σ"); ax.set_ylabel("recovery rate")
    ax.set_ylim(0.45, 1.0); ax.set_xticks(sig)
    ax.set_title("recovery degrades as ramps get steeper", fontsize=8.5, loc="left")
    safe_legend(ax, loc="lower left", fontsize=6.4)


# ============================================================ (d) intermediacy
def panel_d(ax):
    d = load("phase1_recovery_stepiness.csv")
    if d is None:
        placeholder(ax, "(d) data not found\nphase1_recovery_stepiness.csv\n— regenerate"); return
    st = sorted(d["stepiness"].unique())
    pall = [d[d.stepiness == s]["step_won"].mean() for s in st]
    for N in sorted(d["N"].unique()):
        yn = [d[(d.stepiness == s) & (d.N == N)]["step_won"].mean() for s in st]
        ax.plot(st, yn, color=PALETTE["STEP"], alpha=0.30, lw=0.9, zorder=2)
    ax.plot(st, pall, marker="o", color=PALETTE["STEP"], lw=2.0, ms=5, zorder=4,
            label="P(step model selected)")
    draw_chance(ax)
    near = [s for s, p in zip(st, pall) if abs(p - 0.5) < 0.18]
    if near:
        ax.axvspan(min(near) - 0.03, max(near) + 0.03, color=PALETTE["NEUTRAL"], alpha=0.13, zorder=0)
        safe_text(ax, np.mean([min(near), max(near)]), 0.78, "non-identifiable\n(step ↔ ramp)",
                  ha="center", va="center", fontsize=6.2, color=PALETTE["NEUTRAL"])
    safe_text(ax, 0.02, 0.055, "pure ramp", transform=ax.transAxes, ha="left", fontsize=6.2,
              color=PALETTE["RAMP"], fontweight="bold")
    safe_text(ax, 0.74, 0.96, "pure step", transform=ax.transAxes, ha="left", va="top",
              fontsize=6.2, color=PALETTE["STEP"], fontweight="bold")
    ax.set_xlabel("generator stepiness  (ramp → step)")
    ax.set_ylabel("P(step model selected)")
    ax.set_ylim(0, 1); ax.set_xlim(-0.03, 1.03)
    ax.set_title("intermediate dynamics are ambiguous", fontsize=8.5, loc="left")
    safe_legend(ax, loc="upper left", fontsize=6.4)


# ====================================================== supplementary Fig S1: Poisson map
def render_figS():
    """Supplementary Fig. S1 — the clean-Poisson FR x N recovery map (the lenient upper
    bound). Same axes / 0.80 contour / IBL operating point as the realistic NB map (Fig 2a);
    the house-style header and the bottom note flag it as best-case. The "~half the trial
    count" comparison is read live from both maps' FR=20 0.80-crossings, never hardcoded."""
    d = load("phase1_recovery_sweep.csv")
    if d is None:
        return None
    FRs = sorted(d["fr"].unique()); Ns = sorted(d["N"].unique())
    Z = (d.groupby(["fr", "N"])["correct"].mean()
         .reindex(pd.MultiIndex.from_product([FRs, Ns], names=["fr", "N"]))
         .unstack("N").to_numpy())
    fig = plt.figure(figsize=(6.9, 4.29))           # <=180 mm wide for Extended Data; suptitle at x=0.01
    ax = fig.add_subplot(111)
    info = draw_recovery_heatmap(
        ax, fig, Z, FRs, Ns,
        emission_label="Poisson emission\n(clean · best case)\nσ=0.4",
        title="recovery map — best-case (clean Poisson)")

    # bottom note: best case clears 0.80 at ~half the realistic NB map's trial count.
    # Both 0.80-crossings (this map + Fig 2a) read live from the FR=20 row — never hardcoded.
    note = "lenient best case — real spikes are overdispersed (see Fig 2a, NB)"
    nb = nb_recovery_grid()
    if info.get("target_N") and nb is not None:
        Znb, FRsnb, Nsnb = nb
        if 20.0 in FRsnb:
            iy = FRsnb.index(20.0)
            nb_clears = [n for n in Nsnb if Znb[iy, Nsnb.index(n)] >= 0.80]
            nb_N = min(nb_clears) if nb_clears else None
            if nb_N:
                note = (f"clears 0.80 at N≈{info['target_N']} (20 Hz) — "
                        f"~{info['target_N'] / nb_N:.0%} of the realistic NB map's "
                        f"N≈{nb_N} (Fig 2a).\nlenient best case, not the operating regime")
    safe_text(ax, 0.5, 0.06, note, transform=ax.transAxes, ha="center", va="bottom",
              fontsize=6.6, color=PALETTE["NEUTRAL"])

    fig.suptitle("Figure S1  ·  Poisson best-case recovery "
                 "(does not apply to real, overdispersed spikes)",
                 fontsize=9.5, fontweight="bold", x=0.01, ha="left")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "figS1.png")
    fig.savefig(FIGDIR / "figS1.pdf")
    plt.close(fig)
    return info


# ============================================================ assemble
def main():
    figS_info = render_figS()

    fig = plt.figure(figsize=(7.4, 6.2))
    gs = GridSpec(2, 2, figure=fig)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])

    a_info = panel_a(ax_a, fig)
    panel_b(ax_b); panel_c(ax_c); panel_d(ax_d)
    for ax, lab in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")):
        panel_label(ax, lab)
    fig.suptitle("Figure 2  ·  the step-vs-ramp identifiability map", fontsize=10,
                 fontweight="bold", x=0.01, ha="left")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig2.png"); fig.savefig(FIGDIR / "fig2.pdf")
    plt.close(fig)

    print(f"saved -> {FIGDIR/'fig2.png'}\nsaved -> {FIGDIR/'fig2.pdf'}")
    print(f"saved -> {FIGDIR/'figS1.png'} (+ .pdf)")
    if figS_info:
        print(f"  Poisson(best-case) IBL FR20/N40 recovery = {figS_info['ibl_rec']:.3f}, "
              f"pool target N>= {figS_info['target_N']}")
    if a_info:
        print(f"  NB(realistic)     IBL FR20/N40 recovery = {a_info['ibl_rec']:.3f}, "
              f"pool target N>= {a_info['target_N']}")
    else:
        print("  panel (a): NB grid not complete yet -> placeholder drawn")


if __name__ == "__main__":
    main()

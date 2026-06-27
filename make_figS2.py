"""Figure S2 — diffusion-conditional recovery & intermediacy (the σ-sensitivity detail behind
Fig 2c–d and the §2.2 diffusion band). All values aggregated live from results/ (nothing typed).

  (a) recovery vs σ across the FULL trial-count grid (N=40/160/320) — adds the single-session
      N=40 regime Fig 2c omits; Wilson 95% CI bands     -> ramp_validation_robustness.csv
  (b) step-vs-ramp asymmetry: true-ramp recovery collapses toward chance as σ grows while true-step
      stays recovered (the ramp-bias)                   -> ramp_validation_robustness.csv
  (c) the σ-band is firing-rate dependent (fr=20 vs 40) but K-invariant (K=25/50/75 coincide at σ=0.4)
                                                        -> ramp_validation_robustness.csv
  (d) fuller P(step)-vs-stepiness: per-N curves (160 vs 320) with CI bands + the non-identifiable
      crossover region                                  -> phase1_recovery_stepiness.csv

Run:  python make_figS2.py   -> figures/figS2.png (300 dpi) + figures/figS2.pdf (vector)
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

import figstyle  # noqa: E402  (installs rcParams incl. constrained_layout + legend frame)
from figstyle import PALETTE, SEQ_CMAP, panel_label, safe_text, safe_legend  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures"
CMAP = matplotlib.colormaps[SEQ_CMAP]
SIGS = [0.2, 0.4, 0.7, 1.0]


def load(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def wilson(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion (per-seed spread, deterministic)."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def curve(rob, N, fr, K, true=None):
    """(sigma, p, lo, hi, n) per σ for a (N,fr,K[,true]) slice; both generators if true is None."""
    sub = rob[(rob.N == N) & (rob.fr == fr) & (rob.K == K)]
    if true is not None:
        sub = sub[sub.true == true]
    out = []
    for s in sorted(sub.sigma.unique()):
        d = sub[sub.sigma == s]["correct"]
        k, n = int(d.sum()), len(d)
        lo, hi = wilson(k, n)
        out.append((float(s), k / n, lo, hi, n))
    return out


# ---------------------------------------------------------------- shared reference styling
def draw_target(ax):
    ax.axhline(0.80, ls="--", lw=1.0, color=PALETTE["NEUTRAL"], zorder=1)


def draw_chance(ax):
    ax.axhline(0.50, ls=":", lw=1.0, color=PALETTE["NEUTRAL"], zorder=1)


def canonical_sigma(ax, label=False):
    ax.axvline(0.4, ls=(0, (1, 2)), lw=0.9, color=PALETTE["ACCENT"], zorder=1)
    if label:
        safe_text(ax, 0.4, 0.62, "canonical σ", color=PALETTE["ACCENT"], fontsize=6.0,
                  ha="center", va="center", rotation=90)


def sigma_axes(ax):
    ax.set_xticks(SIGS); ax.set_xlim(0.13, 1.07)
    ax.set_xlabel("ramp diffusion σ"); ax.set_ylabel("recovery rate")


# ============================================================ (a) full N grid recovery vs σ
def panel_a(ax):
    rob = load("ramp_validation_robustness.csv")
    if rob is None:
        ax.text(0.5, 0.5, "(a) data not found", ha="center"); return None
    cols = {40: CMAP(0.08), 160: CMAP(0.45), 320: CMAP(0.78)}
    mks = {40: "o", 160: "s", 320: "^"}
    plotted = {}
    for N in (40, 160, 320):
        c = curve(rob, N, 20, 50)
        if not c:
            continue
        s = [r[0] for r in c]; p = [r[1] for r in c]; lo = [r[2] for r in c]; hi = [r[3] for r in c]
        ax.fill_between(s, lo, hi, color=cols[N], alpha=0.16, zorder=2)
        ax.plot(s, p, marker=mks[N], color=cols[N], lw=1.8, ms=5.5, label=f"N={N}", zorder=4)
        plotted[N] = p
    draw_target(ax); draw_chance(ax); canonical_sigma(ax, label=True)
    sigma_axes(ax); ax.set_ylim(0.4, 1.02)
    ax.set_title("recovery vs σ across the trial-count grid", fontsize=8.5, loc="left")
    safe_text(ax, 0.03, 0.045, "fr=20 Hz · K=50 · step+ramp\nbands = Wilson 95% CI",
              transform=ax.transAxes, ha="left", va="bottom", fontsize=6.0, color=PALETTE["NEUTRAL"])
    safe_legend(ax, loc="upper right", fontsize=6.6)
    return plotted


# ============================================================ (b) step vs ramp asymmetry
def panel_b(ax):
    rob = load("ramp_validation_robustness.csv")
    if rob is None:
        ax.text(0.5, 0.5, "(b) data not found", ha="center"); return None
    out = {}
    for N, ls, mk in ((320, "-", "o"), (160, (0, (4, 2)), "s")):
        for tg, col in (("step", PALETTE["STEP"]), ("ramp", PALETTE["RAMP"])):
            c = curve(rob, N, 20, 50, true=tg)
            if not c:
                continue
            s = [r[0] for r in c]; p = [r[1] for r in c]
            ax.plot(s, p, marker=mk, color=col, ls=ls, lw=1.7, ms=4.8,
                    label=f"true {tg}, N={N}", zorder=4)
            out[(tg, N)] = p
    draw_target(ax); draw_chance(ax); canonical_sigma(ax)
    sigma_axes(ax); ax.set_ylim(0.3, 1.04)
    ax.set_title("ramps degrade with σ; steps stay recovered", fontsize=8.5, loc="left")
    safe_text(ax, 0.97, 0.05, "fr=20 Hz · K=50", transform=ax.transAxes, ha="right", va="bottom",
              fontsize=6.0, color=PALETTE["NEUTRAL"])
    safe_legend(ax, loc="lower left", fontsize=5.9)
    return out


# ============================================================ (c) FR sensitivity + K invariance
def panel_c(ax):
    rob = load("ramp_validation_robustness.csv")
    if rob is None:
        ax.text(0.5, 0.5, "(c) data not found", ha="center"); return None
    series = [(20, 160, "-", "o", CMAP(0.18), "fr=20, N=160"),
              (40, 160, "-", "s", CMAP(0.72), "fr=40, N=160"),
              (40, 80, (0, (2, 2)), "^", CMAP(0.72), "fr=40, N=80")]
    for fr, N, ls, mk, col, lab in series:
        c = curve(rob, N, fr, 50)
        if not c:
            continue
        s = [r[0] for r in c]; p = [r[1] for r in c]
        ax.plot(s, p, marker=mk, ls=ls, color=col, lw=1.7, ms=5, label=lab, zorder=4)
    # K-invariance: K=25/50/75 at σ=0.4 (N=160, fr=20) — markers should coincide
    kvals = []
    for K in (25, 50, 75):
        d = rob[(rob.N == 160) & (rob.fr == 20) & (rob.K == K) & (rob.sigma == 0.4)]["correct"]
        if len(d):
            kvals.append(d.mean())
            ax.plot(0.4, d.mean(), marker="D", ms=6, mfc="none", mec=PALETTE["ACCENT"],
                    mew=1.5, zorder=6)
    draw_target(ax); draw_chance(ax); canonical_sigma(ax)
    sigma_axes(ax); ax.set_ylim(0.4, 1.02)
    ax.set_title("σ-band shifts with rate, invariant to grid K", fontsize=8.5, loc="left")
    if kvals:
        safe_text(ax, 0.42, min(kvals) - 0.10, "K=25/50/75 agree\n(σ=0.4, within MC)",
                  ha="left", va="top", fontsize=5.9, color=PALETTE["ACCENT"])
    safe_legend(ax, loc="lower left", fontsize=6.0)
    return kvals


# ============================================================ (d) P(step) vs stepiness, per N
def panel_d(ax):
    st = load("phase1_recovery_stepiness.csv")
    if st is None:
        ax.text(0.5, 0.5, "(d) data not found", ha="center"); return None
    cols = {160: PALETTE["STEP"], 320: "#08306b"}
    mks = {160: "o", 320: "s"}; lss = {160: "-", 320: (0, (4, 2))}
    xs_all = sorted(st.stepiness.unique())
    pooled = []
    out = {}
    for N in (160, 320):
        sub = st[st.N == N]
        xs = sorted(sub.stepiness.unique()); p = []; lo = []; hi = []
        for x in xs:
            d = sub[sub.stepiness == x]["step_won"]; k, n = int(d.sum()), len(d)
            p.append(k / n); a, b = wilson(k, n); lo.append(a); hi.append(b)
        ax.fill_between(xs, lo, hi, color=cols[N], alpha=0.13, zorder=2)
        ax.plot(xs, p, marker=mks[N], ls=lss[N], color=cols[N], lw=1.8, ms=5,
                label=f"N={N}", zorder=4)
        out[N] = p
    # non-identifiable crossover: stepiness where pooled P(step) is within 0.18 of chance
    for x in xs_all:
        d = st[st.stepiness == x]["step_won"]
        pooled.append(d.mean())
    near = [x for x, pv in zip(xs_all, pooled) if abs(pv - 0.5) < 0.18]
    if near:
        ax.axvspan(min(near) - 0.04, max(near) + 0.04, color=PALETTE["NEUTRAL"], alpha=0.13, zorder=0)
        safe_text(ax, np.mean([min(near), max(near)]), 0.30, "non-identifiable\n(step ↔ ramp)",
                  ha="center", va="center", fontsize=6.0, color=PALETTE["NEUTRAL"])
    draw_chance(ax)
    safe_text(ax, 0.02, 0.055, "pure ramp", transform=ax.transAxes, ha="left", fontsize=6.2,
              color=PALETTE["RAMP"], fontweight="bold")
    safe_text(ax, 0.74, 0.965, "pure step", transform=ax.transAxes, ha="left", va="top",
              fontsize=6.2, color=PALETTE["STEP"], fontweight="bold")
    ax.set_xlabel("generator stepiness  (ramp → step)")
    ax.set_ylabel("P(step model selected)")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 1)
    ax.set_title("intermediate generators are ambiguous", fontsize=8.5, loc="left")
    safe_legend(ax, loc="upper left", fontsize=6.4)
    return out


# ============================================================ assemble
def main():
    fig = plt.figure(figsize=(6.9, 5.54))           # <=180 mm wide for Extended Data
    gs = GridSpec(2, 2, figure=fig)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])

    a = panel_a(ax_a); b = panel_b(ax_b); c = panel_c(ax_c); d = panel_d(ax_d)
    for ax, lab in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")):
        panel_label(ax, lab)
    fig.suptitle("Figure S2  ·  diffusion-conditional recovery & intermediacy "
                 "(σ-sensitivity behind Fig 2c–d)",
                 fontsize=9.5, fontweight="bold", x=0.01, ha="left")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "figS2.png"); fig.savefig(FIGDIR / "figS2.pdf")
    plt.close(fig)
    print(f"saved -> {FIGDIR/'figS2.png'} (300 dpi)\nsaved -> {FIGDIR/'figS2.pdf'} (vector)")
    if a:
        print("  (a) recovery vs sigma (fr20,K50,both):",
              {N: [round(v, 3) for v in p] for N, p in a.items()}, "for sigma", SIGS)
    if b:
        print("  (b) step/ramp split:", {f"{t}N{n}": [round(v, 3) for v in p] for (t, n), p in b.items()})
    if c:
        print("  (c) K=25/50/75 @sigma0.4 (N160,fr20):", [round(v, 3) for v in c])
    if d:
        print("  (d) P(step) vs stepiness:", {N: [round(v, 3) for v in p] for N, p in d.items()})


if __name__ == "__main__":
    main()

"""Figure 1 — "the ambiguity, the models, the validation".

(a) THE AMBIGUITY  : ~10 latent single-trial trajectories from the paper's OWN stepping
                     and ramping generators (phase1_recovery.simulate's dynamics), tuned so
                     the two trial-AVERAGES coincide -> same mean ramp, different single trials.
(b) THE MODELS     : minimal schematics — stepping = 2-state baseline->elevated jump;
                     ramping = drift-diffusion on a latent grid to an absorbing bound.
(c) VALIDATION     : the three forward-likelihood |differences| (read from the results JSONs)
                     on a log axis, against a machine-precision band and a 1e-6 reference.

Run:  python make_fig1.py    -> figures/fig1.png (300 dpi) + figures/fig1.pdf (vector)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))     # so we can import the paper's generators

import figstyle  # noqa: E402  (installs rcParams on import)
from figstyle import PALETTE, panel_label, safe_text, safe_legend  # noqa: E402
# the paper's ACTUAL generator parameters + dynamics (NOT a new cartoon generator)
from phase1_recovery import gen_rates, GEN_DRIFT, GEN_SIGMA, DT  # noqa: E402

RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures"

# representative operating point for panel (a)
FR_REP = 20.0          # Hz (paper-typical hi-FR cell; r_lo=13.3, r_hi=26.7)
T_BINS = 41            # 41 x 25 ms = 1.000 s window  (matches GEN_DRIFT reaching bound ~1 s)
N_MEAN = 4000          # trials for the smooth family means
N_SHOW = 10            # faint single-trial traces per family
SEED = 7


# ----------------------------------------------------------------------------- (a) latents
# The ramping single trials are the repo's OWN drift-diffusion generator (gen_rates, GEN_DRIFT,
# GEN_SIGMA, DT, reflect-at-0 / absorb-at-1) -- left untouched. The two families are then made to
# share ONE trial-average BY CONSTRUCTION, honestly: Monte-Carlo the ramp's mean normalised latent
# m(t) in [0,1], and draw each STEP trial's jump time by inverse-CDF sampling with CDF = m(t).
# Because u ~ U(0,1), P(jump <= t) = P(u <= m(t)) = m(t), so E[step latent](t) =
# r_lo + (r_hi-r_lo)*m(t) = the ramp mean exactly -> the dashed average passes through both.
def ramp_latents(rng, T, fr, drift, N):
    r_lo, r_hi = gen_rates(fr)
    x = np.zeros(N)
    rates = np.empty((N, T))
    for t in range(T):
        rates[:, t] = r_lo + (r_hi - r_lo) * x
        x = x + drift * DT + GEN_SIGMA * np.sqrt(DT) * rng.standard_normal(N)
        x = np.clip(x, 0.0, 1.0)               # reflect at 0, absorb at 1
    return rates


def step_from_cdf(u, m, fr):
    """Inverse-CDF stepping latents: each trial is flat at r_lo, then jumps ONCE to r_hi at the
    first bin where the shared ramp-mean CDF m(t) reaches u ~ U(0,1) (trials with u > m[-1] never
    jump). Vectorised over u. Guarantees E[latent](t) = r_lo + (r_hi-r_lo)*m(t)."""
    r_lo, r_hi = gen_rates(fr)
    T = len(m)
    u = np.atleast_1d(np.asarray(u, float))
    jump = np.searchsorted(m, u, side="left")            # first t with m[t] >= u (== T -> no jump)
    idx = np.arange(T)[None, :]
    return np.where(idx >= jump[:, None], r_hi, r_lo).astype(float)


def panel_a(ax_top, ax_bot):
    drift = GEN_DRIFT                                     # repo's own drift-diffusion parameter
    r_lo, r_hi = gen_rates(FR_REP)
    t_ms = np.arange(T_BINS) * DT * 1000.0
    n_avg = 20000                                        # MC trials for the shared mean / CDF

    # ramp family mean -> shared monotone CDF m(t) in [0,1] -> common trial-average (rate)
    ramp_mean = ramp_latents(np.random.default_rng(SEED + 3), T_BINS, FR_REP, drift, n_avg).mean(0)
    m = np.maximum.accumulate(np.clip((ramp_mean - r_lo) / (r_hi - r_lo), 0.0, 1.0))
    common = r_lo + (r_hi - r_lo) * m                    # the ONE average both families produce

    # stepping family mean (large-N inverse-CDF) for the coincidence check
    step_mean = step_from_cdf(np.random.default_rng(SEED + 2).uniform(size=n_avg), m, FR_REP).mean(0)

    # ~10 faint single trials per family (ramp = repo generator; step = inverse-CDF of m)
    ramp_show = ramp_latents(np.random.default_rng(SEED + 5), T_BINS, FR_REP, drift, N_SHOW)
    step_show = step_from_cdf(np.random.default_rng(SEED + 4).uniform(size=N_SHOW), m, FR_REP)

    mean_handles, dash_handle = {}, None
    for ax, traces, mean, key, name, stepwise in (
        (ax_top, step_show, step_mean, "STEP", "stepping", True),
        (ax_bot, ramp_show, ramp_mean, "RAMP", "ramping", False),
    ):
        for tr in traces:                                # faint single trials
            if stepwise:                                 # flat -> sudden JUMP -> flat
                ax.step(t_ms, tr, where="post", color=PALETTE[key], alpha=0.25, lw=0.9)
            else:                                        # gradual noisy climb
                ax.plot(t_ms, tr, color=PALETTE[key], alpha=0.25, lw=0.9)
        mean_handles[key] = ax.plot(t_ms, mean, color=PALETTE[key], lw=2.4, zorder=5)[0]
        dash_handle = ax.plot(t_ms, common, color="black", lw=1.0, ls="--", zorder=6)[0]
        ax.set_ylabel("latent FR (Hz)")
        ax.text(0.97, 0.10, name, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, fontweight="bold", color=PALETTE[key])

    ax_top.tick_params(labelbottom=False)
    ax_bot.set_xlabel("time in window (ms)")
    ax_top.set_xlim(0, t_ms[-1])
    # legend makes explicit that the dashed line is the ONE average both families produce
    mean_handles["STEP"].set_label("stepping mean")
    mean_handles["RAMP"].set_label("ramping mean")
    dash_handle.set_label("common trial-average (both families)")
    safe_legend(ax_top, handles=[mean_handles["STEP"], mean_handles["RAMP"], dash_handle],
                loc="lower right", fontsize=6.0, handlelength=1.5,
                borderpad=0.2, labelspacing=0.25)
    coincide = float(np.max(np.abs(step_mean - ramp_mean)))
    return drift, coincide


# ----------------------------------------------------------------------------- (b) models
def _node(ax, xy, r, key, fill=False, lw=1.4):
    fc = PALETTE[key] if fill else "white"
    ax.add_patch(Circle(xy, r, facecolor=fc, edgecolor=PALETTE[key], lw=lw, zorder=3))


def _arrow(ax, p0, p1, key, rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                 connectionstyle=f"arc3,rad={rad}",
                                 color=PALETTE[key], lw=lw, zorder=2))


def panel_b(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # --- stepping: 2-state left->right jump (upper row) ---
    yS = 0.74
    ax.text(0.02, yS + 0.17, "stepping", fontsize=8.5, fontweight="bold", color=PALETTE["STEP"])
    _node(ax, (0.22, yS), 0.075, "STEP", fill=False)
    _node(ax, (0.60, yS), 0.075, "STEP", fill=True)
    _arrow(ax, (0.305, yS), (0.515, yS), "STEP")
    ax.text(0.22, yS - 0.16, "baseline", ha="center", fontsize=7, color=PALETTE["NEUTRAL"])
    ax.text(0.60, yS - 0.16, "elevated", ha="center", fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.1", fc=PALETTE["STEP"], ec="none"))
    ax.text(0.41, yS + 0.085, "single jump", ha="center", fontsize=6.5,
            color=PALETTE["STEP"], style="italic")

    # --- ramping: drift-diffusion on a latent grid to an absorbing bound (lower row) ---
    yR = 0.26
    ax.text(0.02, yR + 0.17, "ramping", fontsize=8.5, fontweight="bold", color=PALETTE["RAMP"])
    xs = [0.18, 0.32, 0.46, 0.60]
    for i, x in enumerate(xs):
        _node(ax, (x, yR), 0.045, "RAMP", fill=False, lw=1.2)
        _arrow(ax, (x, yR + 0.05), (x - 0.018, yR + 0.05), "RAMP", rad=1.7, lw=1.0)  # self
        if i < len(xs) - 1:
            _arrow(ax, (x + 0.05, yR), (xs[i + 1] - 0.05, yR), "RAMP")               # advance
    _arrow(ax, (xs[-1] + 0.05, yR), (0.715, yR), "RAMP")
    _node(ax, (0.775, yR), 0.058, "RAMP", fill=True)                               # absorbing bound
    ax.add_patch(Circle((0.775, yR), 0.040, facecolor="white", edgecolor=PALETTE["RAMP"],
                        lw=1.0, zorder=4))                                          # double ring
    ax.text(0.775, yR - 0.17, "bound", ha="center", fontsize=7, color=PALETTE["RAMP"])
    ax.text(0.40, yR + 0.135, "drift + diffusion", ha="center", fontsize=6.5,
            color=PALETTE["RAMP"], style="italic")
    ax.text(0.40, yR - 0.165, "gradual latent grid", ha="center", fontsize=7,
            color=PALETTE["NEUTRAL"])


# ----------------------------------------------------------------------------- (c) validation
def read_validation():
    fwd = json.loads((RESULTS / "ramp_validation_forward.json").read_text())
    rec = json.loads((RESULTS / "phase1_recovery_validation.json").read_text())
    return [
        ("ramp vs brute-force", float(fwd["ramp_bruteforce_max_abs_diff"]), "RAMP"),
        ("step vs hmmlearn", float(rec["hmmlearn_check"]["abs_diff"]), "STEP"),
        ("ramp vs hmmlearn", float(fwd["ramp_hmmlearn_k50_abs_diff"]), "RAMP"),
    ]


def panel_c(ax, points):
    xleft, xref = 1e-16, 1e-6
    ax.set_xscale("log")
    ax.set_xlim(xleft, 3e-5)
    ax.set_ylim(-0.6, len(points) - 0.4)

    # machine-precision band behind the points
    ax.axvspan(1e-16, 1e-11, color=PALETTE["NEUTRAL"], alpha=0.13, lw=0)
    ax.text(10 ** -13.5, len(points) - 0.5, "machine precision", ha="center", va="top",
            fontsize=6.6, color=PALETTE["NEUTRAL"], style="italic")
    # inference-relevant reference line far to the right
    ax.axvline(xref, color=PALETTE["ACCENT"], ls="--", lw=1.2)
    ax.text(xref * 1.6, len(points) / 2 - 0.5, "inference-relevant\nscale (1e-6)",
            rotation=90, va="center", ha="left", fontsize=6.8, color=PALETTE["ACCENT"])

    for i, (label, val, key) in enumerate(points):
        ax.hlines(i, xleft, val, color=PALETTE[key], lw=1.4, alpha=0.8)
        ax.plot(val, i, "o", color=PALETTE[key], ms=7, zorder=5)
        safe_text(ax, val * 1.6, i + 0.17, f"{val:.1e}", color=PALETTE[key],
                  fontsize=6.8, fontweight="bold", va="bottom", ha="left")

    ax.set_yticks(range(len(points)))
    ax.set_yticklabels([p[0] for p in points])
    ax.set_xlabel("|Δ forward log-likelihood|  (absolute)")
    ax.set_title("inference-engine validation", fontsize=8.5, loc="left")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


# ----------------------------------------------------------------------------- assemble
def main():
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = GridSpec(2, 2, width_ratios=[1.12, 0.95], height_ratios=[1.0, 0.92],
                  figure=fig, wspace=0.32, hspace=0.42)

    gsa = gs[0, 0].subgridspec(2, 1, hspace=0.12)
    ax_a_top = fig.add_subplot(gsa[0])
    ax_a_bot = fig.add_subplot(gsa[1], sharex=ax_a_top, sharey=ax_a_top)
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])

    drift, coincide = panel_a(ax_a_top, ax_a_bot)
    panel_b(ax_b)
    points = read_validation()
    panel_c(ax_c, points)

    panel_label(ax_a_top, "a")
    panel_label(ax_b, "b")
    panel_label(ax_c, "c", dx=-0.20)

    ax_a_top.set_title("same average, different single trials", fontsize=8.5, loc="left")

    fig.suptitle("Figure 1  ·  the ambiguity, the models, and engine validation", fontsize=10,
                 fontweight="bold", x=0.01, ha="left")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    png, pdf = FIGDIR / "fig1.png", FIGDIR / "fig1.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)

    print(f"tuned ramp drift (to match averages): {drift:.3f}  (paper default GEN_DRIFT={GEN_DRIFT})")
    print(f"trial-average coincidence  max|step_mean - ramp_mean| = {coincide:.4f} Hz "
          f"(envelope span {gen_rates(FR_REP)[1]-gen_rates(FR_REP)[0]:.2f} Hz)")
    print("validation numbers plotted (from JSONs):")
    for label, val, _ in points:
        print(f"   {label:22s} = {val:.4e}")
    print(f"saved -> {png}")
    print(f"saved -> {pdf}")


if __name__ == "__main__":
    main()

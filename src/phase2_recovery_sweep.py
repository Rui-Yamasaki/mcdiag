"""Phase 1 step 2, PART B — recovery sweep at IBL-realistic parameters.

Can we recover the true generator (stepping vs ramping) given:
  - per-trial decision-window lengths DRAWN from our measured 500-2000 ms
    deliberative distribution (curated slow, low-contrast reaction times),
  - plausible mean firing rates {2,5,10,20,40} Hz,
  - the THIN per-session trial counts {20,40,80,160,320}?

Per (true-generator x N x FR) cell: repeat R times { draw N window lengths from
the measured distribution; simulate from the true generator; fit BOTH models;
pick the winner by the SAME k-fold CV held-out log-likelihood }. Recovery rate =
fraction of repeats whose CV verdict equals the true generator.

Mean firing rate is the swept variable; modulation depth is fixed (baseline
0.5*FR -> committed 1.5*FR, so the window-average rate ~= FR).

Outputs: figures/phase1_recovery_sim.png + results/phase2_recovery.csv.

Run (after src/phase2_validate.py passes):  python src/phase2_recovery_sweep.py
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import stepramp as sr  # noqa: E402
from ibl_one import DATA_DIR, PROJECT_ROOT  # noqa: E402

POOLED = DATA_DIR / "phase1_trials_pooled.parquet"
FIG = PROJECT_ROOT / "figures" / "phase1_recovery_sim.png"
CSV = PROJECT_ROOT / "results" / "phase2_recovery.csv"

N_GRID = [20, 40, 80, 160, 320]
FR_GRID = [2, 5, 10, 20, 40]
LOW_CONTRASTS = [0.0, 6.25, 12.5]
WIN_RANGE_S = (0.5, 2.0)          # deliberative window range
M_GRID = 25                       # latent grid (validated by convergence check)
K_FOLDS = 4


def load_window_bins():
    """Empirical per-trial window lengths (bins) from curated slow low-contrast RTs."""
    from phase1b_engagement import add_columns, restrict_to_trained_ephys
    df = add_columns(pd.read_parquet(POOLED))
    df = restrict_to_trained_ephys(df)
    rt = df["reaction_time"].to_numpy(float)
    ac = df["abs_contrast_pct"].to_numpy(float)
    keep = (np.isfinite(rt) & (rt >= WIN_RANGE_S[0]) & (rt <= WIN_RANGE_S[1])
            & np.isin(ac, LOW_CONTRASTS))
    win_s = rt[keep]
    bins = np.maximum(np.round(win_s / sr.DT).astype(int), 2)
    print(f"  measured window pool: {bins.size} trials, "
          f"median {np.median(win_s)*1000:.0f} ms "
          f"(IQR {np.percentile(win_s,25)*1000:.0f}-"
          f"{np.percentile(win_s,75)*1000:.0f} ms), "
          f"-> {np.median(bins):.0f} bins median")
    return bins


def make_params(true, fr, median_T, rng):
    lam0 = 0.5 * fr * rng.uniform(0.9, 1.1)
    lam1 = 1.5 * fr * rng.uniform(0.9, 1.1)
    if true == "step":
        p = (1.6 / median_T) * rng.uniform(0.8, 1.2)
        return (lam0, lam1, float(np.clip(p, 1e-3, 0.5)))
    cross = (median_T / 1.6) * rng.uniform(0.8, 1.2)   # bins to reach the bound
    beta = 1.0 / (cross * sr.DT)
    sigma = 0.7 * rng.uniform(0.8, 1.2)
    return (lam0, lam1, beta, sigma)


def run_cell(true, N, fr, win_bins, median_T, R, n_starts, rng):
    hits = 0
    for r in range(R):
        lengths = rng.choice(win_bins, size=N)
        params = make_params(true, fr, median_T, rng)
        counts, L = sr.simulate(true, params, lengths, sr.DT, rng)
        winner, _, _ = sr.cv_compare(counts, L, sr.DT, M=M_GRID, k=K_FOLDS,
                                     n_starts=n_starts,
                                     rng=np.random.default_rng(rng.integers(1 << 30)))
        hits += (winner == true)
    return hits


def make_figure(table):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, true in zip(axes, ("step", "ramp")):
        grid = np.full((len(FR_GRID), len(N_GRID)), np.nan)
        for _, row in table[table["true"] == true].iterrows():
            i = FR_GRID.index(int(row["FR"]))
            j = N_GRID.index(int(row["N"]))
            grid[i, j] = row["recovery_rate"]
        im = ax.imshow(grid, origin="lower", cmap="RdYlGn", vmin=0.4, vmax=1.0,
                       aspect="auto")
        for i in range(len(FR_GRID)):
            for j in range(len(N_GRID)):
                ax.text(j, i, f"{grid[i, j]*100:.0f}", ha="center", va="center",
                        fontsize=9,
                        color="black" if 0.5 < grid[i, j] < 0.85 else "white")
        ax.set_xticks(range(len(N_GRID)))
        ax.set_xticklabels(N_GRID)
        ax.set_yticks(range(len(FR_GRID)))
        ax.set_yticklabels(FR_GRID)
        ax.set_xlabel("trials per neuron (N)")
        ax.set_ylabel("mean firing rate (Hz)")
        ax.set_title(f"true = {true.upper()}  (recovery %)", fontsize=11, loc="left")
        # IBL-plausible per-session budget: N ~ 40 low-contrast slow trials/neuron
        ax.axvline(N_GRID.index(40), color="navy", lw=2.5, alpha=0.7)
        ax.text(N_GRID.index(40), len(FR_GRID) - 0.4, " IBL per-session ~40",
                color="navy", fontsize=8, va="top")
        fig.colorbar(im, ax=ax, shrink=0.85, label="recovery rate")
    fig.suptitle("Phase 1 step-vs-ramp recovery at IBL-realistic windows "
                 "(per-trial 500-2000 ms; Poisson; CV log-lik)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"Saved figure -> {FIG}")


def glanceable(table):
    def rate(true, N, fr):
        m = table[(table["true"] == true) & (table["N"] == N) & (table["FR"] == fr)]
        return float(m["recovery_rate"].iloc[0]) if len(m) else np.nan
    # average over the two truths at N=40
    print("\n========== GLANCEABLE VERDICT ==========")
    for fr in FR_GRID:
        avg40 = np.nanmean([rate("step", 40, fr), rate("ramp", 40, fr)])
        print(f"  N=40 trials, FR={fr:>2} Hz : mean recovery {avg40*100:.0f}%")
    # find min N,FR for >=80% (averaged over truths)
    best = []
    for fr in FR_GRID:
        for N in N_GRID:
            avg = np.nanmean([rate("step", N, fr), rate("ramp", N, fr)])
            if avg >= 0.80:
                best.append((N, fr, avg))
    if best:
        minN = min(b[0] for b in best)
        frs_at_minN = [b for b in best if b[0] == minN]
        bestfr = min(b[1] for b in frs_at_minN)
        print(f"  >=80% recovery first reached at N>={minN}, FR>={bestfr} Hz")
    else:
        print("  >=80% recovery NOT reached anywhere in the swept grid")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--starts", type=int, default=2)
    args = ap.parse_args()

    print("Loading measured deliberative-window distribution...")
    win_bins = load_window_bins()
    median_T = float(np.median(win_bins))

    rng = np.random.default_rng(2024)
    rows = []
    cells = [(t, N, fr) for t in ("step", "ramp") for N in N_GRID for fr in FR_GRID]
    t0 = time.time()
    CSV.parent.mkdir(exist_ok=True)
    for c, (true, N, fr) in enumerate(cells, 1):
        hits = run_cell(true, N, fr, win_bins, median_T, args.reps, args.starts, rng)
        rec = hits / args.reps
        rows.append({"true": true, "N": N, "FR": fr, "reps": args.reps,
                     "hits": hits, "recovery_rate": rec})
        pd.DataFrame(rows).to_csv(CSV, index=False)   # incremental save
        el = time.time() - t0
        print(f"  [{c:>2}/{len(cells)}] true={true} N={N:>3} FR={fr:>2}Hz "
              f"-> {rec*100:3.0f}%  ({el:.0f}s elapsed)")
    table = pd.DataFrame(rows)
    print(f"\nSaved CSV -> {CSV}")
    make_figure(table)
    glanceable(table)


if __name__ == "__main__":
    main()

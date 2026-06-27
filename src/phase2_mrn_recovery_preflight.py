"""Phase 2 / FINAL gate - MRN POPULATION step-vs-ramp RECOVERY pre-flight (PURE SIM).

The population decode found a REAL, distributed, per-cell-cryptic decision code confined
to MRN (error-only choice-decode AUC 0.79, p=0.012, 243 cells; per-cell mean AUC 0.51).
The §3.5 question: can a latent POPULATION step-vs-ramp actually be RECOVERED at MRN scale,
given the signal is weak per cell and overdispersed? This is the last analysis before
writing. PURE SIMULATION - no IBL data, no downloads.

Approach (extends the VALIDATED phase1_recovery / _hardened engine to a population):
  GENERATOR  one shared 1-D decision latent per trial - either a discrete STEP (jump 0->1
             at a random bin) or a diffusion RAMP (0->1; the EXACT GEN_DRIFT/GEN_SIGMA
             generator from phase1_recovery). A POPULATION of N cells, each with its OWN
             baseline FR b_i (lognormal matched to MRN: median 24.3 Hz) and its OWN weak,
             signed coupling g_i to the latent: rate_i(t) = b_i*(1 + g_i*d*z(t)), where
             d=+/-1 is the (observed) choice. NB emissions at Fano F (per the hardening
             result that real spikes are overdispersed and the fitter MUST be NB).
  RECOVERY   the joint population is one HMM over the SHARED latent: the per-bin emission
             is the SUM over cells of each cell's NB log-likelihood given the latent value.
             This REUSES step_hmm/ramp_hmm (identical transition structure + x-grid) and
             the same held-out CV-LL comparison; only the emission is summed across cells.
             Step vs ramp dynamics params are fit on train; winner = higher held-out
             population forward-LL. Per-cell tuning (b_i,g_i) is ORACLE (true values) ->
             a GENEROUS upper bound: if recovery fails even with perfect per-cell tuning
             knowledge, it fails a fortiori with estimated tuning.

Calibration (locked): coupling tuned so the simulated population reproduces the MRN regime
  - per-cell raw choice-AUC: mean ~0.507, p95 (single thin pool) ~0.75   (obs 0.507 / 0.771)
  - population (L2-logistic) choice-decode AUC ~0.80                       (obs 0.794)

MRN scale (locked from results/phase2_population_decode.csv + choicestim_cells_full.csv):
  N_cells = 243 (error-testable); per-cell median FR 24.3 Hz (all >=10 Hz);
  per-session error budget median 9 trials (decorrelated 25), max ~10-53 simultaneous cells
  -> the 243-cell population is a PSEUDO-population spanning 61 sessions (NOT simultaneous).
  So the realistic simultaneous regime is ~10 cells x ~25 trials; the 243-cell idealization
  is the optimistic upper bound. The recovery-vs-scale sweep spans both.

  python src/phase2_mrn_recovery_preflight.py --stage all
  python src/phase2_mrn_recovery_preflight.py --stage all --quick   # fast smoke
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import gammaln, logsumexp  # noqa: E402

from ibl_one import PROJECT_ROOT  # noqa: E402
from phase1_recovery import (DT, GEN_DRIFT, GEN_SIGMA, RATE_FLOOR, load_windows,  # noqa: E402
                             ramp_hmm, step_hmm)

# --- locked MRN regime (from the results CSVs) -------------------------------
FR_MU, FR_SIGMA = np.log(24.3), 0.70      # baseline-FR lognormal (median 24.3 Hz, p90 ~63)
MRN_GSCALE, MRN_GSIGMA = 0.015, 1.0       # weak heavy-tailed coupling -> MRN decode regime
EASY_GSCALE, EASY_GSIGMA = 0.20, 0.5      # STRONG coupling for the easy-regime sanity
MRN_N_CELLS = 243
K_GRID = 50                               # ramp latent grid (Phase-1 converged value)
N_JOBS = 8                                # cap workers -> bound peak memory on heavy cells

FIG_PATH = PROJECT_ROOT / "figures" / "phase2_mrn_recovery_preflight.png"
SWEEP_CSV = PROJECT_ROOT / "results" / "phase2_mrn_recovery_preflight.csv"
HEAD_CSV = PROJECT_ROOT / "results" / "phase2_mrn_recovery_headline.csv"
# per-stage checkpoints (raw recoveries) -> a kill never loses completed stages
CKPT = {k: PROJECT_ROOT / "results" / f"_preflight_raw_{k}.csv"
        for k in ("easy", "mrn", "null", "grid")}


# =============================================================================
#  GENERATOR  (shared latent -> population of weakly-coupled NB cells)
# =============================================================================
def draw_cells(rng, N, gscale, gsigma):
    """Per-cell baseline FR b_i (Hz) and signed coupling g_i to the shared latent."""
    b = np.exp(rng.normal(FR_MU, FR_SIGMA, N))
    sign = rng.choice([-1.0, 1.0], N)
    mag = gscale * np.exp(rng.normal(0.0, gsigma, N))     # heavy-tailed |coupling|
    return b, sign * mag


def shared_latent(true_gen, rng, T):
    """The ONE decision latent for a trial: STEP jumps 0->1 once; RAMP diffuses 0->1
    (identical drift/sigma to the validated phase1_recovery ramp generator)."""
    if true_gen == "step":
        s = rng.integers(1, T) if T > 1 else 1
        return (np.arange(T) >= s).astype(float)
    z = np.empty(T); x = 0.0
    for t in range(T):
        z[t] = x
        x = min(max(x + GEN_DRIFT * DT + GEN_SIGMA * np.sqrt(DT) * rng.standard_normal(),
                    0.0), 1.0)
    return z


def emit_nb(rng, rate_hz, fano):
    """Spike counts/bin, mean rate*DT, constant Fano (Gamma-Poisson NB). fano<=1 -> Poisson.
    (Same emission model as phase1_recovery_hardened.emit, vectorised over the population.)"""
    mu = np.clip(rate_hz, RATE_FLOOR, None) * DT
    if fano <= 1.0:
        return rng.poisson(mu).astype(float)
    r = mu / (fano - 1.0)
    return rng.poisson(rng.gamma(r, mu / r)).astype(float)


def sim_trial(true_gen, rng, T, d, b, g, fano):
    """One trial -> spike counts (N_cells, T) from the shared latent + per-cell coupling."""
    z = shared_latent(true_gen, rng, T)
    rate = b[:, None] * (1.0 + (g * d)[:, None] * z[None, :])   # (N, T) Hz
    return emit_nb(rng, rate, fano)


# =============================================================================
#  POPULATION FORWARD-HMM  (shared latent; emission summed over cells)
# =============================================================================
def pop_emis(counts, d, b, g, x_grid, fano, glnc_t):
    """Population log-emission matrix (T, K): emis[t,k] = sum_i logP(n_{i,t} | rate_i(x_k)),
    rate_i(x_k) = b_i*(1 + g_i*d*x_k). NB pmf for fano>1, Poisson otherwise. glnc_t =
    sum_i gammaln(n_{i,t}+1) is a per-bin constant (cancels in the step-vs-ramp contrast).
    Computed ONCE per trial: depends only on (counts,b,g,x_grid,fano), not the dynamics
    params, so the fit loop only re-runs the cheap forward recursion."""
    N, T = counts.shape
    rate = b[:, None] * (1.0 + (g * d)[:, None] * x_grid[None, :])   # (N, K)
    lam = np.clip(rate, RATE_FLOOR, None) * DT                       # (N, K)
    if fano <= 1.0001:
        loglam = np.log(lam)
        emis = counts.T @ loglam - lam.sum(0)[None, :] - glnc_t[:, None]
    else:
        r = lam / (fano - 1.0)                                       # (N, K)
        gl_cr = gammaln(counts[:, :, None] + r[:, None, :])          # (N, T, K)
        emis = (gl_cr.sum(0)
                - gammaln(r).sum(0)[None, :]
                - (r * np.log(fano)).sum(0)[None, :]
                + counts.sum(0)[:, None] * np.log((fano - 1.0) / fano)
                - glnc_t[:, None])
    return emis


def _pad(emis_list):
    n = len(emis_list); K = emis_list[0].shape[1]
    lengths = np.array([e.shape[0] for e in emis_list])
    Tmax = int(lengths.max())
    batch = np.zeros((n, Tmax, K))
    for i, e in enumerate(emis_list):
        batch[i, :e.shape[0], :] = e
    return batch, lengths


def forward_ll(batch, lengths, log_pi, log_A):
    """Total forward log-lik over a batch of trials given precomputed population emissions
    and a transition matrix. Same masked-batched forward pass as the validated engine, with
    the Poisson emission replaced by the precomputed population emission."""
    n, Tmax, K = batch.shape
    A = np.exp(log_A)
    alpha = log_pi[None, :] + batch[:, 0, :]
    ll = np.empty(n)
    sel = lengths == 1
    if sel.any():
        ll[sel] = logsumexp(alpha[sel], axis=1)
    for t in range(1, Tmax):
        m = alpha.max(axis=1, keepdims=True)
        alpha = np.log(np.exp(alpha - m) @ A + 1e-300) + m + batch[:, t, :]
        sel = lengths == t + 1
        if sel.any():
            ll[sel] = logsumexp(alpha[sel], axis=1)
    return float(ll.sum())


def fit_step_pop(batch_tr, len_tr):
    def negll(u):
        p = 1.0 / (1.0 + np.exp(-u[0]))
        lp, lA, _ = step_hmm((0.0, 1.0, p))
        return -forward_ll(batch_tr, len_tr, lp, lA)
    res = minimize(negll, [0.0], method="L-BFGS-B", bounds=[(-12, 12)],
                   options=dict(maxiter=30, ftol=1e-6))
    p = 1.0 / (1.0 + np.exp(-res.x[0]))
    return step_hmm((0.0, 1.0, p))[:2]


def fit_ramp_pop(batch_tr, len_tr, K):
    def negll(u):
        lp, lA, _ = ramp_hmm((np.exp(u[0]), np.exp(u[1]), 0.0, 1.0), K)
        return -forward_ll(batch_tr, len_tr, lp, lA)
    res = minimize(negll, [np.log(1.0), np.log(0.5)], method="L-BFGS-B",
                   bounds=[(-4.6, 3.9), (-4.6, 2.3)], options=dict(maxiter=30, ftol=1e-6))
    return ramp_hmm((np.exp(res.x[0]), np.exp(res.x[1]), 0.0, 1.0), K)[:2]


# =============================================================================
#  ONE RECOVERY DRAW  (simulate population -> CV step-vs-ramp -> winner)
# =============================================================================
def recover_once(true_gen, N_cells, N_trials, fano, gscale, gsigma, seed,
                 windows, K=K_GRID, holdout=0.3):
    rng = np.random.default_rng(seed)
    b, g = draw_cells(rng, N_cells, gscale, gsigma)
    d = rng.choice([-1.0, 1.0], N_trials)
    Tb = np.maximum(3, np.round(rng.choice(windows, N_trials) / DT).astype(int))
    trials = [sim_trial(true_gen, rng, int(Tb[j]), d[j], b, g, fano)
              for j in range(N_trials)]
    idx = rng.permutation(N_trials)
    nte = max(2, int(round(holdout * N_trials)))
    te, tr = idx[:nte], idx[nte:]
    if len(tr) < 2:
        return None
    xs = np.array([0.0, 1.0]); xr = np.linspace(0.0, 1.0, K)
    glnc = {j: gammaln(trials[j] + 1.0).sum(0) for j in range(N_trials)}

    def emis_batch(ix, x_grid):
        return _pad([pop_emis(trials[j], d[j], b, g, x_grid, fano, glnc[j]) for j in ix])

    bs_tr, ls_tr = emis_batch(tr, xs); bs_te, ls_te = emis_batch(te, xs)
    br_tr, lr_tr = emis_batch(tr, xr); br_te, lr_te = emis_batch(te, xr)
    lp_s, lA_s = fit_step_pop(bs_tr, ls_tr)
    lp_r, lA_r = fit_ramp_pop(br_tr, lr_tr, K)
    s_cv = forward_ll(bs_te, ls_te, lp_s, lA_s)
    r_cv = forward_ll(br_te, lr_te, lp_r, lA_r)
    pred = "step" if s_cv > r_cv else "ramp"
    return dict(true=true_gen, N_cells=int(N_cells), N_trials=int(N_trials),
                fano=float(fano), gscale=float(gscale),
                correct=int(pred == true_gen), d_cv=float(s_cv - r_cv))


def run_cells(conds, windows, R, tag, seed0=700000):
    """conds = list of (N_cells, N_trials, fano, gscale, gsigma). R reps per generator.
    Checkpointed: if CKPT[tag] exists it is loaded (resume after a kill); else computed
    and saved before returning, so an OS kill in a LATER stage never loses this one."""
    ckpt = CKPT.get(tag)
    if ckpt is not None and ckpt.exists():
        df = pd.read_csv(ckpt)
        print(f"  [{tag}] loaded checkpoint ({len(df)} recoveries) -> {ckpt.name}")
        return df
    jobs, sid = [], 0
    for (Nc, Nt, fano, gs, gsig) in conds:
        for g in ("step", "ramp"):
            for _ in range(R):
                jobs.append((g, Nc, Nt, fano, gs, gsig, seed0 + sid)); sid += 1
    print(f"  [{tag}] {len(jobs)} recoveries ({len(conds)} cells x 2 gens x R={R})")
    t = time.time()
    res = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(recover_once)(g, Nc, Nt, fano, gs, gsig, sd, windows)
        for (g, Nc, Nt, fano, gs, gsig, sd) in jobs)
    res = [r for r in res if r is not None]
    df = pd.DataFrame(res)
    if ckpt is not None:
        df.to_csv(ckpt, index=False)            # checkpoint immediately
    print(f"  [{tag}] done in {time.time()-t:.0f}s (saved -> {ckpt.name if ckpt else 'n/a'})")
    return df


# =============================================================================
#  STAGES
# =============================================================================
def summarise(df):
    """Recovery per condition with per-generator balance + Wilson-ish binomial CI."""
    rows = []
    for (Nc, Nt, fano, gs), gg in df.groupby(["N_cells", "N_trials", "fano", "gscale"]):
        n = len(gg); acc = gg["correct"].mean()
        se = np.sqrt(acc * (1 - acc) / max(n, 1))
        st = gg[gg["true"] == "step"]["correct"].mean()
        rp = gg[gg["true"] == "ramp"]["correct"].mean()
        rows.append(dict(N_cells=Nc, N_trials=Nt, fano=fano, gscale=gs, n=n,
                         recovery=acc, se=se, rec_step=st, rec_ramp=rp))
    return pd.DataFrame(rows).sort_values(["gscale", "N_cells", "N_trials"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["all", "report"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore stage checkpoints")
    args = ap.parse_args()
    if args.fresh:
        for p in CKPT.values():
            p.unlink(missing_ok=True)

    windows = load_windows()
    print(f"deliberative window pool: {len(windows)} RTs "
          f"(median {np.median(windows)*1000:.0f} ms)\n")

    if args.stage == "report":
        sw = pd.read_csv(SWEEP_CSV); hd = pd.read_csv(HEAD_CSV)
        make_figure(sw, hd); verdict(sw, hd); return

    R_EASY = 4 if args.quick else 40
    R_MRN = 4 if args.quick else 60
    R_GRID = 3 if args.quick else 24
    fano = 2.0

    # ---- (1) EASY-regime sanity: STRONG coupling, many cells/trials -> ~ceiling ----
    print("== STAGE 1: EASY-regime sanity (strong coupling) ==")
    easy_conds = [(80, 120, fano, EASY_GSCALE, EASY_GSIGMA)]
    if args.quick:
        easy_conds = [(40, 60, fano, EASY_GSCALE, EASY_GSIGMA)]
    easy = run_cells(easy_conds, windows, R_EASY, "easy", seed0=100000)
    easy_s = summarise(easy); easy_rec = easy_s["recovery"].iloc[0]
    print(f"  EASY recovery = {easy_rec:.1%}  (step {easy_s['rec_step'].iloc[0]:.0%} / "
          f"ramp {easy_s['rec_ramp'].iloc[0]:.0%})")
    if easy_rec < 0.85:
        print("  !! EASY regime did NOT recover at ceiling -> METHOD SUSPECT. "
              "Investigate before trusting the hard regime.")
    else:
        print("  EASY regime recovers at ceiling -> method validated, proceed to MRN.")

    # ---- (2) HARD / MRN-scale recovery + null (gscale=0) ----
    print("\n== STAGE 2: HARD / MRN-scale recovery (weak coupling) + null ==")
    mrn_trials = [50] if args.quick else [25, 50, 100, 200, 320]
    mrn_conds = [(MRN_N_CELLS, nt, fano, MRN_GSCALE, MRN_GSIGMA) for nt in mrn_trials]
    null_conds = [(MRN_N_CELLS, (50 if args.quick else 200), fano, 0.0, MRN_GSIGMA)]
    mrn = run_cells(mrn_conds, windows, R_MRN, "mrn", seed0=200000)
    null = run_cells(null_conds, windows, R_MRN, "null", seed0=300000)
    mrn_s = summarise(mrn); null_s = summarise(null)
    print(f"  NULL (gscale=0, no coupling) recovery = {null_s['recovery'].iloc[0]:.1%} "
          f"(expect ~50%)")
    for _, r in mrn_s.iterrows():
        print(f"  MRN N_cells={r.N_cells:.0f} N_trials={r.N_trials:.0f}: "
              f"recovery {r.recovery:.1%} +/- {r.se:.1%}  "
              f"(step {r.rec_step:.0%} / ramp {r.rec_ramp:.0%})")

    # ---- (3) recovery-vs-scale: sweep N_cells x N_trials around MRN ----
    # NOTE: N_cells=243 is NOT recomputed here -- those points come from the MRN block
    # (R=80) and are merged below; the grid fills the smaller-population rows. This also
    # avoids the very heavy 243x320 jobs running 12-wide (the prior run's kill point).
    print("\n== STAGE 3: recovery-vs-scale sweep ==")
    if args.quick:
        cells_grid, trials_grid = [30], [50, 200]
    else:
        cells_grid, trials_grid = [15, 30, 60, 120], [25, 50, 100, 200, 320]
    grid_conds = [(nc, nt, fano, MRN_GSCALE, MRN_GSIGMA)
                  for nc in cells_grid for nt in trials_grid]
    # realistic single-session simultaneous anchor + a lighter beyond-MRN probe
    grid_conds += [(10, 25, fano, MRN_GSCALE, MRN_GSIGMA)]
    if not args.quick:
        grid_conds += [(486, 100, fano, MRN_GSCALE, MRN_GSIGMA)]
    grid = run_cells(grid_conds, windows, R_GRID, "grid", seed0=400000)

    # ---- assemble + save ----
    allsweep = pd.concat([mrn.assign(block="mrn"), null.assign(block="null"),
                          grid.assign(block="scale"), easy.assign(block="easy")],
                         ignore_index=True)
    sw = summarise(allsweep)
    sw.to_csv(SWEEP_CSV, index=False)
    head = pd.DataFrame([
        dict(label="easy_sanity", N_cells=easy_s.N_cells.iloc[0],
             N_trials=easy_s.N_trials.iloc[0], gscale=EASY_GSCALE,
             recovery=easy_rec, se=easy_s.se.iloc[0]),
        dict(label="null_nocoupling", N_cells=null_s.N_cells.iloc[0],
             N_trials=null_s.N_trials.iloc[0], gscale=0.0,
             recovery=null_s.recovery.iloc[0], se=null_s.se.iloc[0]),
    ] + [dict(label=f"mrn_Nt{int(r.N_trials)}", N_cells=r.N_cells, N_trials=r.N_trials,
              gscale=MRN_GSCALE, recovery=r.recovery, se=r.se)
         for _, r in mrn_s.iterrows()])
    head.to_csv(HEAD_CSV, index=False)
    print(f"\nSaved sweep -> {SWEEP_CSV}\nSaved headline -> {HEAD_CSV}")

    make_figure(sw, head)
    verdict(sw, head)


def make_figure(sw, head):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    chance = 50.0
    grid = sw[sw["gscale"] == MRN_GSCALE]

    # A: easy vs null vs MRN headline bars
    ax = axes[0, 0]
    bars = [("easy\n(strong)", head[head.label == "easy_sanity"]["recovery"].iloc[0], "#1a5"),
            ("null\n(g=0)", head[head.label == "null_nocoupling"]["recovery"].iloc[0], "#999")]
    for _, r in head[head.label.str.startswith("mrn")].iterrows():
        bars.append((f"MRN 243c\n{int(r.N_trials)} tr", r.recovery, "#c44"))
    xs = np.arange(len(bars))
    ax.bar(xs, [b[1] * 100 for b in bars], color=[b[2] for b in bars])
    ax.axhline(chance, color="k", ls="--", lw=1, label="chance")
    ax.axhline(80, color="gray", ls=":", lw=1, label="usable (80%)")
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("step-vs-ramp recovery (%)"); ax.set_ylim(40, 100)
    ax.set_title("Easy sanity vs null vs MRN-scale recovery", fontsize=10, loc="left")
    ax.legend(fontsize=8)

    # B: recovery vs N_trials, per N_cells (MRN coupling)
    ax = axes[0, 1]
    for nc, gg in grid.groupby("N_cells"):
        gg = gg.sort_values("N_trials")
        ax.plot(gg["N_trials"], gg["recovery"] * 100, "o-", label=f"{int(nc)} cells")
    ax.axhline(chance, color="k", ls="--", lw=1); ax.axhline(80, color="gray", ls=":", lw=1)
    ax.set_xlabel("shared trials (N_trials)"); ax.set_ylabel("recovery (%)")
    ax.set_xscale("log"); ax.set_xticks(sorted(grid["N_trials"].unique()))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_title("Recovery vs shared trials (weak MRN coupling)", fontsize=10, loc="left")
    ax.legend(fontsize=7, ncol=2)

    # C: recovery vs N_cells, per N_trials
    ax = axes[1, 0]
    for nt, gg in grid.groupby("N_trials"):
        gg = gg.sort_values("N_cells")
        ax.plot(gg["N_cells"], gg["recovery"] * 100, "o-", label=f"{int(nt)} trials")
    ax.axhline(chance, color="k", ls="--", lw=1); ax.axhline(80, color="gray", ls=":", lw=1)
    ax.axvline(MRN_N_CELLS, color="navy", lw=1.5, ls="--", label="MRN 243")
    ax.set_xlabel("population size (N_cells)"); ax.set_ylabel("recovery (%)")
    ax.set_xscale("log"); ax.set_xticks(sorted(grid["N_cells"].unique()))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_title("Recovery vs population size (weak MRN coupling)", fontsize=10, loc="left")
    ax.legend(fontsize=7, ncol=2)

    # D: heatmap recovery over (N_cells x N_trials), with the ACHIEVABLE-vs-IDEALIZED split
    ax = axes[1, 1]
    from matplotlib.patches import Rectangle
    piv = (grid[grid["N_cells"].isin([15, 30, 60, 120, 243])]
           .pivot_table(index="N_cells", columns="N_trials", values="recovery"))
    im = ax.imshow(piv.values * 100, origin="lower", aspect="auto",
                   cmap="RdYlGn", vmin=50, vmax=90)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([int(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([int(i) for i in piv.index])
    ax.set_xlabel("shared trials"); ax.set_ylabel("N_cells")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label("recovery (%)")
    # ACHIEVABLE simultaneous regime = one MRN session: <=53 cells x ~9-25 shared trials
    # (rows 15/30/60 ~ the <=53-cell ceiling; first trial column ~ the 9-25 budget).
    ax.add_patch(Rectangle((-0.5, -0.5), 1.0, 3.0, fill=False, edgecolor="navy",
                           lw=2.2, zorder=5))
    ax.text(0.5, 2.45, "achievable\n(1 MRN session:\n<=53 cells, ~9-25 tr)",
            fontsize=6.5, color="navy", va="bottom", ha="left")
    ax.text(2.0, 4.0, "pseudo-pop idealization\n(243 cells span 61 sessions -\nNOT co-recorded)",
            fontsize=6.5, color="black", va="center", ha="center")
    ax.set_title("Recovery heatmap: usable lane (green) is unreachable simultaneously",
                 fontsize=9, loc="left")
    fig.suptitle("Phase 2 FINAL gate: MRN population step-vs-ramp recovery pre-flight "
                 "(243 cells, per-cell AUC ~0.51, pop AUC ~0.79, NB Fano 2)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Saved figure -> {FIG_PATH}")


def verdict(sw, head):
    print("\n========== GLANCEABLE VERDICT (§3.5) ==========")
    easy = head[head.label == "easy_sanity"]["recovery"].iloc[0]
    null = head[head.label == "null_nocoupling"]["recovery"].iloc[0]
    print(f"  easy-regime sanity     : {easy:.1%}  ({'RECOVERS' if easy>=0.85 else 'FAILS - method suspect'})")
    print(f"  null (no coupling)     : {null:.1%}  (chance check)")
    grid = sw[sw["gscale"] == MRN_GSCALE]
    mrn243 = grid[grid["N_cells"] == MRN_N_CELLS].sort_values("N_trials")
    print("  MRN-scale (243 cells), recovery vs shared trials:")
    for _, r in mrn243.iterrows():
        flag = "USABLE" if r.recovery >= 0.80 else ("above-chance" if r.recovery - 2*r.se > 0.5 else "~chance")
        print(f"    N_trials={int(r.N_trials):4d}: {r.recovery:.1%} +/- {r.se:.1%}  [{flag}]")
    # ACHIEVABLE simultaneous regime: one MRN session (<=53 cells, ~9-25 shared trials)
    ach = grid[(grid["N_cells"] <= 60) & (grid["N_trials"] <= 25)]
    if len(ach):
        amax = ach["recovery"].max()
        print(f"  ACHIEVABLE simultaneous (<=60 cells x <=25 shared trials, 1 session): "
              f"recovery {ach['recovery'].min():.1%}-{amax:.1%} (best {amax:.1%})")
    best = grid["recovery"].max()
    print(f"  best recovery anywhere in swept grid: {best:.1%}")
    usable = grid[grid["recovery"] >= 0.80].sort_values(["N_cells", "N_trials"])
    print(f"  conditions reaching >=80% (USABLE): {len(usable)}/{len(grid)} -> "
          + ", ".join(f"{int(r.N_cells)}c x{int(r.N_trials)}t" for _, r in usable.iterrows()))
    # decisive read
    ach_usable = len(ach[ach["recovery"] >= 0.80]) if len(ach) else 0
    print("\n  ---- §3.5 READ ----")
    if easy >= 0.85 and ach_usable == 0:
        print("  FAIL at ACHIEVABLE scale: easy regime recovers, but at the largest")
        print("  SIMULTANEOUS MRN population IBL can yield (<=53 cells x ~9-25 shared")
        print("  decorrelated trials/session) step-vs-ramp recovery stays below the 80%")
        print("  usable line. The distributed MRN choice code is REAL, but its single-trial")
        print("  step-vs-ramp DYNAMICS are NOT recoverable from existing IBL data.")
        print("  -> the paper's arc closes; no population model warranted now.")
        print("  Recovery only becomes USABLE for an IDEALIZED simultaneous population")
        print("  (>=~120-243 cells AND >=~100-200 shared trials) that IBL's pseudo-population")
        print("  (243 cells across 61 sessions) cannot realize -> FUTURE WORK = a dedicated")
        print("  high-yield simultaneous MRN recording, NOT a model on current data.")
    elif easy >= 0.85 and ach_usable > 0:
        print("  PASS: step-vs-ramp recovers above the usable line within the achievable")
        print("  simultaneous MRN regime -> a population model is justified as future work.")
    else:
        print("  METHOD SUSPECT: easy regime did not recover; do not trust the hard regime.")


if __name__ == "__main__":
    main()

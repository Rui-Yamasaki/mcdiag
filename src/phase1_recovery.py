"""Phase 1 / step 2 - the FORMAL identifiability gate (recovery simulation).

Self-contained numpy/scipy engine in which BOTH single-trial latent-dynamics
families are expressed as exact forward-algorithm Poisson HMMs, so the
stepping-vs-ramping comparison shares ONE inference routine and ONE held-out
cross-validated metric (no structural bias toward either family):

  STEPPING : 2-state left-to-right Poisson HMM. Latent sits at a baseline rate,
             then jumps ONCE (geometric step time) to a committed rate; state 1
             is absorbing.  params = (lam_lo, lam_hi, p_step)         -> 3
  RAMPING  : discretized diffusion-to-bound. A 1-D latent x in [0,1] does drift-
             diffusion (absorbing upper bound), discretized onto a K-point grid
             -> an exact HMM whose marginal likelihood converges to the
             continuous model as K grows.  rate = lam_lo + (lam_hi-lam_lo)*x.
             params = (drift, sigma, lam_lo, lam_hi)                  -> 4

Both are fit by maximizing the TRAIN forward log-likelihood (L-BFGS-B) and scored
by the SAME forward log-likelihood on HELD-OUT trials (k-fold CV). The model with
the higher held-out CV log-likelihood wins; recovery = winner == true generator.

Stages (see --stage):
  validate : (1) cross-check the Poisson-HMM forward likelihood against the
                 maintained reference hmmlearn.PoissonHMM;
             (2) show the ramping likelihood/verdict CONVERGES as the latent grid
                 K refines;
             (3) Part A2 reproduction sanity: recover step-from-step and
                 ramp-from-ramp on GENEROUS data (long windows, high FR, many
                 trials) -- must pass at high rate before the sweep.
  sweep    : Part B recovery sweep over trial count x firing rate, with per-trial
             window lengths DRAWN from the measured 500-2000 ms deliberative
             distribution (curated slow low-contrast IBL trials). Saves heatmaps
             + a results CSV.

No real spike data is modeled here -- everything is simulated. No git.

Phase 3 TODO (do NOT build now): run the canonical ssm / Latimer code on the
FINAL real data via WSL/Colab to satisfy reviewers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import gammaln, logsumexp  # noqa: E402

from ibl_one import DATA_DIR, PROJECT_ROOT  # noqa: E402

# --- fixed numerics ----------------------------------------------------------
DT = 0.025                # bin width (s)
RATE_FLOOR = 1e-3         # Hz, avoid log(0)
NEG_INF = -1e30
POOLED_PARQUET = DATA_DIR / "phase1_trials_pooled.parquet"
FIG_PATH = PROJECT_ROOT / "figures" / "phase1_recovery_sim.png"
SWEEP_CSV = PROJECT_ROOT / "results" / "phase1_recovery_sweep.csv"
VALID_JSON = PROJECT_ROOT / "results" / "phase1_recovery_validation.json"

# --- generator parameters (true data-generating processes) -------------------
# Stepping and ramping are MATCHED on marginal rate envelope (both rise from
# r_lo=2/3*FR to r_hi=4/3*FR, trial-mean ~ FR); they differ ONLY in dynamics
# (instantaneous jump vs gradual diffusion). FR is swept; these set the shape.
def gen_rates(fr):
    return (2.0 / 3.0) * fr, (4.0 / 3.0) * fr      # r_lo, r_hi  (mean ~ fr)


GEN_DRIFT = 1.0           # ramp latent units / s  (reaches bound ~1 s)
GEN_SIGMA = 0.4           # ramp diffusion std (latent units / sqrt(s)); moderate
                          # -> a clearly GRADUAL ramp, not a bound-slamming step.
                          # Larger sigma makes ramps look steppier (harder); this
                          # is the main robustness knob for the recovery rates.


# =============================================================================
#  EXACT FORWARD-ALGORITHM POISSON-HMM ENGINE  (shared by BOTH families)
# =============================================================================
def build_dataset(counts_list):
    """Pad variable-length trials to a common Tmax for one batched forward pass.

    Returns (counts (n,Tmax), lengths (n,), gammaln_const (n,Tmax), Tmax, n).
    Trials shorter than Tmax are zero-padded; each trial's log-likelihood is read
    off at its true final bin, so padding never contaminates the result.
    """
    n = len(counts_list)
    lengths = np.array([len(c) for c in counts_list], dtype=int)
    Tmax = int(lengths.max())
    counts = np.zeros((n, Tmax), dtype=float)
    for i, c in enumerate(counts_list):
        counts[i, :len(c)] = c
    return counts, lengths, gammaln(counts + 1.0), Tmax, n


def _lse(a, axis):
    """Numerically stable log-sum-exp (faster than scipy for the inner loop)."""
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def model_loglik(dataset, log_pi, log_A, rates):
    """Total forward log-likelihood of a dataset under given HMM params.

    Single masked-batched forward pass over all trials at once. The per-step
    transition update logsumexp_i(alpha_i + logA_ij) is computed in the log
    domain via a BLAS matrix multiply -- exp(alpha - max) @ A -- which avoids
    materialising the (n,K,K) tensor and is ~10-50x faster at large K. Each
    trial's LL is captured at its own final bin, so zero-padding is inert.
    """
    counts, lengths, glnc, Tmax, n = dataset
    if not np.all(np.isfinite(rates)):                   # reject blown-up fits
        return -1e18
    rdt = np.clip(rates, RATE_FLOOR, None) * DT
    log_rdt = np.log(rdt)
    A = np.exp(log_A)                                     # (K,K) transition probs
    emis = (counts[:, :, None] * log_rdt[None, None, :]
            - rdt[None, None, :] - glnc[:, :, None])      # (n, Tmax, K)
    alpha = log_pi[None, :] + emis[:, 0, :]               # (n, K), log domain
    ll = np.empty(n)
    sel = lengths == 1
    if sel.any():
        ll[sel] = _lse(alpha[sel], axis=1)
    for t in range(1, Tmax):
        m = alpha.max(axis=1, keepdims=True)             # (n,1) stabiliser
        alpha = np.log(np.exp(alpha - m) @ A + 1e-300) + m + emis[:, t, :]
        sel = lengths == t + 1
        if sel.any():
            ll[sel] = _lse(alpha[sel], axis=1)
    return float(ll.sum())


# --- STEPPING: 2-state left-to-right Poisson HMM -----------------------------
def step_hmm(params):
    lam_lo, lam_hi, p = params
    log_pi = np.array([0.0, NEG_INF])                     # start in baseline
    p = min(max(p, 1e-6), 1 - 1e-6)
    log_A = np.array([[np.log(1 - p), np.log(p)],
                      [NEG_INF, 0.0]])                    # state 1 absorbing
    return log_pi, log_A, np.array([lam_lo, lam_hi])


# --- RAMPING: discretized drift-diffusion to absorbing bound -----------------
def ramp_hmm(params, K):
    drift, sigma, lam_lo, lam_hi = params
    x = np.linspace(0.0, 1.0, K)
    sd = max(sigma, 1e-4) * np.sqrt(DT)
    mu = x + drift * DT                                   # mean next position
    # Gaussian transition weights on the grid, row-normalized (tails fold onto
    # the edge states -> reflecting at 0, and the top row is forced absorbing).
    z = (x[None, :] - mu[:, None]) / sd
    logw = -0.5 * z * z
    w = np.exp(logw - logw.max(axis=1, keepdims=True))
    A = w / w.sum(axis=1, keepdims=True)
    A[-1, :] = 0.0
    A[-1, -1] = 1.0                                       # absorbing bound
    log_A = np.log(A + 1e-300)
    log_pi = np.full(K, NEG_INF)
    log_pi[0] = 0.0                                       # start at x = 0
    rates = lam_lo + (lam_hi - lam_lo) * x
    return log_pi, log_A, rates


# =============================================================================
#  FITTING  (maximize TRAIN forward log-likelihood; identical optimizer setup)
# =============================================================================
def _data_inits(dataset):
    """Generator-AGNOSTIC inits from early/late bin means (Hz) and mean length.

    These peek only at the data, never at the true generating model, so neither
    family gets a head start.
    """
    counts, lengths, _, _, n = dataset
    early_c = early_t = late_c = late_t = 0.0
    for i in range(n):
        L = int(lengths[i]); ne = max(1, L // 5)
        early_c += counts[i, :ne].sum(); early_t += ne
        late_c += counts[i, L - ne:L].sum(); late_t += ne
    r_early = max(early_c / max(early_t, 1) / DT, RATE_FLOOR)
    r_late = max(late_c / max(late_t, 1) / DT, RATE_FLOOR)
    mean_T = float(np.mean(lengths)) * DT                 # mean window (s)
    return r_early, r_late, mean_T


def fit_step(dataset):
    r_early, r_late, _ = _data_inits(dataset)
    x0 = np.array([np.log(r_early), np.log(max(r_late, r_early)), np.log(0.05)])

    def negll(u):
        p = 1.0 / (1.0 + np.exp(-u[2]))
        return -model_loglik(dataset, *step_hmm((np.exp(u[0]), np.exp(u[1]), p)))
    res = minimize(negll, x0, method="L-BFGS-B",
                   bounds=[(-4.6, 6.2), (-4.6, 6.2), (-12, 12)],  # rates<=~500 Hz
                   options=dict(maxiter=30, ftol=1e-6))
    u = res.x
    return (np.exp(u[0]), np.exp(u[1]), 1.0 / (1.0 + np.exp(-u[2]))), -res.fun


def fit_ramp(dataset, K):
    r_early, r_late, mean_T = _data_inits(dataset)
    drift0 = 1.0 / max(mean_T, DT)                        # latent crosses ~1 over window
    x0 = np.array([np.log(drift0), np.log(0.5),           # neutral sigma init
                   np.log(r_early), np.log(max(r_late, r_early))])

    def negll(u):
        return -model_loglik(
            dataset, *ramp_hmm((np.exp(u[0]), np.exp(u[1]),
                                np.exp(u[2]), np.exp(u[3])), K))
    res = minimize(negll, x0, method="L-BFGS-B",
                   bounds=[(-4.6, 3.9), (-4.6, 2.3), (-4.6, 6.2), (-4.6, 6.2)],
                   options=dict(maxiter=30, ftol=1e-6))
    u = res.x
    return (np.exp(u[0]), np.exp(u[1]), np.exp(u[2]), np.exp(u[3])), -res.fun


# =============================================================================
#  SIMULATION  (true generators; matched marginal rate envelope)
# =============================================================================
def simulate(true_gen, rng, Tbins, fr):
    """Return list of spike-count arrays (one per trial)."""
    r_lo, r_hi = gen_rates(fr)
    out = []
    if true_gen == "step":
        for T in Tbins:
            s = rng.integers(1, T) if T > 1 else 1        # step bin (uniform)
            rate = np.where(np.arange(T) < s, r_lo, r_hi)
            out.append(rng.poisson(rate * DT).astype(float))
    else:  # ramp
        for T in Tbins:
            x = 0.0; rates = np.empty(T)
            for t in range(T):
                rates[t] = r_lo + (r_hi - r_lo) * x
                x = x + GEN_DRIFT * DT + GEN_SIGMA * np.sqrt(DT) * rng.standard_normal()
                x = min(max(x, 0.0), 1.0)                 # reflect 0, absorb 1
            out.append(rng.poisson(rates * DT).astype(float))
    return out


# =============================================================================
#  ONE RECOVERY TRIAL  (simulate -> k-fold CV both models -> winner)
# =============================================================================
def recover_once(true_gen, N, fr, seed, windows_pool, K, holdout=0.3):
    """One Monte-Carlo CV draw: simulate -> single held-out split -> winner.

    A fresh random holdout per call; averaging the 'correct' flag over many calls
    (different seeds) gives the recovery rate. Both models share the SAME split,
    the SAME forward-LL inference, and the SAME held-out scoring -> unbiased.
    """
    rng = np.random.default_rng(seed)
    win = rng.choice(windows_pool, size=N)                # seconds / trial
    Tbins = np.maximum(2, np.round(win / DT).astype(int))
    counts = simulate(true_gen, rng, Tbins, fr)

    idx = rng.permutation(N)
    ntest = max(1, int(round(holdout * N)))
    test, train = idx[:ntest], idx[ntest:]
    ds_tr = build_dataset([counts[i] for i in train])
    ds_te = build_dataset([counts[i] for i in test])
    try:
        ps, _ = fit_step(ds_tr)
        step_cv = model_loglik(ds_te, *step_hmm(ps))
    except Exception:
        step_cv = NEG_INF
    try:
        pr, _ = fit_ramp(ds_tr, K)
        ramp_cv = model_loglik(ds_te, *ramp_hmm(pr, K))
    except Exception:
        ramp_cv = NEG_INF
    pred = "step" if step_cv > ramp_cv else "ramp"
    return dict(true=true_gen, N=int(N), fr=float(fr),
                pred=pred, correct=int(pred == true_gen),
                d_cvll=float(step_cv - ramp_cv))


# =============================================================================
#  WINDOW DISTRIBUTION  (measured 500-2000 ms deliberative window)
# =============================================================================
def load_windows():
    """Reaction times (s) from curated trained-ephys, low-contrast, slow trials."""
    df = pd.read_parquet(POOLED_PARQUET)
    cl = np.nan_to_num(df["contrastLeft"].to_numpy(float))
    cr = np.nan_to_num(df["contrastRight"].to_numpy(float))
    absc = np.round(np.fmax(cl, cr) * 100, 4)
    rt = df["firstMovement_times"].to_numpy(float) - df["stimOn_times"].to_numpy(float)
    # training sessions = contain a 50% contrast trial (drop them)
    train_eids = set(df.loc[absc == 50.0, "eid"].unique())
    trained = ~df["eid"].isin(train_eids)
    low = np.isin(absc, [0.0, 6.25, 12.5])
    resp = df["choice"].to_numpy(float) != 0
    mask = trained.to_numpy() & low & resp & np.isfinite(rt) & (rt >= 0.5) & (rt <= 2.0)
    win = rt[mask]
    return win


# =============================================================================
#  STAGE: VALIDATE
# =============================================================================
def validate_against_hmmlearn():
    """Cross-check our Poisson-HMM forward likelihood vs hmmlearn.PoissonHMM."""
    from hmmlearn.hmm import PoissonHMM
    import hmmlearn

    rng = np.random.default_rng(0)
    # an UNCONSTRAINED 2-state Poisson HMM with fixed params
    pi = np.array([0.7, 0.3])
    A = np.array([[0.85, 0.15], [0.1, 0.9]])
    lams = np.array([3.0, 12.0]) * DT          # per-bin means (lambdas)

    # sample sequences from this HMM
    seqs, our_ll, ref_ll = [], 0.0, 0.0
    ref = PoissonHMM(n_components=2, init_params="", params="")
    ref.startprob_ = pi
    ref.transmat_ = A
    ref.lambdas_ = lams.reshape(2, 1)

    lengths = []
    big = []
    for _ in range(40):
        T = int(rng.integers(20, 60))
        s = 0 if rng.random() < pi[0] else 1
        counts = np.empty(T)
        for t in range(T):
            counts[t] = rng.poisson(lams[s])
            s = 0 if rng.random() < A[s, 0] else 1
        seqs.append(counts); lengths.append(T); big.append(counts.reshape(-1, 1))
        # our forward LL for this single sequence with the SAME fixed params
        ds = build_dataset([counts])
        our_ll += model_loglik(ds, np.log(pi),
                               np.log(A), lams / DT)      # rates in Hz
    ref_ll = ref.score(np.vstack(big), lengths=lengths)
    diff = abs(our_ll - ref_ll)
    print("\n========== validation 1: forward-LL vs hmmlearn.PoissonHMM ==========")
    print(f"  hmmlearn {hmmlearn.__version__} | 40 seqs, fixed 2-state params")
    print(f"  our total forward log-lik : {our_ll:.6f}")
    print(f"  hmmlearn  .score()        : {ref_ll:.6f}")
    print(f"  abs difference            : {diff:.2e}  "
          f"({'MATCH' if diff < 1e-6 else 'MISMATCH'})")
    return dict(our_ll=our_ll, ref_ll=ref_ll, abs_diff=diff,
                match=bool(diff < 1e-6), hmmlearn=hmmlearn.__version__)


def grid_convergence(windows_pool, n_jobs, seeds=range(40)):
    """Show the ramping verdict converges as the latent grid K refines.

    IMPORTANT: the SAME simulated datasets are used across K (seed independent of
    K), so differences reflect grid resolution alone -- not Monte-Carlo noise.
    """
    print("\n========== validation 2: ramping grid-convergence (paired) ==========")
    print("  condition: N=80, FR=20 Hz, measured windows; SAME datasets across K")
    rows = []
    for K in (30, 50, 75, 100):
        recs = Parallel(n_jobs=n_jobs)(
            delayed(recover_once)(g, 80, 20.0, 70000 + i, windows_pool, K)
            for g in ("step", "ramp") for i in seeds)
        rate = np.mean([r["correct"] for r in recs])
        dmean = np.mean([r["d_cvll"] for r in recs])
        rows.append(dict(K=K, recovery=float(rate), mean_d_cvll=float(dmean)))
        print(f"  K={K:3d} : recovery={rate:5.1%}   mean(step-ramp CV-LL)={dmean:+7.3f}")
    # judge stability over the converged tail (K >= 50)
    tail = [r["recovery"] for r in rows if r["K"] >= 50]
    spread = max(tail) - min(tail)
    print(f"  recovery spread for K>=50 : {spread:.1%} "
          f"({'STABLE -> settle on K=50' if spread <= 0.05 else 'still moving'})")
    return rows


def part_a2_sanity(n_jobs, seeds=range(40)):
    """GENEROUS data reproduction sanity: long windows, high FR, many trials."""
    print("\n========== Part A2: reproduction sanity (GENEROUS data) ==========")
    print("  windows fixed 1.75 s, FR=40 Hz, N=150 trials, K=50")
    gen_windows = np.full(2000, 1.75)
    K = 50
    out = {}
    for true_gen in ("step", "ramp"):
        recs = Parallel(n_jobs=n_jobs)(
            delayed(recover_once)(true_gen, 150, 40.0, 90000 + i, gen_windows, K)
            for i in seeds)
        rate = np.mean([r["correct"] for r in recs])
        out[true_gen] = float(rate)
        print(f"  true={true_gen:4s} -> recovered {rate:5.1%} "
              f"({sum(r['correct'] for r in recs)}/{len(recs)})")
    overall = float(np.mean(list(out.values())))
    print(f"  overall reproduction recovery : {overall:.1%} "
          f"({'PASS' if overall >= 0.85 else 'FAIL - implementation suspect'})")
    return out, overall


# =============================================================================
#  STAGE: SWEEP  (Part B)
# =============================================================================
def run_sweep(windows_pool, Ns, FRs, R, K, n_jobs):
    jobs = []
    sid = 0
    for true_gen in ("step", "ramp"):
        for N in Ns:
            for fr in FRs:
                for r in range(R):
                    jobs.append((true_gen, N, fr, 13000 + sid))
                    sid += 1
    print(f"\n========== Part B: recovery sweep ==========")
    print(f"  {len(Ns)}x{len(FRs)} grid x 2 generators x R={R}  = {len(jobs)} sims"
          f"  (K={K}, dt={DT*1000:.0f} ms, {n_jobs} workers)")
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(recover_once)(tg, N, fr, sd, windows_pool, K)
        for (tg, N, fr, sd) in jobs)
    print(f"  swept in {time.time()-t0:.0f} s")
    return pd.DataFrame(results)


def make_heatmaps(df, Ns, FRs, n_per_session=40):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, tg, title in zip(axes, ("step", "ramp"),
                             ("true = STEPPING", "true = RAMPING")):
        sub = df[df["true"] == tg]
        grid = (sub.groupby(["fr", "N"])["correct"].mean()
                .reindex(pd.MultiIndex.from_product([FRs, Ns],
                                                    names=["fr", "N"]))
                .unstack("N"))
        im = ax.imshow(grid.values, origin="lower", aspect="auto",
                       cmap="RdYlGn", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(Ns))); ax.set_xticklabels(Ns)
        ax.set_yticks(range(len(FRs))); ax.set_yticklabels(FRs)
        ax.set_xlabel("trials / neuron  (N)")
        ax.set_title(title, fontsize=11, loc="left")
        for i in range(len(FRs)):
            for j in range(len(Ns)):
                v = grid.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                            fontsize=8, color="black")
        # mark IBL-plausible single-session count (~40 slow low-contrast trials)
        if n_per_session in Ns:
            ax.axvline(list(Ns).index(n_per_session), color="navy", lw=2, ls="--")
    axes[0].set_ylabel("mean firing rate (Hz)")
    cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    cb.set_label("recovery rate (true model picked)")
    fig.suptitle("Phase 1 step 2: step-vs-ramp recovery at IBL-realistic "
                 "deliberation windows (500-2000 ms)", fontsize=12)
    fig.text(0.5, 0.005, "navy dashed = N≈40 (≈ one session of slow low-contrast "
             "trials); pooling across sessions → N≈160-320.   "
             "red ≈ chance (non-identifiable), green ≈ recoverable",
             ha="center", fontsize=9, color="dimgray")
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved heatmaps -> {FIG_PATH}")


def sweep_verdict(df, Ns, FRs):
    print("\n========== GLANCEABLE VERDICT ==========")
    overall = df.groupby(["N", "fr"])["correct"].mean()
    # recovery at N=40 (single session) across FR
    if 40 in Ns:
        r40 = df[df["N"] == 40].groupby("fr")["correct"].mean()
        print(f"  at N=40 (~1 session): recovery by FR(Hz) = "
              + ", ".join(f"{fr}:{r40.get(fr, np.nan)*100:.0f}%" for fr in FRs))
    # min (N, FR) reaching >=80%
    ok = overall[overall >= 0.80]
    if len(ok):
        minN = min(n for (n, f) in ok.index)
        # for that minN, smallest FR with >=80%
        cand = [(n, f) for (n, f) in ok.index]
        best = sorted(cand, key=lambda t: (t[0], t[1]))[0]
        print(f"  >=80% recovery first reached at N>={best[0]}, FR>={best[1]} Hz")
        print(f"  cells reaching >=80%: {len(ok)}/{len(overall)}")
    else:
        print("  >=80% recovery NOT reached anywhere in the swept grid")
    return overall


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["validate", "sweep", "all"],
                    default="validate")
    ap.add_argument("--R", type=int, default=24, help="repeats per sweep cell")
    ap.add_argument("--K", type=int, default=50, help="ramp latent grid (converged)")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = ap.parse_args()

    Ns = [20, 40, 80, 160, 320]
    FRs = [2, 5, 10, 20, 40]
    if args.quick:
        Ns, FRs, args.R = [20, 80], [5, 20], 4

    windows = load_windows()
    print(f"Deliberative window pool: {len(windows):,} curated slow low-contrast "
          f"RTs in [0.5,2.0] s  (median {np.median(windows)*1000:.0f} ms)")

    if args.stage in ("validate", "all"):
        v1 = validate_against_hmmlearn()
        if not v1["match"]:
            raise SystemExit("STOP: forward-LL disagrees with hmmlearn reference.")
        v2 = grid_convergence(windows, args.jobs)
        sane, overall = part_a2_sanity(args.jobs)
        VALID_JSON.parent.mkdir(parents=True, exist_ok=True)
        VALID_JSON.write_text(json.dumps(
            dict(hmmlearn_check=v1, grid_convergence=v2,
                 part_a2=sane, part_a2_overall=overall), indent=2))
        print(f"\nSaved validation report -> {VALID_JSON}")
        if overall < 0.85:
            raise SystemExit("STOP: Part A2 reproduction sanity failed (<85%). "
                             "Fix the implementation before sweeping.")

    if args.stage in ("sweep", "all"):
        df = run_sweep(windows, Ns, FRs, args.R, args.K, args.jobs)
        SWEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SWEEP_CSV, index=False)
        print(f"Saved sweep results -> {SWEEP_CSV}  ({len(df)} sims)")
        make_heatmaps(df, Ns, FRs)
        sweep_verdict(df, Ns, FRs)


if __name__ == "__main__":
    main()

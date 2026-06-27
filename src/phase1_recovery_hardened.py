"""Tier-1 hardening of the Phase-1 recovery / identifiability map (PURE SIMULATION).

Reviewers will object that the clean-Poisson recovery map is optimistic because real
neurons are (a) overdispersed and (b) live BETWEEN clean step and clean ramp. This
re-runs the recovery sweep under those two realisms, reusing the VALIDATED
phase1_recovery.py engine (same forward-algorithm Poisson-HMM fitter, same held-out
CV metric). No IBL data, no downloads (only reads the cached deliberative-window pool).

  (1) OVERDISPERSION: negative-binomial emissions at constant Fano factors
      {1 (Poisson), 1.5, 2, 3}; the fitter stays POISSON (tests robustness to the
      realistic mis-specification). A matched-NB fitter is also provided and run on a
      small grid to show whether the right likelihood recovers the loss.
  (2) INTERMEDIATE generators: a continuous "stepiness" knob morphs a clean ramp
      (s=0) to a clean step (s=1) via a sigmoid transition of tunable width -> "steppy
      ramps" (high s) and "rampy steps" (low s). Recovery vs the knob = how separated
      truth must be to discriminate.

RESOURCE-CAPPED to run concurrently with the confirmatory pass: <=2 workers, BLAS
pinned to 1 thread (set in the launch env), process priority lowered. No network.

  python src/phase1_recovery_hardened.py --stage all      # overdisp + stepiness + matched
  python src/phase1_recovery_hardened.py --stage overdisp # just the overdispersion sweep
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import gammaln  # noqa: E402

from ibl_one import PROJECT_ROOT  # noqa: E402
from phase1_recovery import (DT, GEN_DRIFT, GEN_SIGMA, RATE_FLOOR, build_dataset,  # noqa: E402
                             fit_ramp, fit_step, gen_rates, load_windows,
                             model_loglik, ramp_hmm, step_hmm)

NEG = -1e30
K_GRID = 50                      # match the Phase-1 map (converged)
HOLDOUT = 0.3
N_JOBS = 2                       # HARD cap: never starve the confirmatory pass

_PRIORED = False


def lower_priority():
    """Drop THIS process to BelowNormal (Windows SetPriorityClass via ctypes, no
    psutil dependency). Called in the parent AND once per loky worker so every
    compute process yields CPU to the higher-priority confirmatory pass."""
    global _PRIORED
    if _PRIORED:
        return
    _PRIORED = True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p           # else handle truncates
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL_PRIORITY
    except Exception:  # noqa: BLE001
        pass

FIG_PATH = PROJECT_ROOT / "figures" / "phase1_recovery_hardened.png"
OVERDISP_CSV = PROJECT_ROOT / "results" / "phase1_recovery_overdispersion.csv"
STEEP_CSV = PROJECT_ROOT / "results" / "phase1_recovery_stepiness.csv"
MATCHED_CSV = PROJECT_ROOT / "results" / "phase1_recovery_matched_nb.csv"


# --- emissions ---------------------------------------------------------------
def emit(rng, rate_hz, fano):
    """Spike counts/bin with mean rate*DT and a CONSTANT Fano factor (NB via
    Gamma-Poisson). fano<=1 -> Poisson."""
    mu = np.clip(rate_hz, RATE_FLOOR, None) * DT
    if fano <= 1.0:
        return rng.poisson(mu).astype(float)
    r = mu / (fano - 1.0)                       # per-bin dispersion -> Fano = fano
    g = rng.gamma(r, mu / r)
    return rng.poisson(g).astype(float)


# --- generators --------------------------------------------------------------
def sim_canonical(true_gen, rng, Tbins, fr, fano):
    """Phase-1's canonical step/ramp generators, with overdispersed emission."""
    r_lo, r_hi = gen_rates(fr)
    out = []
    if true_gen == "step":
        for T in Tbins:
            s = rng.integers(1, T) if T > 1 else 1
            out.append(emit(rng, np.where(np.arange(T) < s, r_lo, r_hi), fano))
    else:
        for T in Tbins:
            x = 0.0; rates = np.empty(T)
            for t in range(T):
                rates[t] = r_lo + (r_hi - r_lo) * x
                x = min(max(x + GEN_DRIFT * DT
                            + GEN_SIGMA * np.sqrt(DT) * rng.standard_normal(), 0.0), 1.0)
            out.append(emit(rng, rates, fano))
    return out


def sim_stepiness(rng, Tbins, fr, s):
    """Blend the two CANONICAL Phase-1 latents: s=1 -> pure step (jump at a random
    time), s=0 -> pure ramp (drift-diffusion to bound). Intermediate s gives
    "steppy ramps" (low s) and "rampy steps" (high s) on one continuous knob."""
    r_lo, r_hi = gen_rates(fr)
    out = []
    for T in Tbins:
        xr = np.empty(T); x = 0.0                       # ramp (diffusion) latent
        for t in range(T):
            xr[t] = x
            x = min(max(x + GEN_DRIFT * DT
                        + GEN_SIGMA * np.sqrt(DT) * rng.standard_normal(), 0.0), 1.0)
        sb = rng.integers(1, T) if T > 1 else 1         # step latent (jump)
        xs = (np.arange(T) >= sb).astype(float)
        latent = s * xs + (1.0 - s) * xr
        out.append(emit(rng, r_lo + (r_hi - r_lo) * latent, 1.0))
    return out


# --- matched NB likelihood (optional fitter) ---------------------------------
def model_loglik_nb(dataset, log_pi, log_A, rates, fano):
    if fano <= 1.0001:
        return model_loglik(dataset, log_pi, log_A, rates)
    counts, lengths, glnc, Tmax, n = dataset
    mu = np.clip(rates, RATE_FLOOR, None) * DT
    r = mu / (fano - 1.0)
    A = np.exp(log_A)
    c = counts[:, :, None]
    emis = (gammaln(c + r[None, None, :]) - gammaln(r)[None, None, :] - glnc[:, :, None]
            - r[None, None, :] * np.log(fano) + c * np.log((fano - 1.0) / fano))
    alpha = log_pi[None, :] + emis[:, 0, :]
    ll = np.empty(n)
    sel = lengths == 1
    if sel.any():
        m = alpha[sel].max(axis=1, keepdims=True)
        ll[sel] = (m[:, 0] + np.log(np.exp(alpha[sel] - m).sum(axis=1)))
    for t in range(1, Tmax):
        mx = alpha.max(axis=1, keepdims=True)
        alpha = np.log(np.exp(alpha - mx) @ A + 1e-300) + mx + emis[:, t, :]
        sel = lengths == t + 1
        if sel.any():
            m = alpha[sel].max(axis=1, keepdims=True)
            ll[sel] = (m[:, 0] + np.log(np.exp(alpha[sel] - m).sum(axis=1)))
    return float(ll.sum())


def _data_rates(dataset):
    counts, lengths, _, _, n = dataset
    early = late = ec = lc = 0.0
    for i in range(n):
        L = int(lengths[i]); ne = max(1, L // 5)
        early += counts[i, :ne].sum(); ec += ne
        late += counts[i, L - ne:L].sum(); lc += ne
    return (max(early / max(ec, 1) / DT, RATE_FLOOR),
            max(late / max(lc, 1) / DT, RATE_FLOOR), float(np.mean(lengths)) * DT)


def fit_step_nb(dataset):
    r0, r1, _ = _data_rates(dataset)
    x0 = np.array([np.log(r0), np.log(max(r1, r0)), np.log(0.05), np.log(0.5)])

    def negll(u):
        p = 1 / (1 + np.exp(-u[2])); fano = 1 + np.exp(u[3])
        return -model_loglik_nb(dataset, *step_hmm((np.exp(u[0]), np.exp(u[1]), p)), fano)
    r = minimize(negll, x0, method="L-BFGS-B",
                 bounds=[(-4.6, 6.2), (-4.6, 6.2), (-12, 12), (-6, 2.0)],
                 options=dict(maxiter=40, ftol=1e-6)).x
    return (np.exp(r[0]), np.exp(r[1]), 1 / (1 + np.exp(-r[2]))), 1 + np.exp(r[3])


def fit_ramp_nb(dataset, K):
    r0, r1, mt = _data_rates(dataset)
    x0 = np.array([np.log(1 / max(mt, DT)), np.log(0.5), np.log(r0),
                   np.log(max(r1, r0)), np.log(0.5)])

    def negll(u):
        fano = 1 + np.exp(u[4])
        return -model_loglik_nb(
            dataset, *ramp_hmm((np.exp(u[0]), np.exp(u[1]), np.exp(u[2]),
                                np.exp(u[3])), K), fano)
    r = minimize(negll, x0, method="L-BFGS-B",
                 bounds=[(-4.6, 3.9), (-4.6, 2.3), (-4.6, 6.2), (-4.6, 6.2), (-6, 2.0)],
                 options=dict(maxiter=40, ftol=1e-6)).x
    return (np.exp(r[0]), np.exp(r[1]), np.exp(r[2]), np.exp(r[3])), 1 + np.exp(r[4])


# --- recovery units ----------------------------------------------------------
def _cv(counts, N, rng, K, fit):
    idx = rng.permutation(N); nt = max(1, int(round(HOLDOUT * N)))
    te, tr = idx[:nt], idx[nt:]
    ds_tr = build_dataset([counts[i] for i in tr])
    ds_te = build_dataset([counts[i] for i in te])
    try:
        if fit == "poisson":
            ps, _ = fit_step(ds_tr); s_cv = model_loglik(ds_te, *step_hmm(ps))
            pr, _ = fit_ramp(ds_tr, K); r_cv = model_loglik(ds_te, *ramp_hmm(pr, K))
        else:
            ps, fs = fit_step_nb(ds_tr); s_cv = model_loglik_nb(ds_te, *step_hmm(ps), fs)
            pr, fr_ = fit_ramp_nb(ds_tr, K); r_cv = model_loglik_nb(ds_te, *ramp_hmm(pr, K), fr_)
    except Exception:
        return NEG, NEG
    return s_cv, r_cv


def recover_canonical(true_gen, N, fr, seed, windows, K, fano, fit="poisson"):
    lower_priority()
    rng = np.random.default_rng(seed)
    win = rng.choice(windows, size=N)
    Tbins = np.maximum(2, np.round(win / DT).astype(int))
    counts = sim_canonical(true_gen, rng, Tbins, fr, fano)
    s_cv, r_cv = _cv(counts, N, rng, K, fit)
    pred = "step" if s_cv > r_cv else "ramp"
    return dict(true=true_gen, N=int(N), fr=float(fr), fano=float(fano), fit=fit,
                correct=int(pred == true_gen))


def recover_stepiness(s, N, fr, seed, windows, K):
    lower_priority()
    rng = np.random.default_rng(seed)
    win = rng.choice(windows, size=N)
    Tbins = np.maximum(2, np.round(win / DT).astype(int))
    counts = sim_stepiness(rng, Tbins, fr, s)
    s_cv, r_cv = _cv(counts, N, rng, K, "poisson")
    return dict(stepiness=float(s), N=int(N), fr=float(fr),
                step_won=int(s_cv > r_cv))


# --- sweeps ------------------------------------------------------------------
def run_overdisp(windows):
    FRs, Ns = [10, 20, 40], [40, 80, 160, 320]
    FANOS, GENS, R = [1.0, 1.5, 2.0, 3.0], ["step", "ramp"], 30
    jobs, sid = [], 0
    for fr in FRs:
        for N in Ns:
            for g in GENS:
                for r in range(R):
                    seed = 20000 + sid; sid += 1
                    for fano in FANOS:               # PAIRED across fano (same seed)
                        jobs.append((g, N, fr, seed, fano))
    print(f"Overdispersion sweep: {len(jobs)} sims (Poisson fit), {N_JOBS} workers")
    t = time.time()
    res = Parallel(n_jobs=N_JOBS, verbose=2)(
        delayed(recover_canonical)(g, N, fr, sd, windows, K_GRID, fano)
        for (g, N, fr, sd, fano) in jobs)
    print(f"  done in {time.time()-t:.0f}s")
    df = pd.DataFrame(res)
    df.to_csv(OVERDISP_CSV, index=False)
    return df


def run_stepiness(windows):
    svals = [0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
    conds, R = [(20, 160), (20, 320)], 40
    jobs, sid = [], 0
    for fr, N in conds:
        for s in svals:
            for r in range(R):
                jobs.append((s, N, fr, 40000 + sid)); sid += 1
    print(f"Stepiness sweep: {len(jobs)} sims, {N_JOBS} workers")
    res = Parallel(n_jobs=N_JOBS, verbose=2)(
        delayed(recover_stepiness)(s, N, fr, sd, windows, K_GRID)
        for (s, N, fr, sd) in jobs)
    df = pd.DataFrame(res)
    df.to_csv(STEEP_CSV, index=False)
    return df


def run_matched(windows):
    fr, Ns, FANOS, GENS, R = 20, [160, 320], [1.0, 1.5, 2.0, 3.0], ["step", "ramp"], 20
    jobs, sid = [], 0
    for N in Ns:
        for g in GENS:
            for r in range(R):
                seed = 60000 + sid; sid += 1
                for fano in FANOS:
                    for fit in ("poisson", "nb"):
                        jobs.append((g, N, fr, seed, fano, fit))
    print(f"Matched-NB sweep: {len(jobs)} sims, {N_JOBS} workers")
    res = Parallel(n_jobs=N_JOBS, verbose=2)(
        delayed(recover_canonical)(g, N, fr, sd, windows, K_GRID, fano, fit)
        for (g, N, fr, sd, fano, fit) in jobs)
    df = pd.DataFrame(res)
    df.to_csv(MATCHED_CSV, index=False)
    return df


# --- figure + verdict --------------------------------------------------------
def make_figure(od, st, mt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    FANOS = sorted(od["fano"].unique())
    # A/B: overdispersion recovery vs N, per fano, at FR=20 and FR=40
    for ax, fr in zip(axes[0], (20, 40)):
        sub = od[od["fr"] == fr]
        for fano in FANOS:
            g = sub[sub["fano"] == fano].groupby("N")["correct"].mean()
            ax.plot(g.index, g.values * 100, "o-", label=f"Fano {fano:g}")
        ax.axhline(80, color="gray", ls="--", lw=1)
        ax.set_xscale("log"); ax.set_xticks(sorted(sub["N"].unique()))
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("trials / neuron (N)"); ax.set_ylabel("recovery (%)")
        ax.set_title(f"Overdispersion @ FR={fr} Hz (Poisson fit)", fontsize=10, loc="left")
        ax.legend(fontsize=8); ax.set_ylim(45, 100)
    # C: stepiness separation curve
    ax = axes[1, 0]
    for (fr, N), gg in st.groupby(["fr", "N"]):
        s = gg.groupby("stepiness")["step_won"].mean()
        ax.plot(s.index, s.values * 100, "o-", label=f"N={N}")
    ax.axhline(50, color="gray", ls="--", lw=1)
    ax.set_xlabel("stepiness  (0 = clean ramp, 1 = clean step)")
    ax.set_ylabel("P(step model wins) (%)")
    ax.set_title("Intermediate generators: discrimination vs stepiness",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8)
    # D: matched NB vs Poisson fit, recovery vs fano (pooled N, gens)
    ax = axes[1, 1]
    for fit, col in (("poisson", "#c44"), ("nb", "#27a")):
        g = mt[mt["fit"] == fit].groupby("fano")["correct"].mean()
        ax.plot(g.index, g.values * 100, "o-", color=col, label=f"{fit} fit")
    ax.axhline(80, color="gray", ls="--", lw=1)
    ax.set_xlabel("Fano factor"); ax.set_ylabel("recovery (%)")
    ax.set_title("Mis-specified (Poisson) vs matched (NB) fit @ FR=20",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8)
    fig.suptitle("Phase 1 recovery map HARDENED: overdispersion + intermediate "
                 "generators", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nSaved figure -> {FIG_PATH}")


def verdict(od, st, mt):
    print("\n========== KEY DELIVERABLE: how far does the boundary move? ==========")
    piv = od.groupby(["fano", "fr", "N"])["correct"].mean()

    def min_N_for_80(fano, fr):
        s = piv.loc[fano, fr]
        ok = s[s >= 0.80]
        return int(ok.index.min()) if len(ok) else None
    print("  minimum N for >=80% recovery (Poisson fit), by Fano x FR:")
    for fr in (20, 40):
        row = "  FR=%2d Hz: " % fr + ", ".join(
            f"Fano{f:g}->N{('>=' + str(min_N_for_80(f, fr))) if min_N_for_80(f, fr) else '>320(none)'}"
            for f in sorted(od["fano"].unique()))
        print(row)
    # stepiness separation width
    print("\n  intermediate generators: P(step wins) crosses 0.5 near s~0.5; "
          "discrimination needs stepiness away from 0.5:")
    for (fr, N), gg in st.groupby(["fr", "N"]):
        s = gg.groupby("stepiness")["step_won"].mean()
        lo = s[s <= 0.2].index.max() if (s <= 0.2).any() else None
        hi = s[s >= 0.8].index.min() if (s >= 0.8).any() else None
        print(f"    N={N}: P(step)<=20% for s<={lo}, >=80% for s>={hi} "
              f"-> ambiguous band s in ({lo}, {hi})")
    # matched vs poisson
    mp = mt.groupby(["fit", "fano"])["correct"].mean().unstack(0)
    print("\n  matched-NB vs Poisson fit recovery (FR=20, pooled N): ")
    for fano in mp.index:
        print(f"    Fano {fano:g}: poisson {mp.loc[fano,'poisson']*100:.0f}% vs "
              f"nb {mp.loc[fano,'nb']*100:.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["overdisp", "stepiness", "matched", "all",
                                        "report"], default="all")
    args = ap.parse_args()
    lower_priority()   # parent; each loky worker also lowers itself on first call
    print(f"process priority -> BelowNormal; capped at {N_JOBS} workers, no network")

    windows = load_windows()
    print(f"window pool: {len(windows)} RTs (median {np.median(windows)*1000:.0f} ms)\n")

    if args.stage == "report":
        od = pd.read_csv(OVERDISP_CSV); st = pd.read_csv(STEEP_CSV)
        mt = pd.read_csv(MATCHED_CSV); make_figure(od, st, mt); verdict(od, st, mt)
        return
    od = run_overdisp(windows) if args.stage in ("overdisp", "all") else None
    st = run_stepiness(windows) if args.stage in ("stepiness", "all") else None
    mt = run_matched(windows) if args.stage in ("matched", "all") else None
    if args.stage == "all":
        make_figure(od, st, mt); verdict(od, st, mt)


if __name__ == "__main__":
    main()

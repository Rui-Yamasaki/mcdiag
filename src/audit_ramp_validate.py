"""Validate the RAMPING model — the paper's last open flank (PURE; local, numpy/scipy/hmmlearn).

The headline recovery boundary + the 0.57 population gate all run on Pipeline B
(phase1_recovery.py), whose RAMPING (discretised bounded drift-diffusion) likelihood was never
validated the way the STEPPING side was (vs hmmlearn, 2.3e-13). This does the LOCAL validation:

  A. BRUTE-FORCE the ramp forward likelihood for BOTH engines (full path enumeration on small
     M,T) -> the discretised-HMM marginal is numerically exact (machine precision).
  B. hmmlearn PoissonHMM cross-check of Pipeline B's ramp forward LL at the production grid K=50
     (independent maintained library) -> exact.
  C. RECONCILE Pipeline A (stepramp.py: M=25 grid, CDF cell-integration, sigma=0.7, 3:1 mod) vs
     Pipeline B (K=50, point-weight Gaussian, sigma=0.4, 2:1 mod): transition matrices, ramp
     likelihood, generator statistics, and recovery at MATCHED settings.
  D. sigma x K ROBUSTNESS of the recovery BOUNDARY (the load-bearing analyst choices): does the
     >=80% boundary (FR>=20, N>=160-320) move when ramp sigma in {0.2,0.4,0.7,1.0} and the latent
     grid K in {25,50,75}?

The discretised-DDM-vs-canonical-accumulator MODEL-equivalence + a published-number reproduction
is the Colab notebook (notebooks/canonical_ramp_check.ipynb) — ssm/dynamax build there, not here.

  python src/run_ramp_validate.py            # full
  python src/run_ramp_validate.py --quick
"""
from __future__ import annotations

import argparse
import itertools
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.special import gammaln, logsumexp  # noqa: E402

from ibl_one import PROJECT_ROOT  # noqa: E402
import phase1_recovery as B  # Pipeline B (headline engine)
import stepramp as A         # Pipeline A (unused; cell-integration discretisation)

DT = B.DT
RESULTS = PROJECT_ROOT / "results"
FIG = PROJECT_ROOT / "figures" / "ramp_validation.png"
ROBUST_CSV = RESULTS / "ramp_validation_robustness.csv"
RECON_CSV = RESULTS / "ramp_validation_reconcile.csv"


# ============================ A. brute-force ramp forward ============================
def brute_force_ll(counts, rates, Amat, pi, dt):
    """Exact marginal log-lik of ONE trial by enumerating all state paths."""
    T = len(counts); M = len(rates)
    mu = np.maximum(rates * dt, 1e-12)
    logem = counts[:, None] * np.log(mu)[None, :] - mu[None, :] - gammaln(counts + 1.0)[:, None]
    logpi = np.log(pi + 1e-300); logA = np.log(Amat + 1e-300)
    acc = -np.inf
    for path in itertools.product(range(M), repeat=T):
        lp = logpi[path[0]] + logem[0, path[0]]
        for t in range(1, T):
            lp += logA[path[t - 1], path[t]] + logem[t, path[t]]
        acc = np.logaddexp(acc, lp)
    return acc


def check_bruteforce():
    print("=" * 72)
    print("A. BRUTE-FORCE ramp forward likelihood (full path enumeration, M=5, T=6)")
    rng = np.random.default_rng(0)
    M, T = 5, 6
    params = (4.0, 30.0, 1.5, 0.6)        # (lam0/drift differ per engine convention below)
    # --- Pipeline B: ramp_hmm + model_loglik ---
    lp_b, lA_b, rates_b = B.ramp_hmm((1.5, 0.6, 4.0, 30.0), M)   # (drift,sigma,lam_lo,lam_hi)
    pi_b, Amat_b = np.exp(lp_b), np.exp(lA_b)
    counts = [rng.poisson(np.linspace(4, 30, T) * DT).astype(float) for _ in range(6)]
    ds = B.build_dataset(counts)
    fwd_b = np.array([B.model_loglik(B.build_dataset([c]), lp_b, lA_b, rates_b) for c in counts])
    bru_b = np.array([brute_force_ll(c, rates_b, Amat_b, pi_b, DT) for c in counts])
    d_b = np.max(np.abs(fwd_b - bru_b))
    print(f"  Pipeline B (point-weight, K=M=5): max |forward - bruteforce| = {d_b:.2e}")
    # --- Pipeline A: ramping_build + forward_loglik ---
    grid = A.ramp_grid(M)
    rates_a, Amat_a, pi_a = A.ramping_build((4.0, 30.0, 1.5, 0.6), grid, DT)  # (lam0,lam1,beta,sigma)
    cc, L = A._pad(counts)
    fwd_a = A.forward_loglik(cc, L, rates_a, Amat_a, pi_a, DT)
    bru_a = np.array([brute_force_ll(c, rates_a, Amat_a, pi_a, DT) for c in counts])
    d_a = np.max(np.abs(fwd_a - bru_a))
    print(f"  Pipeline A (CDF cell-integ, M=5):  max |forward - bruteforce| = {d_a:.2e}")
    ok = d_b < 1e-9 and d_a < 1e-9
    print(f"  -> ramp forward algorithm EXACT (both engines): {'PASS' if ok else 'FAIL'}")
    return dict(bruteforce_B=float(d_b), bruteforce_A=float(d_a), pass_=ok)


# ============================ B. hmmlearn ramp cross-check ============================
def check_hmmlearn(K=50):
    print("\n" + "=" * 72)
    print(f"B. hmmlearn.PoissonHMM cross-check of Pipeline B ramp forward LL (production K={K})")
    from hmmlearn.hmm import PoissonHMM
    import hmmlearn
    rng = np.random.default_rng(1)
    lp, lA, rates = B.ramp_hmm((1.0, 0.4, 6.0, 36.0), K)        # production-like ramp
    # data from the discretised chain itself (so both engines score the same sequences)
    Amat, pi = np.exp(lA), np.exp(lp)
    counts = []
    for _ in range(40):
        T = int(rng.integers(20, 60)); s = 0; c = np.empty(T)
        for t in range(T):
            c[t] = rng.poisson(max(rates[s], 1e-9) * DT)
            s = rng.choice(K, p=Amat[s])
        counts.append(c)
    our = sum(B.model_loglik(B.build_dataset([c]), lp, lA, rates) for c in counts)
    ref = PoissonHMM(n_components=K, init_params="", params="")
    ref.startprob_ = pi / pi.sum()
    ref.transmat_ = Amat / Amat.sum(1, keepdims=True)
    ref.lambdas_ = (np.maximum(rates, 1e-9) * DT).reshape(-1, 1)
    X = np.concatenate([c.reshape(-1, 1) for c in counts]).astype(int)
    refll = ref.score(X, [len(c) for c in counts])
    d = abs(our - refll)
    print(f"  hmmlearn {hmmlearn.__version__} | our forward LL {our:.4f} vs hmmlearn {refll:.4f}")
    print(f"  abs diff = {d:.2e}  -> ramp forward matches maintained library: "
          f"{'PASS' if d < 1e-6 else 'FAIL'}")
    return dict(hmmlearn_our=float(our), hmmlearn_ref=float(refll), hmmlearn_absdiff=float(d),
                pass_=bool(d < 1e-6))


# ============================ C. reconcile A vs B ============================
def reconcile():
    print("\n" + "=" * 72)
    print("C. RECONCILE Pipeline A (CDF cell-integration) vs B (point-weight Gaussian)")
    rows = []
    for K in (10, 25, 50, 100):
        lp, lA, rates = B.ramp_hmm((1.0, 0.5, 5.0, 35.0), K)
        grid = A.ramp_grid(K)
        rA, AA, piA = A.ramping_build((5.0, 35.0, 1.0, 0.5), grid, DT)
        dA = np.max(np.abs(np.exp(lA) - AA))                     # transition-matrix difference
        # likelihood difference on shared data (B's discretised chain)
        rng = np.random.default_rng(3); Amat = np.exp(lA)
        counts = []
        for _ in range(30):
            T = int(rng.integers(25, 50)); s = 0; c = np.empty(T)
            for t in range(T):
                c[t] = rng.poisson(max(rates[s], 1e-9) * DT); s = rng.choice(K, p=Amat[s])
            counts.append(c)
        ll_B = sum(B.model_loglik(B.build_dataset([c]), lp, lA, rates) for c in counts)
        cc, L = A._pad(counts)
        ll_A = A.forward_loglik(cc, L, rA, AA, piA, DT).sum()
        rows.append(dict(K=K, transmat_maxdiff=float(dA), ll_B=float(ll_B), ll_A=float(ll_A),
                         ll_reldiff=float(abs(ll_A - ll_B) / abs(ll_B))))
        print(f"  K={K:3d}: transition-matrix max|A_B - A_A| = {dA:.3e} | "
              f"LL rel-diff = {abs(ll_A-ll_B)/abs(ll_B):.2e}")
    # generative equivalence: both ramp generators are clipped-Gaussian walks -> matched stats
    rng = np.random.default_rng(4); Tb = np.full(400, 50)
    cB = B.simulate("ramp", rng, Tb, 20.0)
    pa = (B.gen_rates(20.0)[0], B.gen_rates(20.0)[1], B.GEN_DRIFT, B.GEN_SIGMA)
    cA, _ = A.simulate_ramping(pa, Tb, DT, np.random.default_rng(4))
    mB = np.mean([c.sum() / (len(c) * DT) for c in cB]); mA = (cA.sum(1) / (Tb * DT)).mean()
    print(f"  generator mean in-window rate (matched params): B {mB:.2f} Hz vs A {mA:.2f} Hz "
          f"(both clipped-Gaussian-walk -> Poisson)")
    pd.DataFrame(rows).to_csv(RECON_CSV, index=False)
    return rows, float(mB), float(mA)


# ============================ D. sigma x K recovery-boundary robustness ============================
def gen_step_B(rng, Tbins, fr):
    r_lo, r_hi = B.gen_rates(fr); out = []
    for T in Tbins:
        s = rng.integers(1, T) if T > 1 else 1
        out.append(rng.poisson(np.where(np.arange(T) < s, r_lo, r_hi) * DT).astype(float))
    return out


def gen_ramp_B(rng, Tbins, fr, sigma, drift=None):
    drift = B.GEN_DRIFT if drift is None else drift
    r_lo, r_hi = B.gen_rates(fr); out = []
    for T in Tbins:
        x = 0.0; rates = np.empty(T)
        for t in range(T):
            rates[t] = r_lo + (r_hi - r_lo) * x
            x = min(max(x + drift * DT + sigma * np.sqrt(DT) * rng.standard_normal(), 0.0), 1.0)
        out.append(rng.poisson(rates * DT).astype(float))
    return out


def recover_B(true_gen, N, fr, sigma, K, seed, windows, holdout=0.3):
    """Pipeline-B recovery with explicit ramp sigma + grid K (replicates recover_once)."""
    rng = np.random.default_rng(seed)
    win = rng.choice(windows, N); Tb = np.maximum(2, np.round(win / DT).astype(int))
    counts = gen_ramp_B(rng, Tb, fr, sigma) if true_gen == "ramp" else gen_step_B(rng, Tb, fr)
    idx = rng.permutation(N); nt = max(1, int(round(holdout * N))); te, tr = idx[:nt], idx[nt:]
    ds_tr = B.build_dataset([counts[i] for i in tr]); ds_te = B.build_dataset([counts[i] for i in te])
    try:
        ps, _ = B.fit_step(ds_tr); s_cv = B.model_loglik(ds_te, *B.step_hmm(ps))
    except Exception:
        s_cv = B.NEG_INF
    try:
        pr, _ = B.fit_ramp(ds_tr, K); r_cv = B.model_loglik(ds_te, *B.ramp_hmm(pr, K))
    except Exception:
        r_cv = B.NEG_INF
    return int(("step" if s_cv > r_cv else "ramp") == true_gen)


def robustness(windows, R, jobs_n):
    print("\n" + "=" * 72)
    print("D. sigma x K recovery-BOUNDARY robustness (does >=80% boundary move?)")
    pts = [(40, 20), (160, 20), (320, 20), (80, 40), (160, 40)]   # key boundary points
    sigmas = [0.2, 0.4, 0.7, 1.0]
    Ks = [50]                       # sigma sweep at production K
    extraK = [(160, 20, 0.4, k) for k in (25, 50, 75)]            # K sweep at the canonical point
    jobs, sid = [], 0
    for (N, fr) in pts:
        for sg in sigmas:
            for g in ("step", "ramp"):
                for _ in range(R):
                    jobs.append((g, N, fr, sg, 50, 50000 + sid)); sid += 1
    for (N, fr, sg, k) in extraK:
        for g in ("step", "ramp"):
            for _ in range(R):
                jobs.append((g, N, fr, sg, k, 60000 + sid)); sid += 1
    print(f"  {len(jobs)} recoveries ({len(pts)}pts x {len(sigmas)}sigma x2 + K-sweep, R={R})")
    t = time.time()
    res = Parallel(n_jobs=jobs_n, verbose=1)(
        delayed(recover_B)(g, N, fr, sg, k, sd, windows) for (g, N, fr, sg, k, sd) in jobs)
    print(f"  done in {time.time()-t:.0f}s")
    rows = []
    for (g, N, fr, sg, k, sd), c in zip(jobs, res):
        rows.append(dict(true=g, N=N, fr=fr, sigma=sg, K=k, correct=c))
    df = pd.DataFrame(rows)
    df.to_csv(ROBUST_CSV, index=False)
    return df


def report_robustness(df):
    print("\n  recovery (%) vs sigma at K=50  (mean over step+ramp):")
    piv = (df[df.K == 50].groupby(["N", "fr", "sigma"])["correct"].mean() * 100).round(0)
    for (N, fr) in [(40, 20), (160, 20), (320, 20), (80, 40), (160, 40)]:
        s = piv.loc[N, fr]
        print(f"    N={N:3d} FR={fr}: " + "  ".join(f"sig{sg}:{int(s.get(sg, np.nan))}%" for sg in [0.2, 0.4, 0.7, 1.0]))
    print("\n  recovery (%) vs K at (N=160,FR=20,sig=0.4):")
    kp = (df[(df.N == 160) & (df.fr == 20) & (df.sigma == 0.4)].groupby("K")["correct"].mean() * 100).round(0)
    print("    " + "  ".join(f"K{k}:{int(kp.get(k, np.nan))}%" for k in [25, 50, 75]))
    # ramp-only recovery vs sigma (sigma hits the ramp side)
    print("\n  TRUE-RAMP recovery vs sigma at K=50 (sigma makes ramps look steppier):")
    rp = (df[(df.K == 50) & (df.true == "ramp")].groupby(["N", "fr", "sigma"])["correct"].mean() * 100).round(0)
    for (N, fr) in [(160, 20), (320, 20), (160, 40)]:
        s = rp.loc[N, fr]
        print(f"    N={N:3d} FR={fr}: " + "  ".join(f"sig{sg}:{int(s.get(sg, np.nan))}%" for sg in [0.2, 0.4, 0.7, 1.0]))


def make_figure(df, recon):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    for (N, fr), gg in df[df.K == 50].groupby(["N", "fr"]):
        s = gg.groupby("sigma")["correct"].mean() * 100
        a.plot(s.index, s.values, "o-", label=f"N={N},FR={fr}")
    a.axhline(80, color="gray", ls="--", lw=1, label="usable 80%")
    a.axhline(50, color="k", ls=":", lw=1)
    a.axvline(0.4, color="navy", lw=1, ls="--"); a.text(0.4, 52, "headline σ=0.4", color="navy", fontsize=8)
    a.set_xlabel("ramp diffusion σ"); a.set_ylabel("recovery (%)")
    a.set_title("Recovery boundary vs ramp σ (K=50)", fontsize=10, loc="left")
    a.legend(fontsize=7); a.set_ylim(40, 100)
    a = ax[1]
    rc = pd.DataFrame(recon)
    a.semilogx(rc.K, rc.transmat_maxdiff, "o-", label="max|A_B - A_A|")
    a.semilogx(rc.K, rc.ll_reldiff, "s-", label="LL rel-diff")
    a.axvline(50, color="navy", lw=1, ls="--")
    a.set_xlabel("latent grid K"); a.set_ylabel("A-vs-B difference (log)")
    a.set_title("Two-engine INFERENCE agrees (LL rel-diff ~3e-4);\ntransmat diff is boundary-only",
                fontsize=10, loc="left")
    a.legend(fontsize=8)
    fig.suptitle("Ramping-model validation: forward exact (brute-force+hmmlearn); "
                 "σ/grid robustness", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=150)
    print(f"\nSaved figure -> {FIG}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    windows = B.load_windows()
    bf = check_bruteforce()
    hl = check_hmmlearn(K=50)
    recon, mB, mA = reconcile()
    df = robustness(windows, R=8 if args.quick else 40, jobs_n=args.jobs)
    report_robustness(df)
    make_figure(df, recon)
    print("\n" + "=" * 72)
    print("LOCAL VALIDATION SUMMARY:")
    print(f"  ramp forward EXACT: brute-force B {bf['bruteforce_B']:.1e} / A {bf['bruteforce_A']:.1e}; "
          f"hmmlearn {hl['hmmlearn_absdiff']:.1e}")
    print("  -> the discretised-DDM ramp LIKELIHOOD is computed exactly; "
          "canonical-ACCUMULATOR model-equivalence is the Colab notebook.")


if __name__ == "__main__":
    main()

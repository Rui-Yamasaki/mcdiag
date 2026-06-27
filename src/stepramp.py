"""Self-contained stepping-vs-ramping latent-dynamics models for spike trains.

Both families are expressed as **Poisson Hidden Markov Models** and scored with
the **same exact forward-algorithm** marginal likelihood and the **same held-out
cross-validated log-likelihood** metric. This is deliberate: the comparison must
have no structural bias toward either family (only the latent TRANSITION
structure differs; emissions, binning, inference, and the CV metric are
identical).

  Stepping  : 2-state left-to-right Poisson HMM. Latent starts in a baseline
              state (rate lam0) and jumps ONCE to a committed state (rate lam1)
              with per-bin probability p (geometric step time); committed state
              is absorbing. Params: (lam0, lam1, p).  -> a discrete jump.

  Ramping   : discretized diffusion-to-bound. A 1-D latent x in [0,1] performs a
              Gaussian random walk with drift beta and diffusion sigma, absorbing
              at x=1, discretized onto an M-point grid -> an M-state Poisson HMM.
              Per-state rate = lam0 + (lam1-lam0)*x. Params: (lam0, lam1, beta,
              sigma).  -> a gradual continuous-looking climb.

Generators (the "truth") simulate the genuine continuous processes; the fits use
the HMM/discretized forms. Poisson spiking only (non-Poisson is a later step).

This is the project's own implementation (no maintained Python package ships a
turnkey, cleanly-installable step-vs-ramp comparison on this Win/py3.13/numpy2
box). It is validated against a maintained reference PoissonHMM and by grid
convergence in src/phase2_validate.py.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

DT = 0.025  # default bin width (s); 25 ms, matching the PSTH binning


# ============================================================================
# Shared exact inference: scaled forward algorithm, vectorized over trials
# ============================================================================
def forward_loglik(counts, lengths, rates, A, pi, dt):
    """Per-trial exact marginal log-likelihood of a Poisson HMM.

    counts  : (Ntr, Tmax) int spike counts, zero-padded past each trial length
    lengths : (Ntr,) number of valid bins per trial
    rates   : (K,) per-state firing rate (Hz)
    A       : (K, K) row-stochastic transition matrix
    pi      : (K,) initial state distribution
    Returns : (Ntr,) marginal log-likelihood per trial.

    Padded bins use emission factor 1 and a stochastic A, so they contribute
    exactly 0 to the log-likelihood (probability mass is conserved). The
    data-only gammaln(y+1) term is factored out of the scaled recursion and
    added back at the end.
    """
    counts = np.asarray(counts)
    Ntr, Tmax = counts.shape
    mu = np.maximum(rates * dt, 1e-12)          # (K,) expected counts per bin
    logmu = np.log(mu)
    t_idx = np.arange(Tmax)
    valid = t_idx[None, :] < lengths[:, None]   # (Ntr, Tmax)

    alpha = np.repeat(pi[None, :], Ntr, axis=0)  # (Ntr, K)
    ll = np.zeros(Ntr)
    for t in range(Tmax):
        y = counts[:, t]
        # emission factor B[n,s] = exp(y*log mu_s - mu_s); =1 on padded bins
        B = np.exp(y[:, None] * logmu[None, :] - mu[None, :])
        vmask = valid[:, t]
        if not vmask.all():
            B = np.where(vmask[:, None], B, 1.0)
        araw = (alpha @ A) * B if t > 0 else alpha * B
        c = np.maximum(araw.sum(axis=1), 1e-300)
        alpha = araw / c[:, None]
        ll += np.where(vmask, np.log(c), 0.0)

    gl = gammaln(counts + 1.0)
    gl[~valid] = 0.0
    return ll - gl.sum(axis=1)


# ============================================================================
# Model builders: (params) -> (rates, A, pi)
# ============================================================================
def stepping_build(params):
    lam0, lam1, p = params
    rates = np.array([lam0, lam1], float)
    A = np.array([[1.0 - p, p], [0.0, 1.0]])
    pi = np.array([1.0, 0.0])
    return rates, A, pi


def ramp_grid(M):
    return np.linspace(0.0, 1.0, M)


def ramping_build(params, grid, dt):
    lam0, lam1, beta, sigma = params
    M = grid.size
    rates = lam0 + (lam1 - lam0) * grid
    mean = grid + beta * dt                       # drift per bin
    sd = max(sigma * np.sqrt(dt), 1e-6)
    # cell edges at midpoints; outer edges at +-inf so tail mass clamps to ends
    mids = (grid[:-1] + grid[1:]) / 2.0
    edges = np.concatenate([[-np.inf], mids, [np.inf]])  # (M+1,)
    z = (edges[None, :] - mean[:, None]) / sd            # (M, M+1)
    cdf = norm.cdf(z)
    A = cdf[:, 1:] - cdf[:, :-1]                          # (M, M)
    A = np.maximum(A, 0.0)
    A /= A.sum(axis=1, keepdims=True)
    A[-1, :] = 0.0
    A[-1, -1] = 1.0                                       # absorbing bound at x=1
    pi = np.zeros(M)
    pi[0] = 1.0                                           # start at baseline x=0
    return rates, A, pi


def build(model, params, dt, grid=None):
    if model == "step":
        return stepping_build(params)
    return ramping_build(params, grid, dt)


# ============================================================================
# Generators (the "truth"): genuine continuous processes, vectorized over trials
# ============================================================================
def _pad(counts_list):
    lengths = np.array([len(c) for c in counts_list])
    Tmax = int(lengths.max())
    out = np.zeros((len(counts_list), Tmax), dtype=int)
    for i, c in enumerate(counts_list):
        out[i, : len(c)] = c
    return out, lengths


def simulate_stepping(params, lengths, dt, rng):
    """Genuine single-step process: rate jumps lam0->lam1 at a geometric time."""
    lam0, lam1, p = params
    lengths = np.asarray(lengths)
    N, Tmax = len(lengths), int(lengths.max())
    counts = np.zeros((N, Tmax), dtype=int)
    stepped = np.zeros(N, dtype=bool)
    for t in range(Tmax):
        jump = (~stepped) & (rng.random(N) < p)
        stepped |= jump
        rate = np.where(stepped, lam1, lam0)
        counts[:, t] = rng.poisson(rate * dt)
    for i, L in enumerate(lengths):
        counts[i, L:] = 0
    return counts, lengths


def simulate_ramping(params, lengths, dt, rng):
    """Genuine diffusion-to-bound: x random-walks with drift, absorbs at 1."""
    lam0, lam1, beta, sigma = params
    lengths = np.asarray(lengths)
    N, Tmax = len(lengths), int(lengths.max())
    counts = np.zeros((N, Tmax), dtype=int)
    x = np.zeros(N)
    sq = np.sqrt(dt)
    for t in range(Tmax):
        rate = lam0 + (lam1 - lam0) * np.clip(x, 0.0, 1.0)
        counts[:, t] = rng.poisson(rate * dt)
        x = x + beta * dt + sigma * sq * rng.standard_normal(N)
        x = np.clip(x, 0.0, 1.0)                  # reflect at 0, absorb at 1
    for i, L in enumerate(lengths):
        counts[i, L:] = 0
    return counts, lengths


def simulate(model, params, lengths, dt, rng):
    if model == "step":
        return simulate_stepping(params, lengths, dt, rng)
    return simulate_ramping(params, lengths, dt, rng)


# ============================================================================
# Fitting (MLE on the exact marginal likelihood) — same optimizer for both
# ============================================================================
def _data_rate_inits(counts, lengths, dt):
    """Baseline / committed rate guesses from early vs late bins."""
    N, Tmax = counts.shape
    valid = np.arange(Tmax)[None, :] < lengths[:, None]
    third = np.maximum(lengths // 3, 1)
    early = np.array([counts[i, : third[i]].mean() for i in range(N)]) / dt
    late = np.array([counts[i, lengths[i] - third[i]: lengths[i]].mean()
                     for i in range(N)]) / dt
    lam0 = max(np.nanmean(early), 0.5)
    lam1 = max(np.nanmean(late), 0.5)
    overall = max(counts[valid].mean() / dt, 0.5)
    return lam0, lam1, overall


def _pack(model, lam0, lam1, p_or_beta, sigma=None):
    if model == "step":
        p = min(max(p_or_beta, 1e-3), 0.999)
        return np.array([np.log(lam0), np.log(lam1), np.log(p / (1 - p))])
    return np.array([np.log(lam0), np.log(lam1), p_or_beta, np.log(max(sigma, 1e-3))])


def _unpack(model, theta):
    if model == "step":
        lam0, lam1 = np.exp(theta[0]), np.exp(theta[1])
        p = 1.0 / (1.0 + np.exp(-theta[2]))
        return (lam0, lam1, p)
    lam0, lam1 = np.exp(theta[0]), np.exp(theta[1])
    beta, sigma = theta[2], np.exp(theta[3])
    return (lam0, lam1, beta, sigma)


def fit(model, counts, lengths, dt, M=25, n_starts=2, rng=None, maxiter=80):
    """Fit a model by maximizing the exact marginal log-likelihood."""
    grid = ramp_grid(M) if model == "ramp" else None
    lam0, lam1, overall = _data_rate_inits(counts, lengths, dt)
    meanT = max(float(lengths.mean()), 2.0)

    def negll(theta):
        params = _unpack(model, theta)
        if model == "ramp" and (params[3] <= 0 or not np.isfinite(params[3])):
            return 1e12
        rates, A, pi = build(model, params, dt, grid)
        if not np.all(np.isfinite(rates)) or np.any(rates <= 0):
            return 1e12
        return -forward_loglik(counts, lengths, rates, A, pi, dt).sum()

    # informed starts (+ a perturbed restart)
    starts = []
    if model == "step":
        starts.append(_pack("step", lam0, lam1, 2.0 / meanT))
        starts.append(_pack("step", overall, overall * 1.5, 1.0 / meanT))
    else:
        beta0 = 1.0 / (0.5 * meanT * dt)              # cross [0,1] by mid-trial
        starts.append(_pack("ramp", lam0, lam1, beta0, 0.5))
        starts.append(_pack("ramp", overall, max(lam1, overall * 1.5),
                            beta0 * 0.5, 1.0))
    starts = starts[:max(1, n_starts)]

    best = None
    for s in starts:
        res = minimize(negll, s, method="L-BFGS-B",
                       options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    return _unpack(model, best.x), -best.fun


def eval_loglik(model, params, counts, lengths, dt, M=25):
    grid = ramp_grid(M) if model == "ramp" else None
    rates, A, pi = build(model, params, dt, grid)
    return forward_loglik(counts, lengths, rates, A, pi, dt).sum()


# ============================================================================
# The fair comparison: k-fold CV held-out log-likelihood, identical for both
# ============================================================================
def cv_compare(counts, lengths, dt, M=25, k=4, n_starts=2, rng=None):
    """Return (winner, cvll_step, cvll_ramp): both via SAME folds + SAME metric."""
    rng = rng or np.random.default_rng(0)
    N = len(lengths)
    idx = rng.permutation(N)
    folds = np.array_split(idx, k)
    cvll = {"step": 0.0, "ramp": 0.0}
    for f in range(k):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(k) if g != f])
        for model in ("step", "ramp"):
            params, _ = fit(model, counts[train], lengths[train], dt,
                            M=M, n_starts=n_starts, rng=rng)
            cvll[model] += eval_loglik(model, params, counts[test],
                                       lengths[test], dt, M=M)
    winner = "step" if cvll["step"] >= cvll["ramp"] else "ramp"
    return winner, cvll["step"], cvll["ramp"]


# ============================================================================
# Self-test: forward algorithm correctness + parameter recovery
# ============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("=== forward-algorithm vs brute-force enumeration (K=2, T=6) ===")
    rates = np.array([5.0, 30.0])
    A = np.array([[0.7, 0.3], [0.0, 1.0]])
    pi = np.array([1.0, 0.0])
    y = rng.poisson(np.array([5, 5, 5, 30, 30, 30]) * DT, size=(4, 6))
    lengths = np.full(4, 6)
    fwd = forward_loglik(y, lengths, rates, A, pi, DT)
    # brute force per trial
    from itertools import product
    mu = rates * DT
    bf = []
    for row in y:
        accum = -np.inf
        for path in product(range(2), repeat=6):
            lp = np.log(pi[path[0]])
            for t in range(6):
                lp += row[t] * np.log(mu[path[t]]) - mu[path[t]] - gammaln(row[t] + 1)
                if t > 0:
                    lp += np.log(A[path[t - 1], path[t]] + 1e-300)
            accum = np.logaddexp(accum, lp)
        bf.append(accum)
    bf = np.array(bf)
    print("  max |forward - bruteforce| =", np.max(np.abs(fwd - bf)))
    assert np.allclose(fwd, bf, atol=1e-8), "forward algorithm INCORRECT"
    print("  OK forward algorithm matches brute force.")

    print("\n=== parameter recovery: simulate stepping -> fit stepping ===")
    true = (4.0, 35.0, 0.05)
    counts, L = simulate_stepping(true, np.full(400, 60), DT, rng)
    est, ll = fit("step", counts, L, DT)
    print(f"  true (lam0,lam1,p)={true}")
    print(f"  est  (lam0,lam1,p)=({est[0]:.2f},{est[1]:.2f},{est[2]:.3f})  ll={ll:.1f}")

    print("\n=== parameter recovery: simulate ramping -> fit ramping ===")
    truer = (4.0, 35.0, 2.0, 0.8)
    counts, L = simulate_ramping(truer, np.full(400, 60), DT, rng)
    est, ll = fit("ramp", counts, L, DT, M=30)
    print(f"  true (lam0,lam1,beta,sigma)={truer}")
    print(f"  est  (lam0,lam1,beta,sigma)=({est[0]:.2f},{est[1]:.2f},"
          f"{est[2]:.2f},{est[3]:.2f})  ll={ll:.1f}")
    print("\nself-test done.")

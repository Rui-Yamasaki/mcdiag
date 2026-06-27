"""PHASE 0 — persist the RAMP-side forward-likelihood checks (closes trace flags R6/R7).

The stepping hmmlearn check (~2.27e-13) is already in results/phase1_recovery_validation.json.
The RAMP-side checks (brute-force enumeration + hmmlearn at production K=50) lived only in the
console / docs/ramp_validation.md. This persists them to results/ramp_validation_forward.json.

  python src/persist_ramp_validation.py
"""
from __future__ import annotations

import json

import numpy as np

import phase1_recovery as B
from audit_ramp_validate import brute_force_ll
from ibl_one import PROJECT_ROOT

OUT = PROJECT_ROOT / "results" / "ramp_validation_forward.json"
FIX = PROJECT_ROOT / "results" / "ramp_fixtures"
K_BF = 5            # brute-force latent grid (enumerable)
K_PROD = 50        # production ramp grid


def ramp_bruteforce_diff():
    """Canonical-engine ramp forward LL vs full path enumeration on a small enumerable case."""
    rng = np.random.default_rng(0)
    lp, lA, rates = B.ramp_hmm((1.5, 0.6, 4.0, 30.0), K_BF)   # (drift,sigma,lam_lo,lam_hi)
    pi, Amat = np.exp(lp), np.exp(lA)
    T = 6
    counts = [rng.poisson(np.linspace(4, 30, T) * B.DT).astype(float) for _ in range(6)]
    fwd = np.array([B.model_loglik(B.build_dataset([c]), lp, lA, rates) for c in counts])
    bru = np.array([brute_force_ll(c, rates, Amat, pi, B.DT) for c in counts])
    return float(np.max(np.abs(fwd - bru)))


def ramp_hmmlearn_diff_k50():
    """Canonical-engine ramp forward LL vs hmmlearn.PoissonHMM at production K=50, scoring the
    SAME spike trains from a results/ramp_fixtures/ ramp fixture under the fitted ramp params."""
    from hmmlearn.hmm import PoissonHMM
    fx = np.load(FIX / "fixtures.npz")
    man = json.load(open(FIX / "manifest.json"))
    ci = next(c["i"] for c in man["conditions"] if c["true_gen"] == "ramp")   # a ramp fixture
    p = man["conditions"][ci]["our_ramp_params"]
    C = fx[f"counts_{ci}"]; L = fx[f"lengths_{ci}"]
    counts = [C[i, :L[i]].astype(float) for i in range(len(L))]
    lp, lA, rates = B.ramp_hmm((p["drift"], p["sigma"], p["lam_lo"], p["lam_hi"]), K_PROD)
    our = sum(B.model_loglik(B.build_dataset([c]), lp, lA, rates) for c in counts)
    ref = PoissonHMM(n_components=K_PROD, init_params="", params="")
    pi, Amat = np.exp(lp), np.exp(lA)
    ref.startprob_ = pi / pi.sum()
    ref.transmat_ = Amat / Amat.sum(1, keepdims=True)
    ref.lambdas_ = (np.maximum(rates, 1e-9) * B.DT).reshape(-1, 1)
    X = np.concatenate([c.reshape(-1, 1) for c in counts]).astype(int)
    refll = ref.score(X, [len(c) for c in counts])
    return float(abs(our - refll)), int(len(L))


def main():
    bf = ramp_bruteforce_diff()
    hl, nfix = ramp_hmmlearn_diff_k50()
    out = dict(
        ramp_bruteforce_max_abs_diff=bf,
        ramp_hmmlearn_k50_abs_diff=hl,
        K_bruteforce=K_BF,
        K_production=K_PROD,
        note=("RAMP-side forward-likelihood checks (discretised bounded-DDM Poisson-HMM). "
              "Distinct from the STEPPING hmmlearn_check (~2.27e-13) in "
              "phase1_recovery_validation.json. brute-force = canonical model_loglik vs full "
              f"path enumeration at K={K_BF}, T=6; hmmlearn = model_loglik vs PoissonHMM.score() "
              f"at production K={K_PROD} on a ramp fixture ({nfix} trials). brute-force ~2e-14 "
              "matches docs/ramp_validation.md; hmmlearn diff is machine precision (~3.6e-12 "
              "absolute, ~1e-15 relative to the ~1e3 LL) -- the doc's literal 0.0 was on data "
              "sampled from the discretised chain, not real fixture spike trains."))
    OUT.write_text(json.dumps(out, indent=2))
    print(f"ramp brute-force max|diff| = {bf:.3e}  (target ~2e-14)")
    print(f"ramp hmmlearn K=50 |diff|  = {hl:.3e}  (target 0.0)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

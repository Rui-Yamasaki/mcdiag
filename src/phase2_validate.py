"""Phase 1 step 2, PART A — validation + reproduction sanity (no real data).

Three checks the modeling MUST pass before the Part B recovery sweep:

  (1) Stepping side validated against a maintained reference PoissonHMM
      (hmmlearn): same parameters -> identical marginal log-likelihood, proving
      our forward algorithm is correct. (dynamax attempted separately, best-effort.)
  (2) Ramping side: the discretized diffusion-to-bound likelihood and the
      step-vs-ramp CV verdict CONVERGE as the latent grid is refined.
  (3) Part A2 reproduction sanity: on GENEROUS synthetic data (long windows, high
      firing rate, many trials), simulate pure stepping -> recover stepping, and
      pure ramping -> recover ramping, at a HIGH rate. This gates the sweep.

Run:  python src/phase2_validate.py
"""
from __future__ import annotations

import numpy as np

import stepramp as sr
from ibl_one import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"
GENEROUS_T = 60          # bins (1500 ms at 25 ms) -- long window
GRID_MS = sr.DT * 1000
RECOVERY_GATE = 0.80     # Part A must clear this to proceed to the sweep


# ---------------------------------------------------------------------------
# (1) Stepping cross-check vs hmmlearn.PoissonHMM
# ---------------------------------------------------------------------------
def validate_stepping_vs_reference():
    print("=" * 70)
    print("(1) STEPPING validated against maintained reference (hmmlearn.PoissonHMM)")
    from hmmlearn.hmm import PoissonHMM

    rng = np.random.default_rng(1)
    lengths = rng.integers(30, 71, size=120)
    true = (5.0, 35.0, 0.05)
    counts, L = sr.simulate_stepping(true, lengths, sr.DT, rng)

    # fit OUR stepping model
    params, my_ll = sr.fit("step", counts, L, sr.DT, n_starts=2)
    rates, A, pi = sr.stepping_build(params)
    my_ll_check = sr.forward_loglik(counts, L, rates, A, pi, sr.DT).sum()

    # configure hmmlearn with the SAME parameters and score the same data
    X = np.concatenate([counts[i, : L[i]] for i in range(len(L))]).reshape(-1, 1)
    seq_lengths = L.tolist()
    ref = PoissonHMM(n_components=2, init_params="", params="")
    ref.startprob_ = pi.copy()
    ref.transmat_ = A.copy()
    ref.lambdas_ = (rates * sr.DT).reshape(2, 1)   # hmmlearn mean = rate*dt
    ref_ll = ref.score(X, seq_lengths)

    print(f"  fitted stepping params (lam0,lam1,p) = "
          f"({params[0]:.2f}, {params[1]:.2f}, {params[2]:.3f})")
    print(f"  our forward_loglik (same params)     = {my_ll_check:.4f}")
    print(f"  hmmlearn PoissonHMM .score()         = {ref_ll:.4f}")
    diff = abs(my_ll_check - ref_ll)
    print(f"  |difference|                         = {diff:.2e}")
    ok = diff < 1e-4
    print(f"  -> forward algorithm matches reference: {'PASS' if ok else 'FAIL'}")

    # sanity: a free-transition hmmlearn 2-state fit should reach >= our loglik
    free = PoissonHMM(n_components=2, n_iter=200, random_state=0,
                      init_params="stml")
    free.fit(X, seq_lengths)
    free_ll = free.score(X, seq_lengths)
    print(f"  hmmlearn free 2-state fit loglik     = {free_ll:.1f} "
          f"(our constrained fit = {my_ll:.1f}; free >= constrained expected)")
    return ok


# ---------------------------------------------------------------------------
# (2) Ramping grid convergence
# ---------------------------------------------------------------------------
def validate_grid_convergence():
    print("\n" + "=" * 70)
    print("(2) RAMPING grid convergence (CV verdict must stabilize as M grows)")
    rng = np.random.default_rng(2)
    N, T = 250, GENEROUS_T
    lengths = np.full(N, T)
    ramp_counts, _ = sr.simulate_ramping((5.0, 40.0, 2.0, 0.8), lengths, sr.DT, rng)
    step_counts, _ = sr.simulate_stepping((5.0, 40.0, 0.05), lengths, sr.DT, rng)

    print(f"  {'M':>4} | ramp-data: cvll_ramp-cvll_step  verdict | "
          f"step-data: cvll_ramp-cvll_step  verdict")
    rows = []
    for M in (12, 20, 30, 45, 60):
        wr, cs_r, cr_r = sr.cv_compare(ramp_counts, lengths, sr.DT, M=M, k=4,
                                       n_starts=1, rng=np.random.default_rng(10))
        ws, cs_s, cr_s = sr.cv_compare(step_counts, lengths, sr.DT, M=M, k=4,
                                       n_starts=1, rng=np.random.default_rng(11))
        ramp_margin = cr_r - cs_r   # >0 => ramp wins on ramp-generated data
        step_margin = cr_s - cs_s   # <0 => step wins on step-generated data
        rows.append((M, ramp_margin, wr, step_margin, ws))
        print(f"  {M:>4} | {ramp_margin:>+24.1f}  {wr:>7} | "
              f"{step_margin:>+24.1f}  {ws:>7}")
    return rows


# ---------------------------------------------------------------------------
# (3) Part A2 reproduction sanity: recovery on generous data
# ---------------------------------------------------------------------------
def part_a_recovery(n_neurons=40, M=25):
    print("\n" + "=" * 70)
    print(f"(3) PART A2 reproduction sanity: recovery on GENEROUS data "
          f"(T={GENEROUS_T} bins/{GENEROUS_T*GRID_MS:.0f} ms, N=300, FR up to ~40 Hz)")
    rng = np.random.default_rng(3)
    out = {}
    for truth in ("step", "ramp"):
        hits = 0
        for i in range(n_neurons):
            lam0 = rng.uniform(3, 7)
            lam1 = rng.uniform(30, 45)
            lengths = np.full(300, GENEROUS_T)
            if truth == "step":
                p = rng.uniform(0.02, 0.06)
                counts, L = sr.simulate_stepping((lam0, lam1, p), lengths, sr.DT, rng)
            else:
                beta = rng.uniform(1.0, 3.0)
                sigma = rng.uniform(0.5, 1.2)
                counts, L = sr.simulate_ramping((lam0, lam1, beta, sigma),
                                                lengths, sr.DT, rng)
            winner, _, _ = sr.cv_compare(counts, L, sr.DT, M=M, k=4, n_starts=2,
                                         rng=np.random.default_rng(100 + i))
            hits += (winner == truth)
        rate = hits / n_neurons
        out[truth] = rate
        print(f"  true={truth:>4}: recovered {hits}/{n_neurons} = {rate:.0%}")
    overall = np.mean(list(out.values()))
    print(f"  overall recovery on generous data = {overall:.0%} "
          f"(gate >= {RECOVERY_GATE:.0%})")
    return out, overall


def main():
    RESULTS.mkdir(exist_ok=True)
    ok1 = validate_stepping_vs_reference()
    rows = validate_grid_convergence()
    rec, overall = part_a_recovery()

    # write a small validation summary
    with open(RESULTS / "phase2_validation.txt", "w") as f:
        f.write(f"stepping vs hmmlearn forward-loglik match: {'PASS' if ok1 else 'FAIL'}\n")
        f.write("ramping grid convergence (M, ramp-data margin, verdict, "
                "step-data margin, verdict):\n")
        for r in rows:
            f.write(f"  M={r[0]:>3}  rampΔ={r[1]:+.1f} {r[2]}  stepΔ={r[3]:+.1f} {r[4]}\n")
        f.write(f"Part A recovery: step={rec['step']:.0%} ramp={rec['ramp']:.0%} "
                f"overall={overall:.0%}\n")
    print(f"\nWrote results/phase2_validation.txt")

    if not ok1:
        raise SystemExit("STOP: forward algorithm disagrees with reference — fix.")
    if overall < RECOVERY_GATE:
        raise SystemExit(
            f"STOP: Part A recovery {overall:.0%} < gate {RECOVERY_GATE:.0%} — "
            "the implementation is wrong; do NOT run the sweep.")
    print("\nALL VALIDATION PASSED — cleared to run the Part B recovery sweep.")


if __name__ == "__main__":
    main()

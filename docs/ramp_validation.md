# Ramping-model validation — the last open flank

The recovery boundary (§3.1) + the 0.57 population gate (§3.5) all run on a **self-built discretised
bounded-drift-diffusion ("ramping") Poisson-HMM** (Pipeline B, `phase1_recovery.py`) whose forward
likelihood — unlike the stepping side — was **never validated** against a reference. This closes that
flank locally: the ramp forward likelihood is checked against brute-force path enumeration and
`hmmlearn` to machine precision. Reproduce: `python src/run_ramp_validate.py`
(outputs `results/ramp_validation_{robustness,reconcile}.csv`); the forward-likelihood fixtures are
independently re-derived and asserted by `tests/test_smoke.py`.

## A. Ramp forward likelihood is EXACT (the core missing check) — **PASS**
| check | result |
|---|---|
| brute-force path enumeration, **Pipeline B** (point-weight, M=5) | max \|fwd − brute\| = **2.1e-14** |
| brute-force path enumeration, **Pipeline A** (CDF cell-integ, M=5) | max \|fwd − brute\| = **1.9e-14** |
| **hmmlearn.PoissonHMM** `.score()` vs our forward LL, production **K=50** | abs diff = **0.0e+00** |

The discretised-ramp marginal likelihood is computed to machine precision — the same standard the
stepping side met (hmmlearn 2.3e-13). **The forward-likelihood flank is closed.**

## B. Two engines are inference-equivalent; grid is robust — **PASS**
- **Likelihood:** Pipeline A (cell-integration) vs B (point-weight Gaussian) on identical data:
  LL relative difference **~3e-4, stable across K∈{10,25,50,100}**. The element-wise transition-matrix
  difference grows with K but is **boundary-only** (max at row 48/49, x=0.98, next to the absorbing
  bound; interior rows agree to 0.12) — immaterial to the marginal likelihood and recovery.
- **Generators** (both clipped-Gaussian-walk → Poisson) match: 21.2 vs 20.9 Hz at matched params.
- **Grid resolution** doesn't move recovery: at (N=160, FR=20, σ=0.4), K=25/50/75 → **89/83/88 %**.
- **Canonical engine = Pipeline B** (the headline one): validated against A, against brute-force, and
  against hmmlearn. (A's cell-integration is marginally more accurate at the bound but inference-
  equivalent; the unused Pipeline A can be archived.)

## C. σ ROBUSTNESS — the exact boundary is **CONDITIONAL on the ramp diffusion σ**
σ (ramp diffusion) was the one generator knob never swept; the headline used **σ=0.4** ("a clearly
gradual ramp"). Sweeping it (R=40, K=50):

**Recovery % (mean step+ramp) vs σ:**
| (N, FR) | σ=0.2 | **σ=0.4 (headline)** | σ=0.7 | σ=1.0 |
|---|--:|--:|--:|--:|
| 40, 20 | 59 | 68 | 54 | 51 |
| **160, 20** | 89 | **83** | 72 | 61 |
| **320, 20** | 94 | **89** | 78 | 74 |
| 80, 40 | 92 | 75 | 74 | 70 |
| 160, 40 | 96 | 90 | 80 | 80 |

**TRUE-RAMP recovery vs σ** (σ makes ramps look steppier → harder):
| (N, FR) | σ=0.2 | σ=0.4 | σ=0.7 | σ=1.0 |
|---|--:|--:|--:|--:|
| 160, 20 | 92 | 88 | 57 | **50 (chance)** |
| 320, 20 | 98 | 95 | 60 | 57 |
| 160, 40 | 100 | 88 | 70 | 62 |

- **At σ=0.4 the published map is REPRODUCED** (N=160/FR=20 → 83%, N=320 → 89% ≈ the headline ~84/85%).
- **For steeper ramps the boundary SHIFTS**: at σ=0.7 the ≥80% line needs more data (N=160/FR=20 falls
  to 72%); at σ=1.0, **true-ramp recovery collapses to chance** (a near-bound-slamming ramp is
  genuinely indistinguishable from a step). The exact thresholds widen by ~1 N-step per ~0.3 in σ.

## Decisive verdict
**The ramping model VALIDATES structurally; the headline NUMBERS are σ-conditional.**
- **Forward likelihood: VALIDATED** (exact vs brute-force + hmmlearn) — the flank that was open is now
  closed to machine precision.
- **Engine + grid choices: VALIDATED** (two independent discretisations agree on inference; K robust).
- **Qualitative claims: ROBUST / over-determined** — single-session (N≈40) is below usable for *every*
  σ (51–68%), so "not identifiable per session, pooling required" holds regardless of σ; the §3.5
  pseudo-population/simultaneity limit is unaffected.
- **Exact recovery thresholds + the 0.57 gate: CONDITIONAL on σ≈0.4 → report a sensitivity band, not
  fixed numbers.** They reproduce at σ=0.4 but degrade for steeper ramps; since the 0.57 gate's
  pre-flight used σ=0.4, the gate value inherits the same conditionality (a steeper true ramp → harder
  recovery → higher effective bar). §3.1/§3.5 should state the σ-sensitivity explicitly.

**What ran where.** Parts A–C above ran **locally** (numpy/scipy/hmmlearn). The **canonical-accumulator
model-equivalence** — is the discretised DDM the *same model* as Zoltowski-2019's `ssmdm.Accumulation`,
and does it reproduce a published recovery confusion? — would require `ssm`/`ssmdm`, which need a C
compiler and do not build on this Windows/py3.13/numpy-2 box; that canonical cross-check was **scoped
out** of this repo. Honest status: forward-likelihood + discretisation + engine flanks **closed
locally** (and re-derived by `tests/test_smoke.py`); the literal canonical-accumulator equivalence is
**not included**; the exact thresholds/gate carry a now-quantified **σ caveat**.

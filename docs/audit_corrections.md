# Audit corrections — honest recomputed numbers

The cheap, do-now corrections from [technical_audit.md](technical_audit.md). Cached data only, no
downloads, no git. Reproduce: `python src/audit_corrections.py --stage all`. Outputs:
`results/audit_{pooled_effects,cascade_fdr,gate_debias,decode_2000perm}.csv`.

## 1. Pooled decision effect — cell-clustered bootstrap CI is now PRIMARY; MWU p dropped
The Mann–Whitney p was computed on **pooled single trials** (10–16 k) while the independent unit is
the **cell** (200–1000) — pseudoreplication, so the p is trial-count-driven (audit §C). It is dropped.
The cell-clustered bootstrap CI (resample cells) is the honest inference, and the comparison is now
**MRN-to-MRN** (the gate was comparing IBL *all-region* to Steinmetz *MRN-only*).

| effect | SD | 95% CI (cell-clustered) | cells | (pseudorep MWU p — dropped) |
|---|--:|---|--:|--:|
| **IBL MRN** (error) | **+0.059** | **[+0.001, +0.117]** | 356 | 0.034 |
| **Steinmetz MRN** (equal) | **+0.095** | **[+0.050, +0.139]** | 223 | 5.6e-7 |
| IBL all-region (error) — *context, not MRN* | +0.095 | [+0.060, +0.131] | 815 | 1.9e-7 |

- The headline "+0.085, p=5.6e-7" is **all-region**; the comparable **IBL MRN** effect is **+0.059 and
  marginal** (CI lower bound ≈ 0). **Steinmetz MRN (+0.095) is cleaner than IBL MRN**, not "identical."
- The "+0.085 vs +0.095, same p=5.6e-7" coincidence is dead: scope-mismatched + pseudoreplicated +
  rounded (true p's 5.587e-7 vs 5.592e-7). Both effects are real **by the bootstrap CI**, both small,
  both well below the 0.24 SD population gate.

## 2. Hard-coded constant fixed
`steinmetz_replicate.py: IBL_EFFECT_SD` **0.085 → 0.059** (it was the all-region value; the MRN value
is +0.059). The gate **verdict is unchanged** — the decision uses the 0.24 SD bar, not this constant;
only the displayed IBL-MRN reference becomes honest. The comparison figure should be regenerated
(`python src/steinmetz_replicate.py`) so panel 3 shows IBL-MRN 0.059 vs Steinmetz-MRN 0.095.

## 3. Population decode — proper permutation test (was floored AND under-dispersed)
The original null (`N_PERM=80`, `R_SUB=12`) had **two** problems: (a) p floored at 1/81 = 0.0123;
(b) each null draw re-shuffled **every rep**, averaging over *different* permutations → it estimated
the null *mean* with artificially collapsed variance (anti-conservative). The corrected test fixes
**one** permutation per draw and computes the **identical** mean-of-R statistic (matched `R=20`, 400
perms):

| decode | AUC | corrected p | null mean | **null p95** |
|---|--:|--:|--:|--:|
| IBL MRN error-only | 0.865 | **0.0025** | 0.483 | **0.734** |
| Steinmetz MRN equal-only | 0.913 | **0.0075** | 0.509 | **0.809** |

- **The decode SURVIVES** (p < 0.01, above the proper null) — the distributed, per-cell-cryptic choice
  code is **real**. The corrected p is comparable to / smaller than the old floor, so the qualitative
  conclusion holds.
- **But the proper null is enormously broad** (p95 ≈ 0.73–0.81, not ~0.5): a **single decode rep hits
  AUC ≈ 1.0 even under shuffle**, because 300 resampled pseudo-trials × 243 cell-features overfit. So
  **the AUC magnitude is overfitting-inflated and is NOT literal decodability** — report it as
  "significantly above the permutation null," never as "79–92 % of choices decodable." (New finding
  from this pass; folded into audit §B4.)

## 4. Selectivity cascade — BH-FDR through the chain collapses the per-cell "decision cells" to ZERO
The lead/lock/move/stim/**triple** chain used **raw p<0.05**. Applying BH-FDR (q<0.05) to each branch:

| region | triple **raw** | triple **FDR** | (stim-survive raw → FDR) |
|---|--:|--:|--:|
| IBL MRN | 29 | **0** | 62 → 0 |
| IBL SCm | 17 | **0** | 36 → 0 |
| IBL GRN/SNr/IRN | 2/3/8 | **0/0/0** | — |
| Steinmetz MRN | 4 | **0** | 7 → 0 |
| Steinmetz SC | 4 | **0** | 4 → 0 |

- **No single cell's choice selectivity survives the stimulus control under FDR** (`stim_surv_fdr = 0`
  everywhere) → **`triple_fdr = 0` in every region, both datasets.** The "29 triple decision cells"
  etc. were a raw-p phenomenon.
- This is not fatal — it **strengthens** the "distributed but per-cell-cryptic" thesis: there is **no
  detectable single-cell** stimulus-independent decision unit after correction; the decision signal
  exists **only at the population level** (the decode, §3). Report `triple_fdr` (=0) as the honest
  per-cell count and lean the decision claim entirely on the population decode + the pooled bootstrap.

## 5. Gate debiasing consistency — bar re-expressed in the measured metric
The 0.57 bar was defined on **clean (1500-trial) per-cell AUC, no debiasing**; Steinmetz's 0.535 was
**debiased ~50-trial AUC (clip ≥0)** — different estimators. Applying the data estimator to simulated
populations at each gscale:

| gscale | clean AUC (defines bar) | debiased AUC (data estimator) | pooled eff (SD) |
|--:|--:|--:|--:|
| 0.045 (the 0.57 bar) | 0.572 | **0.545** | 0.241 |

- In the **consistent (debiased) metric the bar is ~0.545**; Steinmetz measured **0.535 → still BELOW**,
  but the margin is thin (0.01). The **robust route is the SD metric**: Steinmetz **0.095 vs bar 0.24**
  (2.5×). **Verdict unchanged (arm gated)**, now apples-to-apples; lead with the SD metric, not the
  per-cell-AUC metric.

---

## 6. DECISIVE TEST — real-trial decode: the strong distributed-code headline ERODES
The §3 corrected decode still used **pseudo-trial resampling** (300 pseudo-trials/class from a
handful of real trials × 243 features → single-rep AUC≈1.0). The honest test decodes on **real
trials**. Since the 243-cell pool is a **pseudo-population** (cells from disjoint sessions, never
co-recorded), a real-simultaneous-trial decode is only possible **per session**; we run it on every
decodable session (≥8 simultaneous cells, ≥5/side real decorrelated trials), held-out CV, movement
residualised within fold, a-priori-fixed L2 (no leakage), aggregated across sessions with a properly-
specified permutation null (`src/audit_realtrial_decode.py`, `results/audit_realtrial_decode.csv`).
Movement control: **none → linear (4 covariates) → EXPANDED** (degree-2 polynomial + interactions =
nonlinear, multidimensional).

| region | sessions | **old pseudo AUC** | real-trial, **no mov** | + **linear** mov | + **nonlinear** mov |
|---|--:|--:|--:|--:|--:|
| **IBL MRN** | 28 | 0.79–0.87 | **0.615 (p=.002)** | 0.581 (p=.002) | **0.528 (p=.057)** |
| **Steinmetz MRN** | 7 | 0.91–0.92 | **0.651 (p=.002)** | 0.621 (p=.007) | **0.541 (p=.095)** |
| IBL SCm | 18 | — | 0.619 (p=.002) | 0.543 (p=.05) | 0.503 (p=.42) |
| Steinmetz SC | 6 | — | 0.600 (p=.017) | 0.544 (n.s.) | 0.546 (n.s.) |
| IBL/Stein SNr | 1/4 | — | n.s. (too few) | n.s. | n.s. |

**Findings.** (1) **The strong AUC was pseudo-trial overfitting**: real-trial CV with *no* movement
control is only **0.60–0.65** (vs the inflated 0.79–0.92) — a ~0.15–0.30 AUC drop from removing the
resampling. (2) **A weak real signal exists**: MRN is significant on real trials (p=0.002) and survives
**linear** movement control (IBL 0.581 p=.002; Steinmetz 0.621 p=.007). (3) **It does NOT robustly
survive a stringent nonlinear movement control**: MRN falls to **marginal** (IBL 0.528 **p=0.057**;
Steinmetz 0.541 **p=0.095**), and SCm/SC erode already at the linear stage. (The 14-feature expanded
model on ~30 trials risks some over-removal — so the truth sits between the linear (sig) and nonlinear
(n.s.) results — but the null stays tight at ~0.50/p95~0.53, so the test has power, not a degenerate
collapse.)

**VERDICT — ERODES (strong → weak-and-movement-fragile).** The paper **cannot** headline a *strong
distributed MRN decision code*: the 0.79–0.92 AUC was pseudo-trial overfitting, and what remains on real
trials is a **weak (AUC ~0.58–0.62) MRN choice signal that is only partly separable from movement** —
significant after linear movement control but **not robust to a stringent nonlinear one** (choice ≡
wheel direction). SCm/SC do not survive even linear movement control. **The honest claim is a weak,
movement-entangled MRN population choice signal, not a strong distributed decision code** — i.e. the
positive finding should be downgraded to cautionary, and the framework/identifiability arc carries the
paper. (The small MRN *pooled decision effect* from §1, cell-clustered CI [+0.050,+0.139] on equal-
contrast trials, is the one decision-signal statistic that does not depend on the decode and is not a
pure movement artifact; it remains the most defensible positive, but it is small.)

## Net effect on the claims
- **DOWNGRADED (the big one, §6):** the **strong distributed MRN decision decode does NOT survive** a
  real-trial CV with nonlinear movement control — 0.79–0.92 was pseudo-trial overfitting (real-trial
  0.60–0.65), and the remnant is weak (~0.58–0.62) and only partly movement-separable (sig under linear,
  marginal p=.06–.10 under nonlinear movement control). **No strong distributed-code headline.**
- **Most defensible positive that remains:** the **small MRN pooled decision effect** (cell-clustered
  bootstrap CI [+0.050,+0.139] on equal-contrast trials, Steinmetz) — real but small, and not a pure
  movement artifact (it is the cell-aligned rate difference, not a high-dim decoder).
- **Corrected/removed:** the "+0.085≈+0.095, same p" coincidence (scope-mismatch + pseudoreplication);
  the per-cell `triple` decision-cell counts (→ 0 under FDR); the hard-coded IBL constant; the floored/
  under-dispersed decode p; **and now the inflated decode AUC itself.**
- **Verdicts unchanged:** step-vs-ramp not identifiable single-session; population arm gated; framework
  + identifiability arc is the durable contribution. The cross-dataset *replication* now rests on the
  weak pooled effect + the (eroded) decode, **not** on a strong distributed code or "identical numbers."

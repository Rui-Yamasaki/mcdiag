# Technical audit — adversarial review of every method, parameter, and decision

Scope: an end-to-end, hostile-reviewer audit of the whole project (identifiability sim →
deliberation window → selectivity cascade → decision signal / population decode → population
recovery pre-flights → Steinmetz replication + the 0.57 gate). Read from the **code**, not memory.
Conventions honoured: no git; only cheap inline checks run (flagged); expensive robustness runs
listed as follow-ups. Honesty over reassurance.

**Two findings before Part A even starts:**
1. **`docs/manuscript_draft_v1.md` does not exist** (only `steinmetz_replication_feasibility.md` is
   in `docs/`). There are no `[CC]` markers to resolve against a draft. Part A below therefore *is*
   the ground-truth Methods parameter table; the manuscript must be written from it.
2. **There are two parallel implementations of the core identifiability engine**, and only one was
   ever run (see §B0). This is the first thing a reviewer reading the repo will trip over.

---

## PART A — ground-truth parameter / decision table (from code)

### A0. Which engine is canonical (critical)
| pipeline | files | executed? | forward LL validation | ramp grid | CV | modulation | ramp σ |
|---|---|---|---|---|---|---|---|
| **B (headline)** | `phase1_recovery.py`, `_hardened.py`, `phase2_mrn_recovery_preflight.py`, `steinmetz_*` | **YES** (`phase1_recovery_*.csv`, `_validation.json` exist) | vs hmmlearn, \|Δ\|=2.3e-13 | K=50 (swept 30–100) | single 0.3 hold-out | r_lo=⅔FR→r_hi=4⁄3FR (2:1) | 0.4 |
| A (unused) | `stepramp.py`, `phase2_validate.py`, `phase2_recovery_sweep.py` | **NO** (no `phase2_recovery.csv`, no `phase2_validation.txt`) | vs hmmlearn (\|Δ\|<1e-4) **+ brute-force enumeration** (atol 1e-8) | M=25 (swept 12–60) | 4-fold | 0.5FR→1.5FR (3:1) | 0.7 |

The headline numbers (recovery map, §3.5 gate, Steinmetz gate) all run on **Pipeline B**. Pipeline A
is dead code that ships the *strongest* correctness test (brute-force) and a *different* parameterisation.

### A1. Recovery engine (Pipeline B — `phase1_recovery.py`)
- bin `DT=0.025 s`; `RATE_FLOOR=1e-3 Hz`; `NEG_INF=-1e30`.
- **Stepping**: 2-state left→right Poisson HMM, params `(λ_lo, λ_hi, p)`, start state 0, state 1 absorbing, geometric step time.
- **Ramping**: discretised drift-diffusion to absorbing bound on a `K=50`-point grid; Gaussian transition *weights at grid points* (row-normalised; reflecting at 0, forced-absorbing at 1); params `(drift, σ, λ_lo, λ_hi)`. Generator constants `GEN_DRIFT=1.0`, `GEN_SIGMA=0.4`.
- rate envelope (both families): `λ_lo=⅔·FR`, `λ_hi=4⁄3·FR` (trial-mean ≈ FR; 2:1 modulation).
- fit: L-BFGS-B, `maxiter=30`, `ftol=1e-6`; generator-agnostic inits from early/late bin means.
- CV: **single Monte-Carlo hold-out, `holdout=0.3`**; winner = higher held-out forward-LL (no k-fold).
- sweep: `N∈{20,40,80,160,320}`, `FR∈{2,5,10,20,40}`, `R=60` reps/cell, seeds `13000+sid`.
- validation (`phase1_recovery_validation.json`): hmmlearn match 2.27e-13; grid K=30/50/75/100 recovery 72.5/73.8/72.5/72.5 %; Part A2 (1.75 s, 40 Hz, N=150) overall 92.5 %.

### A2. Hardening (`phase1_recovery_hardened.py`)
- NB (Gamma-Poisson) emission, constant Fano `∈{1,1.5,2,3}`; Poisson-fitter-on-NB collapses to chance, matched-NB fitter restores. Stepiness knob `s∈{0,…,1}` blends ramp↔step. `K=50`, `HOLDOUT=0.3`, `N_JOBS=2`, BelowNormal priority.

### A3. Deliberation window / engagement
- engagement (Phase 1): `choice≠0`, finite `stimOn/firstMovement/response`, window in `(0,60] s`. **No IBL per-trial QC/engagement column exists** — this filter is ours.
- curation: **drop every session containing a 50 % contrast trial** (training-contaminated); keep trained biased-ephys (64 sessions).
- recovery window pool (`load_windows`): trained, `|contrast|∈{0,6.25,12.5}%`, `choice≠0`, `RT∈[0.5,2.0] s` → 2 074 RTs, median 1020 ms.

### A4. Census + selectivity cascade (`phase2_census.py`, `phase2_selectivity.py`)
- **unit QC**: `clusters.metrics.label ≥ 1.0` (IBL composite of 3 metrics — refractory, amplitude, noise-cutoff; ==1 ⇒ all pass).
- **FR floor**: in-window `fr_window ≥ 10 Hz`.
- **trials**: `RT∈[0.5,2.0] s`, `|contrast|≤25 %`, `choice≠0`, `MIN_PER_SIDE=8`.
- region: Beryl acronym from `brainLocationIds_ccf_2017` at the **peak channel**; core `{GRN,MRN,SNr,SCm,IRN}`.
- selectivity: choice ROC-AUC; permutation null `N_PERM=2000`; **BH-FDR `q<0.05`** on the deliberative-window test only. Conditional classifications (raw `p<0.05`): movement-survive = linear residual of `[wheel_speed, wheel_disp, (+DLC paw/nose if present)]`; stimulus-survive = linear residual of `[signed, pL]`; lead window `[stimOn+0.10, fm−0.20]`, peri `[fm−0.10, fm+0.10]`. `triple = sig_raw ∧ leading ∧ move_survive ∧ stim_survive`.

### A5. Choice-vs-stimulus (`phase2_choice_vs_stim.py`)
- `N_PERM=1000`, `FDR_Q=0.05`, `MIN_SIDE=3`, `MIN_DECORR=6`.
- decorrelated = **error** (`|c|>0 ∧ chose≠stim`) + **0 %-contrast**; `joint_choice_test` = OLS choice coeff controlling `[signed, pL]`, permutation **within stimulus-side strata**.
- pooled cell-aligned error effect: per-cell z-scored error rates, congruent vs incongruent (pref from correct trials), **MWU on pooled trials**, **cell-clustered bootstrap CI (2000)**.

### A6. Population decode (`phase2_population_decode.py`)
- `MIN_SIDE=4`; pseudo-trials `N_TR=300`/`N_TE=120`; `R_OBS=40`; **`N_PERM=80`**, `R_SUB=12`; ridge `RIDGE=1.0`; `LOGIT_C=0.05` (fixed, not CV-selected); `CURVE_REPS=30`.
- within-fold ridge residualisation of `[wheel, body=mean(paw,nose), signed, pL]` (β fit on train, applied to test); StandardScaler on train; L2-logistic; held-out ROC-AUC; perm null shuffles choice within cell. Strict control = error-only.
- **perm-p floor = 1/(80+1) = 0.0123** (see §B4).

### A7. §3.5 population pre-flight (`phase2_mrn_recovery_preflight.py`)
- per-cell baseline FR lognormal `μ=ln24.3, σ=0.70`; coupling `MRN_GSCALE=0.015, MRN_GSIGMA=1.0`; easy `0.20/0.5`; `K_GRID=50`; `Fano=2.0`; hold-out 0.3; **oracle per-cell tuning**; `N_JOBS=8`; reps easy40/mrn60/grid24; null `gscale=0`.

### A8. Steinmetz pre-flight + replication (`steinmetz_population_preflight.py`, `steinmetz_adapter.py`, `steinmetz_replicate.py`)
- coupling sweep `gscale∈{.015,.022,.032,.045,.065,.090}` → calibrated per-cell AUC 0.529–0.622 / eff 0.10–0.41 SD; aggregation `n_ens=600`, majority + pooled-evidence, balance guard. **0.57 gate ≙ 0.24 SD** (calibration map).
- adapter: `chose=-response`; `fm_bin=50+round(RT_ms/10)` capped at response bin; FR floor 10 Hz; regions MRN / SC=SCm+SCig / SNr; movement = wheel+face+pupil; **no prior** (`pL≡0.5`); decorrelated = equal-contrast ∪ error.
- replicate: cascade at **full contrast** (`CONTRAST_MAX=0.25` defined but cascade reverted to all-contrast for power); choice coding + gate on **equal-contrast** trials; gate debiasing = per-cell `|AUC−.5|` minus 200-shuffle null, **clipped ≥0**; `IBL_EFFECT_SD=0.085` hard-coded as the benchmark **(this is the all-region value — see §C)**.

---

## PART B — per-analysis adversarial audit

### B0. Two-engine problem — verdict NEEDS-CHECK (process), not a result bug
- **Assumption**: the recovery boundary is engine-invariant. Untested: Pipelines A and B use different grids (25 vs 50), CV (4-fold vs single 0.3 hold-out), modulation (3:1 vs 2:1) and ramp σ (0.7 vs 0.4), and were **never cross-checked against each other**.
- **Degrees of freedom**: every one of those four knobs.
- **Attack**: "you shipped two engines, ran the one with the *weaker* validation (no brute-force check) and the *gentler* ramp (σ=0.4 looks more ramp-like ⇒ easier to tell from a step ⇒ optimistic recovery), and deleted the other's outputs." The repo cannot currently rebut this.
- **Bug hunt**: no logic bug, but the README documents Pipeline B while Pipeline A's separate figure-output path and `phase2_*` naming imply it was meant to supersede B — a provenance hazard.
- **Verdict NEEDS-CHECK**: pick ONE engine; run the brute-force enumeration test on it; delete or clearly archive the other; confirm the recovery map is materially the same on both.

### B1. Recovery engine faithfulness (the load-bearing sim) — verdict NEEDS-CHECK→VULNERABLE
- **Assumption**: the self-built 2-state step HMM and the K-grid discretised diffusion are faithful proxies for the canonical Latimer/Zoltowski stepping and ramping (DDM) models.
- **What is validated**: the *forward algorithm* computes the correct Poisson-HMM marginal likelihood (hmmlearn 2.3e-13; brute-force in the unused engine). The *stepping* model is a textbook 2-state HMM — solid.
- **What is NOT validated**: the *ramping* model is a home-grown discretised DDM. It is checked only by internal grid-convergence and by recovering its *own* generator on generous data. It is **never compared to canonical `ssm`/`dynamax`/Latimer code** (ssm wouldn't compile — the stated reason). So "step-vs-ramp is/ isn't recoverable" rests on a ramp model no external reference has blessed.
- **Attack surfaces**: (i) the absorbing-bound discretisation and the reflecting-at-0 boundary are modelling choices that change the likelihood; (ii) the generator simulates a *clipped* Gaussian walk while the fitter uses Gaussian *grid* transitions — generator and fit are not the same discretisation (a known small inconsistency); (iii) σ=0.4 is a single, gentle diffusion — the README itself calls σ "the main robustness knob," and only one value was run.
- **Bug hunt**: in `ramp_hmm` the transition weights are evaluated at grid *points* (not integrated over cells), unlike Pipeline A's cell-edge CDF integration — the cruder of the two; at K=50 the effect is small but it is a real approximation difference between the two engines.
- **Verdict VULNERABLE**: the entire identifiability boundary *and* the 0.57 gate inherit the unvalidated ramp model. This is the single biggest technical exposure (Tier-2 #4). It does not obviously *break* the qualitative conclusion (per-session non-identifiability is over-determined), but every *number* is contingent.

### B2. Deliberation window / engagement — verdict SOLID (with one caveat)
- **Assumption**: `firstMovement−stimOn∈[0.5,2.0] s`, low contrast, `choice≠0`, trained-only = "engaged deliberation."
- **Degrees of freedom**: RT bounds (0.5, 2.0), the contrast set {0,6.25,12.5}, the 50 %-contrast session drop, the (0,60] s engagement window.
- **Attack/answer**: the 50 %-drop curation is *post-hoc* (chosen because the naïve pool failed monotonicity) — a forking-paths move, but it is principled and disclosed, and `firstMovement` is wheel-derived/imperfect (17 % missing, 9 % negative). Engagement is inferred, not measured (no QC column).
- **Verdict SOLID**: defensible and disclosed; the only real exposure is that the window *distribution* feeds the recovery sim, so RT-bound sensitivity propagates (Tier-2 #6).

### B3. Selectivity cascade — verdict SOLID-but-confounded-by-design
- **Assumptions**: linear movement residualisation removes movement; `triple` = genuinely decision-related.
- **Degrees of freedom (many)**: FR floor 10 Hz; QC label ≥1; AUC vs other selectivity; `N_PERM=2000`; FDR vs raw for each branch (note: only the *primary* test is FDR-controlled; lead/lock/move/stim survival use **raw p<0.05**, so those counts are chance-inflated by ~5 %·N); lead window margins (0.10, 0.20); peri (±0.10).
- **Attack**: "your `leading`/`move_survive` counts use raw p, not FDR — they include ~`0.05·N` false positives, and `triple` chains four raw filters." True; only `choice_fdr` is corrected. The `triple` count is presented as the headline thinned population and is *not* itself FDR-controlled end-to-end.
- **Bug hunt**: none functional; the linearity limitation is the substance (→ B5).
- **Verdict SOLID** for the *direction* (movement+stimulus dominate; thin leading, movement- and stimulus-controlled subset), **NEEDS-CHECK** for the exact `triple` counts (raw-p chaining).

### B4. Population decode — verdict SOLID-direction / NEEDS-CHECK-magnitude
- **Assumptions**: within-fold residualisation removes the mean-zero artifact and the confounds; pseudo-population is legitimate.
- **Confirmed correct**: `_ridge_resid` fits β on **train** indices and applies to **test** (no leakage); StandardScaler fit on train; `LOGIT_C=0.05` is **fixed** (no hyperparameter-selection leakage); the within-fold split *does* fix the global-residualisation anti-correlation trap. Good.
- **Three real problems** (the 3rd found during the corrections pass — see `audit_corrections.md` §3):
  (1) **`N_PERM=80` ⇒ perm-p floored at 1/81 = 0.0123.** (2) **the permutation null is UNDER-DISPERSED
  / mis-specified**: each null draw is `decode(perm=True, reps=12)`, which re-shuffles labels *every
  rep* and averages over 12 *different* permutations → it estimates the null *mean* with collapsed
  variance (anti-conservative), so "0/80 exceeded" overstated the result. The **correct** test fixes one
  permutation per draw with matched reps; redone (`R=20`, 400 perms): IBL-MRN AUC 0.865 **p=0.0025**,
  Steinmetz-MRN AUC 0.913 **p=0.0075** — **still significant**, but the proper **null is enormously broad
  (p95 ≈ 0.73–0.81)**: a *single* decode rep hits AUC≈1.0 even under shuffle (300 pseudo-trials × 243
  features overfit), so **the AUC is NOT literal decodability** — only "above the permutation null."
  (3) **movement residualisation is linear** (→ B5): choice ≡ wheel direction, so nonlinear/multi-D
  motor signal is under-removed and reappears as "decision."
- **Verdict**: the *qualitative* "distributed code beats per-cell" is SOLID and **survives the corrected
  permutation test (p<0.01)**. The *AUC magnitude* (0.79–0.92) is overfitting-inflated and must be
  reported as relative-to-null, not literal; the linear-movement attribution remains NEEDS-CHECK (B5).

### B5. Movement control linearity — verdict VULNERABLE (quantify)
- The control everywhere (`residual_rate`, `_ridge_resid`) is a single linear regression on `[wheel, body, …]`. If firing depends on movement **nonlinearly or multidimensionally**, linear residualisation under-removes movement, and because **choice direction = wheel direction**, the leftover inflates the decision residual/decode. No analysis bounds this. The deliberative window ends at movement onset (so gross movement is largely excluded *by construction*), which mitigates but does not eliminate motor-preparation leakage.
- **Verdict VULNERABLE → CONFIRMED (corrections pass).** Done: real-trial per-session CV decode with an
  EXPANDED nonlinear movement model (degree-2 polynomial + interactions). The decode **erodes** — MRN
  real-trial AUC 0.60–0.65 (no mov) → 0.58–0.62 (linear, sig) → **0.528/0.541 marginal (p=.057/.095)
  under nonlinear movement**; SCm/SC erode at the linear stage. So a meaningful part of the "decision
  decode" **was residual movement** (choice ≡ wheel direction), and the rest was pseudo-trial overfitting.
  The strong distributed-code headline does not survive. (`audit_corrections.md` §6.)

### B6. Decorrelated trials truly confound-free? — verdict NEEDS-CHECK
- **IBL** 0 % + error: 0 %-contrast choice is **prior-driven** (block pL); the decode's error-only strict control removes the prior driver, and `joint_choice_test` includes pL — the prior fix is real **but partial** (error trials still carry graded stimulus; the regression leaves only the orthogonal choice variance, which is thin). Engagement, RT, and **choice-history** are *not* controlled on decorrelated trials — a hostile reviewer notes errors are non-random (cluster after errors, at lapses, slow RTs), so "decision" may be confounded with post-error/low-engagement state.
- **Steinmetz** equal-contrast: no block prior by design (good), but equal-contrast choice is internally/bias-driven; choice-history and engagement again uncontrolled, and on equal-**nonzero** contrast (0.25=0.25, 1=1) bilateral stimulus drive is present (no *side*, but arousal/whisking differs by contrast).
- **Verdict NEEDS-CHECK**: stimulus and prior are handled; **choice-history, engagement, and RT are not** — list as confounds or control them.

### B7. Per-cell-AUC debiasing + the 0.57 gate consistency — verdict NEEDS-CHECK
- **The threshold (0.57)** was defined in `steinmetz_population_preflight.calibrate`: per-cell AUC = mean over cells of `max(AUC,1−AUC)` on **1500 clean simulated trials**, **no debiasing** (large-n ⇒ tiny finite-sample inflation). The 0.57↔0.24 SD map comes from the *same* sim's `percell_eff_sd` = mean per-cell |Cohen's d|.
- **The measured value (0.535)** in `steinmetz_replicate.gate_measurement`: per-cell `max(AUC,1−AUC)` on **~50 equal-contrast trials**, **debiased** by subtracting a 200-shuffle null and **clipping ≥0**.
- **These are different estimators** (1500 vs 50 trials; no-clip vs clip-at-0; AUC-deviation vs Cohen's-d for the SD route). The clip-at-0 biases the debiased AUC **upward**, and the SD-route statistic (pooled cell-aligned effect) is not identical to the calibration's mean-|d|. So the gate compares quantities measured by **non-identical procedures**.
- **Why the conclusion still holds**: Steinmetz lands at 0.095 SD vs a 0.24 SD bar (2.5× margin) and 0.535 vs 0.57 — the gap is large enough that procedural inconsistency does not flip the verdict. But it is **not airtight**.
- **Verdict NEEDS-CHECK**: measure threshold and data with the *identical* estimator (run the sim-calibration per-cell-AUC procedure on the real equal-contrast data, or measure the real pooled-SD statistic in the sim).

### B8. Steinmetz adapter — verdict SOLID (smoke-validated) with caveats
- sign convention verified (`corr(response, sign(cr−cl)) = −1.0` on correct ⇒ `chose=−response`); movement-onset bin sanity 100 % in `(stim, response]` after the cap; PSTH diverges post-stim. Good.
- caveats: `reaction_time` reference frame inferred (ms-rel-stim), not documented — a wrong frame would shift the lead/lock split (the rate-window result is robust since it ends at the response cap); binned 10 ms data cannot give true movement-onset-locked windows (flagged: needs the figshare spike-time release for finer alignment); face motion-energy ≠ DLC body pose (partial movement control).

---

## PART C — the "+0.085 vs +0.095, same p=5.6e-7" — RESOLVED: not a coincidence

**How each p is computed** (`phase2_choice_vs_stim.report`, `steinmetz_replicate.pooled_decision_effect`):
identical code — `mannwhitneyu(z_congruent, z_incongruent, alternative='greater')` on **pooled
single-trial z-scored rates**, plus a cell-clustered bootstrap CI.

**Recomputed exactly (cheap inline check):**
| statistic | effect | MWU p | bootstrap CI | pooled n | cells |
|---|---|---|---|---|---|
| IBL **all-region** (error) — *the README "0.085 / 5.6e-7"* | +0.0849 SD | 5.587e-7 | — | 15 859 | 1007 |
| Steinmetz **MRN-only** (equal) | +0.0946 SD | 5.592e-7 | [+0.051,+0.138] | 10 139 | 223 |
| IBL **MRN-only** (error) — *the apples-to-apples one* | **+0.0589 SD** | **0.034** | **[+0.003,+0.115]** | 6 027 | 356 |

**Findings:**
1. **Scope mismatch.** The headline "IBL +0.085, p=5.6e-7" is the **all-region** pooled effect; it was
   compared to Steinmetz's **MRN-only** effect. The correct MRN-to-MRN comparison is **IBL +0.059
   (p=0.034, CI barely excludes 0) vs Steinmetz +0.095** — Steinmetz MRN is *cleaner* than IBL MRN,
   not "identical."
2. **The matched p is a rounding artifact, not a coincidence.** True values are 5.587e-7 and 5.592e-7
   (differ in the 3rd sig fig); both were rounded to "5.6e-7."
3. **The p itself is pseudoreplicated.** MWU treats each *trial* as independent, but trials cluster
   within *cells* (the independent unit). Subsampling the same effect to 25 %/5 % of trials moves the
   p to 0.025 / 0.13 — i.e. **the p is trial-count-driven, not effect-driven.** It massively overstates
   significance.
4. **Honest inference** = the **cell-clustered bootstrap CI**, which the code already computes. By that
   measure IBL-MRN is **marginal** (CI lower bound +0.003) and Steinmetz-MRN is real (CI excludes 0).

**Mandate**: do **NOT** present "same p=5.6e-7" as a cross-dataset replication coincidence. Report
the cell-clustered CIs; compare MRN-to-MRN; downgrade the MWU p to a descriptive (pseudoreplicated)
statistic. The qualitative replication (a small real MRN decision effect, well below the 0.24 SD gate)
**survives** this correction — but the rhetorical "identical numbers" does not.

---

## PART D — cross-cutting

- **Multiple comparisons / forking paths.** Per-analysis BH-FDR is applied, but there is **no paper-level
  correction** across the many regions × windows × thresholds × datasets, and several decisive choices
  are post-hoc: the 50 %-contrast session drop, equal-vs-error decorrelation, full-vs-≤25 % cascade
  contrast, the 0.57 gate (read off a pre-flight, then used to judge the data), region pooling (all vs
  MRN). **Nothing is pre-registered**, and the final analyses differ from the first attempts (the
  Steinmetz error→equal switch, the cascade contrast revert). This is a garden-of-forking-paths exposure;
  the defence is that the *qualitative* conclusions are over-determined and the changes are disclosed.
- **Reproducibility.** Good: nearly every script seeds `np.random.default_rng(0)` (or explicit seeds),
  pinned `requirements.txt`, incremental CSV checkpoints. Gaps: **no one-command figure-regen / Makefile**;
  no session/env manifest emitted with results; `steinmetz_features.csv` (15 MB) is a regenerable
  intermediate sitting in tracked `results/`; the two-engine ambiguity defeats "one canonical pipeline."
- **Known-undone exposure ranking:**
  - **#4 canonical-engine validation — HIGHEST.** Everything (boundary + 0.57 gate) rests on an
    unvalidated ramp model; a reviewer who knows Latimer/Zoltowski will hit this first.
  - **#5 nonlinear/multidimensional movement control — HIGH.** The headline decision decode's magnitude
    is directly exposed; choice = wheel direction makes this the most likely "it's just movement" attack.
  - **#6 robustness panels — MEDIUM.** σ (the stated main knob), window, bins, FR floor, lead/lock margins,
    engagement, FDR α are all single-valued; one-way sensitivity is currently absent.

---

## PART E — prioritized fix list + honest overall verdict

### Prioritized fix list (ranked by how badly a reviewer can hurt us)
1. **PARTLY DONE (`docs/ramp_validation.md`).** Ramp **forward likelihood now validated locally to
   machine precision** (brute-force enumeration + `hmmlearn.PoissonHMM` at K=50, abs-diff 0.0) — the
   flank is closed at the likelihood level. Two engines (point-weight vs cell-integration) are
   inference-equivalent; grid K robust. **New caveat quantified:** the exact recovery thresholds + the
   0.57 gate are **σ-conditional** — they reproduce at the headline σ=0.4 but the boundary widens for
   steeper ramps (σ=0.7→1.0; true-ramp recovery → chance at σ=1.0). **Scoped out (not in this repo):**
   the literal canonical-accumulator (`ssmdm.Accumulation`) model-equivalence + a published-number
   reproduction would require ssm/ssmdm, which do not build on this box.
2. **DONE (corrections pass):** pooled effect re-framed to cell-clustered bootstrap CI, MRN-to-MRN
   (IBL +0.059 [+0.001,+0.117] *marginal* vs Steinmetz +0.095 [+0.050,+0.139]); pseudoreplicated MWU p
   dropped; `IBL_EFFECT_SD` 0.085→0.059. Also DONE: FDR'd cascade (`triple`→0 everywhere) and gate-
   debiasing consistency (bar→0.545 debiased; SD route 0.095 vs 0.24 is robust). See `audit_corrections.md`.
3. **Nonlinear/expanded movement control** on the headline decode (wheel², |wheel|, interactions, DLC
   velocity; or a movement-GLM residual): show the decision decode survives. (Tier-2 #5)
4. **Make the gate debiasing identical** between where 0.57 is defined and where 0.535 is measured (one
   estimator, same trial count, same clip policy). (B7)
5. **Resolve the two-engine problem**: choose one, run the brute-force forward test on it, archive the
   other, confirm the recovery map matches. (B0/B1)
6. **DONE (corrections pass):** decode permutation test re-specified — fixed-permutation null with
   matched reps (was under-dispersed *and* floored); decode survives (IBL-MRN p=.0025, Steinmetz-MRN
   p=.0075) but the AUC is overfitting-inflated (null p95≈0.73–0.81) → report relative-to-null, and
   ideally replace pseudo-trial resampling with real-trial cross-validation. (B4, `audit_corrections.md`)
7. **FDR-control (or label as raw) the cascade `triple` chain** and the lead/lock/move/stim survival
   counts. (B3)
8. **Robustness panels** (one-way sensitivity): ramp σ∈{0.2,0.4,0.7}, bin∈{10,25,50} ms, FR floor
   ∈{5,10,20}, RT bounds, engagement definition, FDR α. (Tier-2 #6)
9. **Add the missing confound controls / disclosures**: choice-history, engagement, RT on decorrelated
   trials. (B6)
10. **Repro hygiene**: one-command `make figures`, emit an env/session manifest with each result,
    gitignore the 15 MB `steinmetz_features.csv`, write the missing `manuscript_draft_v1.md` Methods from
    Part A. (D)

### Honest overall verdict  *(revised after the corrections pass)*
- **Rock-solid:** the *qualitative identifiability arc*. Step-vs-ramp is **not** identifiable at IBL
  single-session scale (over-determined by Phase-1b's thin budget + any reasonable engine); the stepping
  forward algorithm is correct (hmmlearn 2.3e-13); the population step-vs-ramp arm is **gated** (real
  simultaneity + thin trials); Steinmetz coverage facts (MRN well-covered, GRN/IRN absent).
- **DOWNGRADED — the main empirical positive does NOT hold as stated:** the "**strong distributed,
  per-cell-cryptic MRN decision code**" was **pseudo-trial overfitting + residual movement**. Per-cell
  signal is **FDR-null**; the decode's 0.79–0.92 collapses to **0.60–0.65 on real trials** and to
  **marginal (p=.057/.095) under a nonlinear movement control** (`audit_corrections.md` §6). What
  survives is a **weak, movement-entangled MRN choice signal** + a **small pooled effect** (cell-clustered
  CI [+0.050,+0.139]) — honest but not headline-strength.
- **Still vulnerable / unfixed:** (i) the ramp **forward likelihood is now validated** (brute-force +
  hmmlearn, machine precision) and the boundary is engine/grid-robust, BUT the exact thresholds + 0.57
  gate are **σ-conditional** (validate at σ=0.4, shift for steeper ramps) and the **literal canonical-
  accumulator equivalence is still pending the Colab notebook** (`docs/ramp_validation.md`); (ii) IBL real-trial decode does
  not separately remove stimulus/prior (Steinmetz equal-contrast is clean and is the decisive result);
  (iii) the expanded movement model risks some over-removal on thin per-session data (truth between the
  linear-sig and nonlinear-n.s. results).

**Bottom line (revised):** the durable contribution is the **identifiability/recovery framework, the
population-gate analysis, and the methods lessons** (pseudo-trial overfitting, FDR-null per-cell signal,
the linear-movement confound). The **empirical "distributed decision code" is weak and movement-fragile**,
not the strong cross-dataset positive originally claimed. The retracted artifacts (decode AUC, "identical
p", raw `triple` counts) must not appear in a paper. Fixes #1–#4 remain; none
needed to make the quantitative claims defensible.

# Tier-1 referee-response analyses — results

**Data snapshot:** IBL public Open Alyx (`openalyx.internationalbrainlab.org`), BWM project tag
`brainwide`; ONE-api 3.5.2, iblatlas 1.0.0; cache table dated 2026-06-13, local data products
cached 2026-06-11. All analyses reuse the published pipeline (no reimplementation), canonical
settings (10 Hz floor, deliberation window, window-mean rate, decorrelated = error + 0%-contrast).

## Reproduce-first anchor checks (my number vs manuscript)

| anchor | manuscript | reproduced |
|---|---|---|
| MRN movement-controlled (expanded) real-trial decode AUC | 0.528 | **0.5279** |
| IBL MRN pooled residual (error trials) | +0.059 SD [0.001,0.117] | **+0.0589 [0.0013,0.1166]** |
| IBL dataset-wide BH cascade | 301→98, 131→0, 59→0 | **301→98, 131→0, 59→0** |
| sampled 35-session all-QC census maxima | MRN 56, SCm 79, global 79 | **MRN 56, SCm 79, global 79** |

## PART 1 — over-correction injection control  (`injection_control.csv/.png`)

Realistic choice↔movement coupling (MRN, decorrelated trials): pooled multiple-R = **0.493**
(median 0.519); movement→choice CV-AUC = 0.642. → realistic ρ = 0.493.

d=0 stays at chance for every ρ (0.48–0.51) → no signal fabricated.

**DECISION NUMBER:** at threshold injection d=0.24 SD, realistic ρ=0.493, the movement-controlled
(EXPANDED) decode = **AUC 0.540 [0.534, 0.547] — PULLED BELOW the 0.57 bar** (uncontrolled 0.594;
only 43% of the above-chance signal retained). In the SD metric the injected 0.24 SD is recovered as
only +0.137 SD — also below the 0.24 bar.

Key diagnostics:
- The EXPANDED (nonlinear, 14-regressor) control over-corrects a **movement-ORTHOGONAL** signal too:
  ρ=0, d=0.24 → 0.546 (retains 53%); ρ=0, d=0.30 → 0.561. This excess loss is overfitting on ~30
  trials, not movement removal.
- The LINEAR control does NOT over-correct the orthogonal signal (ρ=0, d=0.24 → 0.602 ≈ uncontrolled
  0.587) and retains 74% at realistic ρ vs the expanded control's 43%.
- At realistic ρ the expanded-controlled decode never reaches 0.57 for ANY injected d up to 0.30
  (max 0.554) → 0.57 is at/above the ceiling of this statistic for these session sizes.

**Verdict: over-correction objection VALIDATED.** A genuine, threshold-sized, realistically
movement-correlated decision signal lands at AUC 0.540 / +0.137 SD after the published control —
below both bars. The observed 0.528 is therefore where a true threshold signal would also land.

## TASK B — FDR scope  (`fdr_scope.csv`)

Survivor counts (choice / movement-indep / triple), all q<0.05:

IBL:
| scheme | MRN triple | SCm triple | others | total triple |
|---|---|---|---|---|
| dataset-wide BH (published) | 0 | 0 | 0 | **0** |
| within-region BH | 0 | **3** | 0 | **3** |
| hierarchical BB | 0 | **3** | 0 | **3** |

Steinmetz: triple = 0 under every scheme (its stim/prior-independent count is 0 everywhere, so
triple collapses regardless of movement survivors MRN 24 / SC 18 / SNr 7).

**DECISION NUMBER:** the "0 triple-coded" is **burden-sensitive**. Under within-region BH (and
hierarchical Benjamini-Bogomolov, which here selects all regions → same), **3 triple-coded cells
survive — all in IBL SCm, none in MRN.** They are robust high-FR cells (34–63 Hz, choice-AUC
0.79–0.81, all confound p at the 0.0005 permutation floor). Steinmetz stays 0. (Same 3 SCm cells
appear under dataset-wide BH at the 25 Hz floor — Table S1.)

## TASK C — full-cohort all-QC simultaneity census  (`full_census.csv`)

Full IBL BWM coverage of the 5 decision-core regions: 205 insertions / 176 sessions (vs the 35-session
sample). Simultaneous = good-QC units (clusters.metrics.label==1, any FR) summed across all probes in
a session × region.

| region | region-sessions | median | max good-QC | ≥120 good-QC | max all-clusters | ≥120 all-clusters |
|---|---|---|---|---|---|---|
| MRN | 118 | 18 | **91** | 0 | 639 | 76 |
| SCm | 68 | 17.5 | 90 | 0 | 723 | 48 |
| IRN | 33 | 18 | 73 | 0 | 626 | 21 |
| SNr | 23 | 3 | 36 | 0 | 195 | 6 |
| GRN | 17 | 28 | 79 | 0 | 822 | 12 |

**DECISION NUMBER:** under the manuscript's all-QC (good-label) definition, **NO region-session reaches
≥120 simultaneous units (global max = 91, MRN; 0/259 region-sessions).** The full cohort raises the max
from the sampled 79 to 91 — exactly as the sampled-census footnote predicted it might — but nowhere near
120, so the recovery wall HOLDS and is strengthened. Only if one drops the QC filter entirely and counts
every cluster (incl. noise/MUA, no recoverable rate signal) does the count exceed 120 (163/259, max 822).

## TASK D — exact numbers for manuscript text fixes

1. **Decode p per setting** (Table S1 sweep): floors 10/15/25 Hz → p 0.057/0.035/0.152 (AUC 0.528/0.536/0.524);
   windows early/delib/peri → 0.170/0.057/0.040 (0.514/0.528/0.533); bins 10/20/50 ms → 0.065/0.060/0.077
   (0.528/0.528/0.527). **Across-settings range = 0.035–0.170.** The phrase "per-session p≈0.06–0.10" is
   NOT a separate per-session computation — it is the two datasets' MRN expanded-movement real-trial decode
   p: **IBL 0.057 / Steinmetz 0.095**. Correct the text to either the across-settings range (0.035–0.170)
   or the two-dataset MRN values (0.057, 0.095), not both conflated.
2. **Fig 3a peri vs pre** (n=1620 hi-FR): pre 256 (15.8%) vs peri 488 (30.1%). Two-proportion z-test
   z=9.69, p≈2×10⁻²². McNemar paired (correct test — same cells; peri-only 354, pre-only 122) χ²=112, p≈3×10⁻²⁶.
3. **Body-pose increment:** wheel-only 161 → wheel+pose 131; the 30 removed = **18.6% of the 161 wheel-survivors**
   (NOT of 301). Confirmed.
4. **AUC mapping:** population movement-controlled decode AUC = 0.528; Gaussian SD→AUC Φ(0.06/√2)=**0.517**.
   "≈0.06 SD, about AUC 0.53" conflates two statistics: the SD-map gives ≈0.52, the population decode ≈0.53.
5. **Most-generous shortfall:** MRN pooled effect across FR floors {10:0.059, 15:0.070, 25:0.133}; largest
   = 0.133 SD; 0.24/0.133 = **1.8×** (not "two or more").

## Bottom line

The central claim ("the controlled decision signal is real-but-below-the-recoverability-bar; action not
deliberation") **does not survive as stated and needs reframing.** Part 1 shows the published control —
specifically the EXPANDED/nonlinear control behind the 0.528 anchor — cannot preserve a genuine
threshold-sized, movement-correlated decision signal above 0.57 (it lands at 0.540), and over-corrects
even movement-orthogonal signal via overfitting; so "below threshold" is confounded by the control, not
evidence against a decision signal. Task B shows "0 triple-coded" is an artifact of dataset-wide BH: 3
SCm cells survive within-region. The robust, defensible results are the small MRN pooled effect (Task A,
CI just clears 0) and the population recovery wall (Task C: full-cohort good-QC max 91 < 120). The
identifiability-framework and methods contributions stand; the "action not deliberation" headline does not.

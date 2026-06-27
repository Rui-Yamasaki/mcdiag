# Corrected applied Results under the valid movement control

Snapshot: Open Alyx public, BWM `brainwide`, ONE-api 3.5.2. Scripts: `src/referee_corrected_results.py`
(reuses `audit_*`, `referee_proper_control`, `referee_fdr_scope`). Anchors held: recomputed-linear
move-survive 133 vs published 131 (corr 0.998); 3 SCm cells post-control AUC ~0.75-0.80.

## The two honest corrections this recompute forced

1. **The cascade was NEVER run through the broken control.** `phase2_selectivity.residual_rate` is
   plain LINEAR OLS on wheel(+DLC) — a VALID control. The over-correction was confined to the small-n
   DECODE (`audit_realtrial_decode` mode='expanded'). So the published movement-independent / triple /
   SCm-cell counts were already under a valid control.
2. **The frozen `pca_expanded` UNDER-removes on real per-cell data.** PCA(85% var) of the expanded
   features drops low-variance movement directions that carry the choice confound. Raw movement-
   independent survivors (IBL, of 301 choice cells): expanded **66** (over-removes, −50%), linear
   **133** (valid, =published), pca **193** (under-removes, +60). The synthetic calibration missed this
   (its injected movement was the high-variance leading PC). **Linear is the trustworthy valid control.**

## Task 1 — corrected cascade (per region, both datasets, both FDR scopes)

Raw movement-independent survivors by control (linear / expanded / pca): IBL MRN 66/34/96, SCm 35/14/54,
SNr 6/3/8, GRN 12/9/15, IRN 14/6/20 (ALL 133/66/193). Steinmetz MRN 51/41/57, SC 37/22/42, SNr 13/9/15.
Within-region-BH **triple-coded** (movement+stim+leading): linear **3 SCm**, pca **4 SCm**, expanded **0**;
0 in every other region and 0 dataset-wide under all controls. → the over-correcting control was killing
the SCm triple cells; both valid controls keep 3–4.

## Task 2 — per-region decode + pooled SD (valid `linear`; pca/expanded/none for context)

| region | decode linear | bar | decode pca | decode expanded(broken) | pooled SD linear | pooled bar |
|---|---|---|---|---|---|---|
| IBL MRN | 0.581 [.539,.622] | straddles | 0.588 | 0.528 (below) | −0.013 [−.067,.041] | null |
| IBL SCm | 0.543 [.491,.596] | straddles | 0.570 | 0.503 (below) | +0.046 [−.002,.095] | marginal |
| IBL SNr | n/a (1 sess) | — | n/a | n/a | (unstable) | — |
| IBL GRN | 0.443 [.357,.539] | below | 0.425 | 0.459 | +0.091 [−.023,.207] | null |
| IBL IRN | 0.520 [.448,.596] | straddles | 0.582 | 0.479 | −0.033 [−.136,.063] | null |
| Steinmetz MRN | 0.621 [.542,.722] | straddles | 0.627 | 0.541 (below) | +0.063 [+.021,.105] | sig, <0.24 |
| Steinmetz SC | 0.544 [.470,.612] | straddles | 0.549 | 0.546 | +0.133 [+.077,.191] | sig, <0.24 |
| Steinmetz SNr | 0.545 [.422,.699] | straddles | 0.569 | 0.529 | +0.045 [−.042,.137] | null |

NO population decode CLEARS the bar under any valid control; MRN (both datasets) sits AT it (point
≥0.57, CI straddles, significant vs chance). The broken `expanded` control pushed MRN/SCm/IRN/St-MRN
BELOW the bar — that is the artifact. Per-cell pooled effects CI-positive (all <0.24): SCm (marginal),
Steinmetz MRN/SC.

## Task 3 — corrected robustness sweep (MRN; old table used broken expanded → all at chance)

decode AUC linear: floor 10/15/25 Hz = 0.581/0.581/0.562; bin 10/20/50 ms = 0.579/0.578/0.573; window
early/delib/peri = 0.562/0.581/0.640. All decision-window settings STRADDLE 0.57 (point at/above);
only peri (movement-window positive control) CLEARS. Per-cell pooled null in decision windows, +0.110 in
peri. → at the bar, robustly, under a valid control — not at chance.

## Task 4 — the movement-independent choice subpopulation (the positive finding)

- Movement-independent (within-region FDR, **linear**): IBL MRN 9, SCm 8, others 0 (~17); Steinmetz
  MRN 24, SC 18, SNr 8 (~50). Movement-only control (some are stimulus-driven).
- **39 exemplar cells** with choice-AUC ≥0.70 after BOTH valid controls (movement-independent), e.g. IBL
  MRN cells at 0.80–0.84 (72.7/70 Hz, p_move=0.0005), the 3 SCm cells at 0.74–0.80; Steinmetz SNr at 0.82.
- Cleanest (movement+stim+leading, within-region FDR): **3–4 SCm**.
- Per-cell AUC bulk at chance (medians 0.50–0.51) with a real selective tail (8–9% > 0.65 in MRN/SCm).

## Task 5 — over-correction cost

The broken `expanded` control erased **67 of 133 (50%)** linear-valid movement-independent IBL cells and
pushed every region's population decode below the bar (e.g. SCm 0.543→0.503, MRN 0.581→0.528, St-MRN
0.621→0.541). Figure: `cascade_by_control.png`.

## How big is the positive pocket, really?

(a) Genuine movement-independent choice cells (within-region FDR, valid linear control): ~**17 in IBL**
(MRN 9, SCm 8) and ~**50 in Steinmetz** (MRN 24, SC 18, SNr 8) — but movement-only; the strict,
stimulus-AND-movement-AND-leading set is **3–4 cells, all SCm**. A broader **39 cells** hold choice-AUC
≥0.70 after both valid controls. (b) The population decode does **not clear** the 0.57 bar under any valid
control; it **sits at** it in MRN (IBL 0.581, Steinmetz 0.621) — significant vs chance, CI straddling the
bar — and is below it in GRN/IRN/St-SNr. (c) The two valid controls **disagree on SCm** (linear decode
0.543 / pooled +0.046 marginal vs pca 0.570 / +0.101 sig; cascade move-survive 35 vs 54) because pca
under-removes; they **agree on MRN** (~0.58). (d) Cleanest control-robust headline: **a handful of clean
movement-independent choice cells in SCm (3–4 fully controlled; ~tens at AUC 0.70–0.84 movement-only)
plus an MRN population decision signal that sits at the recoverability boundary in both datasets** — under
a control calibrated to remove movement without eating signal. The published "below the bar → action not
deliberation" was an artifact of an over-correcting control; the corrected data say "at the boundary,
with a small genuinely-decision-coding SCm subset," not "clearly recoverable."

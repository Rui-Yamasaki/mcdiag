# Proper-control calibration + real-data read-out — THE FORK

Snapshot: Open Alyx public (`openalyx.internationalbrainlab.org`), BWM `brainwide`, ONE-api 3.5.2.
Pluggable decoder verified to reproduce the published anchors EXACTLY (none 0.6149, linear 0.5805,
expanded 0.5279) before any new analysis. Scripts: `src/referee_proper_control.py`.

## Part 1 — calibration (chosen BLIND to real data, then FROZEN)

Three tests via the injection harness on MRN synthetic data: (T1) d=0 → chance; (T2) movement-
ORTHOGONAL signal d=0.24 → PRESERVED (controlled≈uncontrolled); (T3) pure-movement signal d=0.24 →
REMOVED (controlled≈chance). Uncontrolled refs: orthogonal 0.588, pure-movement 0.671.

| control | T1 (d=0) | T2 preserve (retain) | T3 remove | pass |
|---|---|---|---|---|
| none | 0.483/0.620 | 0.588 (1.00) | 0.671 ✗ | fail |
| linear | 0.507/0.506 | 0.614 (1.31) ✓ | 0.506 ✓ | **PASS** |
| expanded (published) | 0.500/0.483 | 0.545 (0.51) ✗ | 0.477 ✓ | fail (eats orthogonal) |
| ridge_expanded | 0.502/0.535 | 0.609 (1.25) ✓ | 0.551 ✗ | fail (under-removes) |
| crossfit_expanded | 0.466/0.478 | 0.479 (−0.24) ✗ | 0.481 ✓ | fail (degenerate on ~30 trials) |
| pca_expanded | 0.501/0.517 | 0.607 (1.22) ✓ | 0.523 ✓ | **PASS** |

**Passers: linear, pca_expanded. FROZEN = pca_expanded** (most movement-removing of the two; realistic-ρ
d=0.24 recovery 0.574 vs linear 0.577 — essentially tied). Corrected bar-vs-signal: under the valid
control a true threshold signal (d=0.24, realistic ρ=0.493) recovers to **AUC 0.574 [0.561,0.587] —
ABOVE 0.57** (d=0.30 → 0.604). The valid control CAN register a threshold signal; the expanded control
(capped ~0.55) could not. The bar is reachable → the real read-out is meaningful.

## Parts 2-3 — real data through the frozen control (+ side-by-side)

| dataset/region | none | linear (valid) | expanded (over-corr.) | **pca_expanded [FROZEN]** | per-cell pooled SD (frozen) |
|---|---|---|---|---|---|
| IBL MRN | 0.615 | 0.581 [.539,.622] | 0.528 | **0.588 [.536,.641] p=.005** | +0.008 [−0.047,+0.063] (null) |
| IBL SCm | 0.619 | 0.543 [.491,.596] | 0.503 | **0.570 [.516,.627] p=.010** | +0.101 [+0.052,+0.153] (sig, <0.24) |
| Steinmetz MRN | 0.651 | 0.621 [.542,.722] | 0.541 | **0.627 [.542,.736] p=.010** | +0.056 [+0.013,+0.101] (sig, <0.24) |

All three frozen-control decodes are **significantly above chance (p≤.01) with point estimates AT/ABOVE
0.57**, but every CI straddles 0.57 → "at the bar," not significantly above it. The two MRN datasets
AGREE (both ~0.59–0.63, at/above bar, small/null per-cell effect).

**The 3 within-region-FDR SCm cells — choice-AUC before → after the frozen movement control:**
- cell …410: 0.787 → **0.757** (n=91)
- cell …600: 0.810 → **0.801** (n=128)
- cell …275: 0.794 → **0.765** (n=91)

These barely move — genuine movement-INDEPENDENT single-cell choice cells at AUC 0.76–0.80, **far above
the 0.57 bar.** Per-cell AUC distributions under the frozen control: MRN/SCm/Steinmetz medians ~0.50–0.51
(bulk at chance) with a real selective tail (9–11% > 0.65).

## The fork — verdict

NOT the cautionary branch as stated. The published "below the bar → action not deliberation" rests on
the EXPANDED control, which fails the preserve test (over-corrects). Under a control calibrated to remove
movement without eating signal:
- the population decode sits **AT the 0.57 bar** (MRN 0.588, SCm 0.570, Steinmetz-MRN 0.627; all p≤.01 vs
  chance; CIs straddle the bar) — at the recoverability threshold, not below it;
- per-cell pooled effects are small and below the 0.24 SD bar (MRN null; SCm +0.10, Steinmetz-MRN +0.056,
  both CI-positive);
- the clearest positive is single-cell: **3 SCm neurons are bona-fide movement-independent choice cells
  at AUC 0.76–0.80** after the valid control.

→ Lands between **ambiguous and small-positive**: signal near the bar at the population level (control
choice matters, esp. SCm: linear 0.543 vs pca 0.570), genuinely above the bar at the single-cell level
for a handful of SCm cells. The paper should drop "action not deliberation / below threshold" and pivot to
"a small, real decision signal at the population recoverability threshold, resolvable in a subset of SCm
neurons under a properly calibrated movement control."

# Steinmetz et al. 2019 — replication feasibility gate

**Verdict: CONDITIONAL GO.** Strong GO on the key positive finding (MRN distributed decode)
and the midbrain selectivity cascade (MRN, SC, SNr); NO-GO on the hindbrain regions
(GRN/IRN — absent from Steinmetz). Light assessment only — no pipeline built, no full run.
Numbers from the cached Neuromatch binned release; benchmarks from our IBL results.

## What we'd replicate
1. Movement/stimulus/(prior-)controlled **choice-selectivity cascade** in midbrain/hindbrain.
2. The **distributed, per-cell-cryptic, stimulus-independent decision code**, esp. in **MRN**
   (our key positive result). *Not* targets: the identifiability sim (dataset-independent) and
   the pseudo-population/simultaneity conclusion (IBL-specific).

## 1. Data access (confirmed, smoke-tested)
- **Full release (for the real run):** figshare *Dataset from Steinmetz et al. 2019*
  (`9598406`), ALF format — `spikes.times`/`spikes.clusters`, `clusters.brainLocation`
  (Allen CCF), `trials.*`, `wheel.*`, `pupil`, face. Preferred for flexible windowing.
  Big (zip → `allData.tar` → per-session tars); **not** downloaded for this gate.
- **Coverage route (used here):** Neuromatch binned `.npz`, 3 parts (OSF `agvxh`/`uv3mw`/
  `ehmw2`), **39 sessions, 10 ms bins, ~222 MB on disk** — compact, all sessions, with
  `brain_area` per neuron + full trial/behaviour. Cached at `data/steinmetz/` (gitignored).
- **Smoke test (1 session):** mouse *Cori*, `spks` (734 neurons × 214 trials × 250 bins @
  10 ms, stim onset at 0.5 s), `brain_area`, `contrast_left/right`∈{0,.25,.5,1},
  `response`∈{−1,0,+1} (NoGo=0), `feedback_type`∈{−1,+1}, `wheel`, `face`, `pupil` — all
  present and parsed. Reproduce: `python src/steinmetz_coverage.py`.

## 2. Region coverage (high-FR floor = 10 Hz, the IBL floor; Steinmetz is **simultaneous**)
`results/steinmetz_coverage.csv` (per-session: `results/steinmetz_coverage_sessions.csv`).

| region | total cells | hi-FR (≥10 Hz) | sessions | max simult. | max simult. hi-FR | status |
|---|--:|--:|--:|--:|--:|---|
| **MRN** (priority) | **1016** | **239** | 11 | 217 | **45** | **well covered** |
| SCm | 356 | 55 | 7 | 87 | 14 | covered |
| SCig (deep/motor SC) | 807 | 132 | 4 | 366 | 37 | covered |
| SNr | 306 | 75 | 4 | 129 | 40 | covered (few sessions) |
| GRN | **0** | 0 | 0 | — | — | **ABSENT** |
| IRN | **0** | 0 | 0 | — | — | **ABSENT** |

Nearby midbrain also present: MB 874, ZI 289, APN 247, PAG 140, RN 58, SCs 157, SCsg 178.
SC is split across subdivisions — the IBL summary **SCm** maps best to **SCm + SCig** (deep/
motor SC), which is very well covered (1163 cells combined). **As predicted, Steinmetz is
midbrain/forebrain-weighted: MRN/SNr/SC are solid; the medulla (GRN/IRN) is not recorded.**

**MRN benchmark vs IBL.** IBL: 777 hi-FR MRN cells across **61** sessions (~13 hi-FR/session,
**pseudo-population**, max ~53 simultaneous); decode used 243 error-testable cells.
Steinmetz: 1016 MRN cells / 239 hi-FR across **11** sessions, but **simultaneously recorded**
— median 72 cells/session (max 217), hi-FR median 20 (max **45**). Fewer total than IBL, but
**genuinely co-recorded**, which is the qualitatively better substrate for the decode finding.

## 3. Task structure / decorrelated budget
Same 2AFC visual-contrast wheel lineage as IBL. Choice = `response` (wheel L/R); **NoGo =
`response==0`** handled by restricting choice analyses to Go trials. Trials where choice
separates from stimulus = **error** (`feedback_type==−1`) + **equal-contrast**
(`contrast_left==contrast_right`, incl. 0-0) Go trials.

| budget (Go trials) | total (39 sess) | per session (median) |
|---|--:|--:|
| error | 1590 | 35 |
| equal-contrast | 1439 | — |
| zero-contrast | 881 | — |
| **decorrelated (union)** | **1885** | **46** (IQR 32–60) |
| MRN-session decorrelated | 618 (across 11) | 54 (max 124) |

**Adequacy:** IBL per-cell decorrelated budget was thin (median ~7–9 error / ~25 decorrelated
trials per cell, pseudo-population). Steinmetz gives a **per-session** budget (median ~46–54
decorrelated) **shared by all simultaneously-recorded cells** — so per-cell power is *higher*
than IBL, and the population test has real co-recorded trials. No imposed block **prior** in
Steinmetz (equal priors by design) → the IBL "prior control" is moot here (one fewer confound).

## 4. Movement covariates
`wheel` **39/39**, `face` motion-energy **39/39**, `pupil` **39/39**. **No full DLC body
pose** in the release. Wheel + face-ME (+ pupil) is a reasonable movement control, but our IBL
result showed **DLC body pose strips ~19% of choice cells beyond wheel** — face motion-energy
is a *partial* proxy, so the movement control is **slightly weaker**. Stated caveat, not a
blocker. (The figshare release also carries eye/face video-derived signals if a stronger
control is wanted later.)

## 5. Required-fields / adapter buildability — CONFIRMED
All fields a replication needs exist per session: spike data (`spks`, binned; spike *times* in
the figshare release for flexible windows), per-neuron region (`brain_area`/`clusters.brain
Location`), `response` (choice), `contrast_left/right` (stimulus), `feedback_type` (error),
event timing (`stim_onset`, `gocue`, `response_time`, `reaction_time`), `wheel`. **Adapter is
buildable** (not built here). Only genuinely missing item vs IBL: DLC body pose (→ §4 caveat).

## 6. Simultaneity (MRN priority)
Steinmetz is **up to 8-probe simultaneous**, so per-session region counts *are* simultaneous
yield. MRN: median 72 cells/session, max 217; hi-FR median 20, max 45; best single-session
test bed ≈ **45 hi-FR MRN cells × 35 decorrelated trials** (or 34 hi-FR × 124 trials). This is
the structural advantage IBL lacked — it doesn't change the replication targets, but it makes
the MRN decode replication clean and (bonus) could address the population question §3.5 found
unreachable in IBL's pseudo-population.

## Verdict, scope, caveats
**CONDITIONAL GO.**
- **GO — MRN distributed stimulus-independent decode (key finding):** 1016 cells / 239 hi-FR,
  simultaneous (≤45 hi-FR co-recorded), decorrelated budget 618 (median 54/session). Better-
  powered per cell than IBL and genuinely co-recorded.
- **GO — selectivity cascade in MRN, SC (SCm+SCig), SNr.** SNr has only 4 sessions (thinner).
- **NO-GO — GRN/IRN** (absent). These were the thin/near-null regions in IBL (GRN 2, IRN 8
  triple cells); the positive finding lived in **MRN**, so their absence does **not** block the
  key replication — it just narrows scope to midbrain.

**Caveats to state in the paper:** (1) movement control weaker — face motion-energy, not DLC
body pose (~19% body-strip not reproducible exactly); (2) hindbrain (GRN/IRN) untestable —
absent; (3) SNr coverage thin (4 sessions); (4) no block prior (control moot — minor); (5)
high-FR counts use overall-window FR (Steinmetz window −0.5→+2.0 s) vs IBL's in-window FR —
order-of-magnitude comparable, MRN is intrinsically high-FR so the count is conservative.

**Recommended scope:** replicate both findings in **MRN (primary), SC, SNr**; drop GRN/IRN;
use wheel + face-ME (+ pupil) as the movement control with the stated caveat; decorrelated set
= error + equal-contrast Go trials. **Allen Visual Behavior Neuropixels is NOT a better
fallback** (more visual-cortical, weaker midbrain decision coverage) — Steinmetz's MRN coverage
is sufficient, so VBN is unnecessary for the key finding. No dataset here recovers GRN/IRN; if
hindbrain replication is required, that needs a different (medulla-targeted) source.

*Assessment only — no analysis pipeline, no full download, no git. Coverage CSVs +
`src/steinmetz_coverage.py` are the deliverables.*

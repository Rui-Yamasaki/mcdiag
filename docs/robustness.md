# Robustness sweep — one-at-a-time sensitivity (IBL, cached data only)

No canonical results file modified; all cells under `results/robustness/`. Reproduce-before-innovate gate passed (default == canonical: triple 59 raw / 0 FDR, MRN pooled +0.059, MRN decode 0.528).

## Feasible knobs swept (from cache)

- **fr_floor** {10,15,25} Hz (filter `fr_window`; can only raise >=10) — all 3 analyses.
- **window** {delib,early,peri} (cached `rate_delib/early/peri`) — pooled + decode + a coherent window-cascade (choice-AUC-p in that window & movement & stimulus, FDR'd).

## Knobs NOT swept (require spike re-extraction = re-download, forbidden)

- **bin_size**: 3 analyses use a window-MEAN rate, not binned counts; binning is a recovery-sim knob.
- **deliberation_window**: features CSV has no per-trial event times; re-windowing needs spikes (deleted).
- **rt_engagement**: features CSV has no per-trial RT; re-filtering on RT/engagement needs spikes (deleted).

## Cells

| knob | value | triple_raw | triple_FDR | triple_FDR/region | MRN pooled SD | MRN pooled CI | MRN pooled p | MRN decode AUC | MRN decode p |
|---|---|--:|--:|---|--:|---|--:|--:|--:|
| fr_floor | 10 | 59 | 0 | {'GRN': 0, 'MRN': 0, 'SNr': 0, 'SCm': 0, 'IRN': 0} | +0.0589 | [0.0013, 0.1166] | 0.0338 | 0.5279 | 0.0574 |
| fr_floor | 15 | 43 | 0 | {'GRN': 0, 'MRN': 0, 'SNr': 0, 'SCm': 0, 'IRN': 0} | +0.0697 | [0.0041, 0.1351] | 0.0242 | 0.5356 | 0.0349 |
| fr_floor | 25 | 33 | 3 | {'GRN': 0, 'MRN': 0, 'SNr': 0, 'SCm': 3, 'IRN': 0} | +0.1327 | [0.057, 0.2098] | 0.0003 | 0.5239 | 0.1521 |
| window | early | 60 | 0 | {'GRN': 0, 'MRN': 0, 'SNr': 0, 'SCm': 0, 'IRN': 0} | +0.0006 | [-0.0587, 0.0598] | 0.4677 | 0.5145 | 0.1696 |
| window | peri | 53 | 0 | {'GRN': 0, 'MRN': 0, 'SNr': 0, 'SCm': 0, 'IRN': 0} | +0.1836 | [0.1219, 0.2452] | 0.0000 | 0.5333 | 0.0399 |

## Verdict

1. **Per-cell decision code FDR-null** (triple_FDR==0 in every region, every cell): **BROKEN**
2. **MRN pooled effect small & positive** (|SD|<0.15, SD>0, every cell): **BROKEN**
3. **MRN decode below the bar** (AUC<=0.56, every cell): **STABLE**

### Flags (loud — these are findings)

- **BREAK conclusion-1 (FDR-null) at fr_floor_25: triple_FDR=3 {'GRN': 0, 'MRN': 0, 'SNr': 0, 'SCm': 3, 'IRN': 0}**
- **BREAK conclusion-2 (pooled small+positive) at window_peri: SD=+0.1836**

### Interpretation

- **MRN (the headline region) is FDR-null at EVERY cell** (MRN triple_FDR==0 across all fr_floor and all windows): True. The conclusion-1 break is **SCm at fr_floor=25 Hz (3 cells)** — raising the floor shrinks the cell set (1620->767), lightening the BH-FDR burden so a few SCm cells survive. So 'per-cell signal is completely FDR-null' is **count/FR-floor-sensitive**; the MRN-specific FDR-null is robust.
- **The conclusion-2 break is the movement-locked PERI window (+0.184 SD)** — a positive control: choice==wheel-turn direction, so a movement-window rate measures movement, not decision. In the decision windows (delib +0.059, early ~0) the pooled effect stays small+positive: True. The effect also grows with FR floor (+0.059->+0.070->+0.133 at fr 10/15/25; +0.133 nears the 0.15 line at 25 Hz).
- **Conclusion 3 (movement-controlled decode) is STABLE at every cell** (AUC 0.514-0.536 <= 0.56), INCLUDING the peri window — the nonlinear movement control removes the movement leakage that inflates the uncontrolled pooled effect there. This is the strongest result and validates the movement control.

**Net:** at the **canonical settings** (deliberation window, fr_floor=10 Hz) all three hold. The breaks are at **non-canonical settings** and are interpretable (movement window = a movement positive-control; 25 Hz floor = lighter FDR). The movement-controlled MRN decode conclusion is robust across the entire feasible sweep.

*Caveat: bin_size / deliberation-window / RT-engagement could not be swept from cached data (spikes deleted post-extraction; re-download forbidden) — see above.*

---

## Bin-size sweep (spike RE-EXTRACTION — authorized one-off, target regions only)

Resolves the `bin_size` line of the caveat above. The canonical feature is a window-**mean**
rate `count([a,b]) / (b−a)` with **no temporal binning** (`phase2_selectivity.py::_rate`), and all
three headline analyses consume only that per-trial scalar. A window-mean is mathematically
bin-**invariant** — summing spike counts over 10/20/50 ms sub-bins tiling `[a,b]` gives the
identical total, hence the identical mean — so the **only** channel by which a bin size can move
the result is **window-edge truncation**.

We re-streamed spikes for the exact canonical cohort (**1620 hi-FR units / 91 sessions /
105 insertions**, target regions only; **spikes only** — movement covariates `wheel_speed/disp`,
`paw/nose` DLC speed and trial labels are window-means/labels → bin-invariant → reused verbatim
from cache), recomputed the per-trial rate at bin sizes {canonical, 10, 20, 50 ms} in a single
stream per insertion (spikes deleted per session, disk bounded), and re-ran cascade / pooled /
decode at each. Binned rate (complete-bins-only): `K = floor((b−a)/w); rate = count([a, a+K·w]) /
(K·w)`; the `<w` remainder at the window end is dropped. `peri` (0.2 s) is an exact multiple of all
three bins → never truncated; `delib` (≥0.5 s) and `early` (≥0.2 s) lose ≤ w at the window end.

**Re-extraction reproduce gate (CRITICAL): PASS.**

- *Feature-match* (re-streamed canonical window-means vs cached `phase2_sel_features_full.csv`):
  max `|abs|` ≤ **5.7e-14**, max `|rel|` ≤ **3.0e-16**, **0 NaN-mismatches** across all 1620 cells ×
  {delib, early, peri} = machine precision (`binsize_feature_match.json`).
- *Headline reproduction at the canonical bin*: triple_raw **59**, triple_FDR **0** (all regions);
  MRN pooled **+0.059 SD** [+0.001, +0.117] p=0.034; MRN decode AUC **0.530** p=0.055 — matches the
  locked canonical numbers (triple 59/0; pooled +0.059; decode 0.528). The decode 0.530 vs 0.528 is
  rank-statistic quantization from the ~1e-14 re-stream perturbation, not a real shift.

### Bin-size table

| knob | value | triple_raw | triple_FDR | triple_FDR/region | MRN pooled SD | MRN pooled CI | MRN pooled p | MRN decode AUC | MRN decode p |
|---|---|--:|--:|---|--:|---|--:|--:|--:|
| bin_size | canonical | 59 | 0 | {GRN:0, MRN:0, SNr:0, SCm:0, IRN:0} | +0.0589 | [+0.0013, +0.1166] | 0.0338 | 0.5295 | 0.0549 |
| bin_size | 10 ms | 57 | 0 | {GRN:0, MRN:0, SNr:0, SCm:0, IRN:0} | +0.0639 | [+0.0066, +0.1212] | 0.0236 | 0.5284 | 0.0648 |
| bin_size | 20 ms | 56 | 0 | {GRN:0, MRN:0, SNr:0, SCm:0, IRN:0} | +0.0499 | [−0.0086, +0.1060] | 0.0726 | 0.5278 | 0.0599 |
| bin_size | 50 ms | 49 | 0 | {GRN:0, MRN:0, SNr:0, SCm:0, IRN:0} | +0.0476 | [−0.0096, +0.1042] | 0.0925 | 0.5269 | 0.0773 |

### Verdict (canonical-anchored)

1. **Per-cell decision-code FDR-null** (triple_FDR==0 in every region, every bin — incl. MRN): **STABLE**
2. **MRN pooled effect small & positive** (point estimate 0 < SD < ~0.15 in the deliberation window, every bin): **STABLE**
3. **MRN movement-controlled decode below the bar** (AUC ≤ ~0.56, every bin) — *the headline*: **STABLE**

### Flags (loud)

- **The decode-at-chance HEADLINE is BIN-INVARIANT.** MRN real-trial CV-AUC under the nonlinear
  (expanded) movement control = **0.530 / 0.528 / 0.528 / 0.527** at canonical / 10 / 20 / 50 ms
  (monotone but Δ < 0.003), always ≤ 0.53 ≪ 0.56, p always ≥ 0.055. The strongest conclusion is the
  most robust knob-to-knob.
- `triple_raw` drifts **59 → 57 → 56 → 49** as bins coarsen: coarser bins truncate more of the short
  `early` window, dropping a few borderline "leading" cells from the RAW (uncorrected) chain. The
  **FDR-corrected count stays 0 at every bin** — the multiple-comparison-controlled conclusion is
  untouched. (The raw count is the expected bin-sensitive quantity; it is not a reported result.)
- MRN pooled effect: the point estimate stays small + positive (+0.048…+0.064 SD) at every bin, but
  the 95% CI lower bound **crosses 0 at 20 ms and 50 ms** ([−0.009, …]). So CI-significance is
  fragile (sig at canonical/10 ms, n.s. at 20/50 ms) while the *magnitude* conclusion (small,
  ≪ 0.15 SD) is robust — coarser bins make the already-marginal effect **weaker, not larger**; no
  spurious inflation, no break of "small."

### Interpretation

Bin size cannot move a window-mean rate except by edge truncation, so bin-invariance is expected
*by construction*; the sweep confirms the residual edge-truncation channel is small, acts mainly on
the short `early` window (hence the raw-cascade drift), and never reverses a canonical conclusion.
At the **canonical settings** all three hold, and across 10–50 ms bins the decision-relevant
headline — MRN decision activity decodes at chance once movement is nonlinearly controlled — is
unchanged.

*Outputs: `results/robustness/binsize_{canon,b10,b20,b50}.json`, `results/robustness/binsize_summary.csv`,
`results/robustness/binsize_feature_match.json`. Re-extraction: `src/robustness_binsize.py`
(spikes-only, resumable per-(pid) checkpoint in `binsize_cache/`, heartbeat `binsize_run.log`).
No canonical results file modified; cached feature CSVs byte-identical afterward.*

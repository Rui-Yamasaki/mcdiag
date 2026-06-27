# Extended Data Table 1. Step-vs-ramp recovery grid (firing rate x pooled trials).

Recovery rate for the step-vs-ramp model-selection test, by mean firing rate (rows) and pooled trial
count N (columns), at the canonical ramp diffusion sigma = 0.4. Recovery is the fraction of synthetic
units whose true generator (step or ramp) is correctly recovered. Two emission models are shown: the
realistic negative-binomial (overdispersed, Fano >= 1.5; the regime of real spikes, Fig 2a) and the
clean-Poisson best case (the lenient upper bound, Extended Data Fig 1). Cells at or above the 0.80
recovery threshold are **bold**; the last column gives the smallest N that reaches 0.80 in that row.

### Negative-binomial emission (realistic; Fig 2a)

| mean FR (Hz) | N=20 | N=40 | N=80 | N=160 | N=320 | first N reaching 0.80 |
|---|---|---|---|---|---|---|
| 2 | 0.467 | 0.542 | 0.525 | 0.471 | 0.500 | -- |
| 5 | 0.533 | 0.475 | 0.529 | 0.583 | 0.625 | -- |
| 10 | 0.579 | 0.583 | 0.546 | 0.654 | 0.704 | -- |
| 20 | 0.596 | 0.562 | 0.633 | 0.729 | **0.800** | 320 |
| 40 | 0.604 | 0.750 | 0.742 | **0.854** | **0.879** | 160 |

### Clean-Poisson emission (best case; Extended Data Fig 1)

| mean FR (Hz) | N=20 | N=40 | N=80 | N=160 | N=320 | first N reaching 0.80 |
|---|---|---|---|---|---|---|
| 2 | 0.517 | 0.608 | 0.533 | 0.458 | 0.558 | -- |
| 5 | 0.567 | 0.533 | 0.517 | 0.508 | 0.658 | -- |
| 10 | 0.558 | 0.550 | 0.608 | 0.667 | **0.808** | 320 |
| 20 | 0.542 | 0.683 | 0.792 | **0.842** | **0.850** | 160 |
| 40 | 0.658 | **0.817** | **0.850** | **0.900** | **0.975** | 40 |

**Reading the grid.** At the IBL single-unit operating point (about 20 Hz, about 40 trials) neither
emission recovers the generator: realistic NB recovery is 0.562 and needs pooling to
N = 320 to clear 0.80, while the lenient clean-Poisson case is 0.683 and clears 0.80 at
N = 160 (about half the NB trial count). The grids confirm that overdispersion, not just trial
count, sets the recovery floor.

Sources:
- Negative-binomial block: results/phase1_recovery_nb_grid_R40.csv (column recovery; rows with
  fano >= 1.5 averaged over fano in {1.5, 2, 3} and over both generators; R = 40 repeats). sigma = 0.4 is
  fixed for the whole file (no sigma column).
- Poisson block: results/phase1_recovery_sweep.csv (column correct; averaged over both generators).
  sigma = 0.4 is fixed for the whole file (no sigma column).
Cross-reference: Fig 2a (NB) and Extended Data Fig 1 / former Supplementary Fig 1 (Poisson).

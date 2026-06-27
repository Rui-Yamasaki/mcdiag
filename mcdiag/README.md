# mcdiag

A movement-control diagnostic for neural decision-signal analyses.

Many studies that look for a "decision signal" in spiking activity first regress out movement, then
decode choice from the residual. That step is where conclusions are won or lost. A control that is
too flexible overfits and removes genuine signal (over-correction), so a real decision signal looks
absent. A control that ignores low-variance movement leaves confound in place (under-removal), so a
movement artifact looks like a decision signal. mcdiag tells you which controls are valid for your
own data, and whether your recording can resolve the question at all.

## The one idea

The calibration injects test signals built from your own movement regressors, preserving their real
variance structure. This is the property that makes the tool worth running. A naive calibration that
injects movement only along the dominant (high-variance) movement direction will bless a control that
removes high-variance movement, even when that control leaves low-variance confound untouched. We saw
exactly this: a PCA-reduced control passed a leading-direction calibration but under-removed real
confound that lived in low-variance movement dimensions. mcdiag draws the injected movement signal
from the full movement covariance (a random projection of the movement features, resampled every
repeat), so it exercises low-variance directions too and catches under-removal. Calibrating against a
synthetic movement structure that does not match your data is the mistake this tool exists to avoid.

## Install

```
pip install -e .            # from this directory
# optional extras
pip install -e ".[test]"    # pytest
pip install -e ".[pandas]"  # pandas for csv convenience
```

Dependencies: numpy, scipy, scikit-learn (pandas optional).

## Quickstart, full-data mode

```python
import mcdiag

# activity: [n_cells, n_trials] rates or counts in the analysis window
# choice:   [n_trials] binary
# movement: [n_trials, n_features] the regressors you would control for
res = mcdiag.calibrate_controls(activity, choice, movement)
print(res)                       # full report
print(res.recommended)           # e.g. "linear"
```

For pseudo-population data (cells from many sessions), pass lists of co-recorded blocks:

```python
res = mcdiag.calibrate_controls(list_of_activity, list_of_choice, list_of_movement)
```

## Quickstart, summary-stats recoverability mode

No spike data needed, just the numbers that describe your recording.

```python
import mcdiag
r = mcdiag.recoverability_check(firing_rate_hz=12, n_trials=200, n_simultaneous_cells=91)
print(r)     # per-cell distinguishable yes or no, population recoverable yes or no, with gaps
```

## Failure modes this tool separates

| control | failure mode | what the calibration sees |
|---|---|---|
| expanded | over-correction by overfitting | fails the preserve test, eats a movement-orthogonal signal at low trial counts |
| pca | under-removal of low-variance confound | fails the remove test, leaves a covariance-preserving pure-movement signal |
| crossfit | degeneracy at few trials | unstable residuals on small inner folds, fails preserve or no-signal |
| linear | the validated middle | removes all linear movement, preserves orthogonal signal, recommended when it passes |

The point is not that one control is always right. It is that the right control depends on your trial
count and your movement covariance, and the calibration measures which one is right for your data.

## Worked example

```
python examples/synthetic_demo.py
```

This builds one synthetic recording (activity encodes movement plus a small genuine choice signal,
movement has both high- and low-variance directions, choice is partly correlated with movement) and
prints the full diagnose report: the recommended control, why each rejected control fails, and the
recoverability verdict.

## Command line

```
mcdiag calibrate --activity act.npy --choice ch.npy --movement mov.npy
mcdiag recover --rate 12 --trials 200 --cells 91
mcdiag diagnose --activity act.npy --choice ch.npy --movement mov.npy --rate 12
```

## Validation

The package ships tests (`pytest`) that reproduce the paper's findings before you trust the tool:

- on the real IBL midbrain (MRN) data fixture it recommends `linear`, flags `expanded` as
  over-correcting and `pca` as under-removing;
- on pure synthetic data with known ground truth the three tests behave correctly and the recommended
  control recovers the injected signal;
- the recoverability check returns not-distinguishable at low rate and trials, distinguishable at
  high, and reports the 91-versus-120-cell population gap for the IBL cohort.

## What the bars mean

A threshold-sized single-trial decision signal is about 0.24 standard deviations of a cell's
across-trial activity, which maps to a per-cell choice AUC of about 0.57. Distinguishing single-trial
step from ramp dynamics at the population level needs on the order of 120 simultaneously recorded
cells. Both bars are derived from the paper's recovery grid and are documented constants you can
override.

## Citation

If you use mcdiag, please cite the paper (placeholder):

```
Yamasaki, R. (2026). Movement controls and the recoverability of single-trial decision signals
in midbrain and hindbrain. [journal, volume, pages, doi to be added].
```

## License

BSD-3-Clause. See LICENSE.

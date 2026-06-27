"""calibrate_controls: choose a valid movement control for the user's own data.

Runs three ground-truth tests, each by injecting a known signal built from the user's real
movement covariance (see injection.py) and passing it through every candidate control:

  no-signal test    inject d = 0. A valid control must return chance for every entanglement.
  preserve test     inject a movement-orthogonal signal at the threshold size. A valid control
                    must keep it (controlled accuracy close to uncontrolled). Failing here means
                    the control eats genuine signal, usually by overfitting at this trial count.
  remove test       inject a pure-movement signal at the threshold size. A valid control must
                    reduce it to chance. Failing here means the control leaves residual movement,
                    usually because it ignores low-variance movement directions.

A control is valid only if it passes all three. Valid controls are ranked by how much real
movement they remove (the remove-test score), and the most-removing valid control is recommended.
If nothing passes, the result says so and recommends collecting more trials per session.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import injection as inj
from .controls import CONTROL_NAMES, cv_decode

#: threshold-sized per-cell decision signal (about 0.57 choice AUC, see recoverability.py)
THRESHOLD_D = 0.24
#: a no-signal decode this far from 0.5 counts as a false positive. This is a finite-sample
#: tolerance: a single small block leaves the held-out no-signal AUC a few hundredths off 0.5 by
#: chance, so the tolerance is set above that floor. Aggregating blocks tightens the estimate.
CHANCE_TOL = 0.045
#: a preserve-test control must keep at least this fraction of the uncontrolled above-chance signal
PRESERVE_MIN_RETAIN = 0.85
#: a remove-test control must bring a pure-movement signal within this of chance
REMOVE_TOL = 0.05


@dataclass
class ControlResult:
    name: str
    no_signal_auc: float
    preserve_auc: float
    preserve_retain: float
    remove_auc: float
    realistic_auc: float
    passes_no_signal: bool
    passes_preserve: bool
    passes_remove: bool
    valid: bool
    reason: str
    under_removes: bool = False


@dataclass
class CalibrationResult:
    controls: dict = field(default_factory=dict)
    recommended: str | None = None
    valid_ranked: list = field(default_factory=list)
    measured_rho: float = 0.0
    realistic_rho_used: float = 0.0
    threshold_d: float = THRESHOLD_D
    n_blocks: int = 0
    n_trials_per_block: tuple = ()
    n_cells_per_block: tuple = ()
    uncontrolled_preserve_auc: float = 0.0
    note: str = ""

    def __str__(self):
        from .report import format_calibration
        return format_calibration(self)


def _normalize_blocks(activity, choice, movement):
    """Accept a single block (arrays) or a list of co-recorded blocks. Returns a list of
    (X [n_trials, n_cells], y [n_trials], movement [n_trials, n_features])."""
    if isinstance(activity, (list, tuple)):
        blocks = []
        for a, c, m in zip(activity, choice, movement):
            blocks.append((np.asarray(a, float).T, np.asarray(c).astype(int), np.asarray(m, float)))
        return blocks
    return [(np.asarray(activity, float).T, np.asarray(choice).astype(int), np.asarray(movement, float))]


def _decode_point(blocks, d, rho, controls, n_repeats, base_seed, n_rep_decode):
    """Mean decode AUC per control at one (d, rho), injecting once per repeat (shared across
    controls) and averaging across blocks and repeats."""
    acc = {c: [] for c in controls}
    for rep in range(n_repeats):
        rng = inj_rng = np.random.default_rng(base_seed + 1009 * rep)
        injected = []
        for (Xbg, _y, M) in blocks:
            Xinj, yv, _ = inj.inject(Xbg, M, d, rho, inj_rng)
            injected.append((Xinj, yv, M))
        for ctl in controls:
            vals = [cv_decode(Xinj, yv, M, ctl, np.random.default_rng(7), n_rep_decode)
                    for (Xinj, yv, M) in injected]
            acc[ctl].append(np.nanmean(vals))
    return {c: float(np.nanmean(acc[c])) for c in controls}


def calibrate_controls(activity, choice, movement, controls=CONTROL_NAMES,
                       inject_sizes=(0.0, THRESHOLD_D), rho_grid=("measured", 0.0, 1.0),
                       n_repeats=12, n_rep_decode=5, seed=2024):
    """Run the three control-validity tests on the user's data and recommend a control.

    Parameters
    ----------
    activity : array [n_cells, n_trials] or list of such arrays (one per co-recorded block)
        Per-trial activity (rates or counts) in the analysis window. For pseudo-population data
        pass a list of co-recorded blocks (sessions); the tests aggregate across them.
    choice : array [n_trials] or list of arrays
        Binary choice per trial.
    movement : array [n_trials, n_features] or list of arrays
        The movement design matrix per trial (the regressors the user would control for).
    controls : sequence of str
        Candidate controls to test (subset of mcdiag.controls.CONTROL_NAMES).
    inject_sizes : sequence of float
        Per-cell injected separations in SD units. Must include 0.0 and the threshold (0.24).
    rho_grid : sequence
        Movement entanglements to test. The literal 'measured' uses the data's actual
        choice-to-movement correlation.
    n_repeats : int
        Synthetic repeats per grid point (more is steadier).

    Returns
    -------
    CalibrationResult
    """
    controls = list(dict.fromkeys(["none", *controls]))   # always include the uncontrolled reference
    blocks = _normalize_blocks(activity, choice, movement)
    rho_real = float(np.mean([inj.measured_choice_movement_corr(y, M) for (_X, y, M) in blocks]))
    rhos = [rho_real if (isinstance(r, str) and r == "measured") else float(r) for r in rho_grid]
    sizes = sorted(set(float(d) for d in inject_sizes) | {0.0, THRESHOLD_D})

    grid = {}
    for d in sizes:
        for rho in rhos:
            grid[(round(d, 6), round(rho, 6))] = _decode_point(
                blocks, d, rho, controls, n_repeats, seed, n_rep_decode)

    def at(d, rho, ctl):
        return grid[(round(d, 6), round(rho, 6))][ctl]

    unc_preserve = at(THRESHOLD_D, 0.0, "none")
    results = {}
    for ctl in controls:
        if ctl == "none":
            continue
        # no-signal (false positive) test uses the movement-orthogonal label (rho = 0): with no
        # signal injected, does the control invent choice structure from movement-orthogonal noise.
        # Movement removal at high entanglement is the separate remove test (rho = 1), so it is not
        # double-counted here.
        no_sig_auc = at(0.0, 0.0, ctl)
        no_sig = abs(no_sig_auc - 0.5)
        preserve = at(THRESHOLD_D, 0.0, ctl)
        retain = (preserve - 0.5) / (unc_preserve - 0.5) if unc_preserve > 0.5 + 1e-6 else np.nan
        remove = at(THRESHOLD_D, 1.0, ctl) if 1.0 in rhos else np.nan
        realistic = at(THRESHOLD_D, rho_real, ctl) if rho_real in rhos else np.nan
        p1 = no_sig < CHANCE_TOL
        p2 = (not np.isnan(retain)) and retain >= PRESERVE_MIN_RETAIN
        p3 = (not np.isnan(remove)) and abs(remove - 0.5) < REMOVE_TOL
        valid = p1 and p2 and p3
        reason = _reason(ctl, p1, p2, p3, retain, remove, no_sig)
        results[ctl] = ControlResult(
            name=ctl, no_signal_auc=float(no_sig_auc), preserve_auc=float(preserve),
            preserve_retain=float(retain) if not np.isnan(retain) else float("nan"),
            remove_auc=float(remove) if not np.isnan(remove) else float("nan"),
            realistic_auc=float(realistic) if not np.isnan(realistic) else float("nan"),
            passes_no_signal=bool(p1), passes_preserve=bool(p2), passes_remove=bool(p3),
            valid=bool(valid), reason=reason)

    valid_ranked = sorted([r for r in results.values() if r.valid], key=lambda r: r.remove_auc)
    recommended = valid_ranked[0].name if valid_ranked else None
    # flag valid controls that remove meaningfully less movement than the recommended one. This is
    # the under-removal signature (for example a PCA control that leaves low-variance confound):
    # the control passes the remove test but a less flexible control removes more.
    if recommended:
        best_remove = results[recommended].remove_auc
        for r in valid_ranked[1:]:
            if r.remove_auc > best_remove + 0.012:
                r.under_removes = True
                r.reason = (f"{r.name}: valid, but removes less movement than {recommended} "
                            f"(remove-test AUC {r.remove_auc:.3f} vs {best_remove:.3f}), can "
                            f"under-remove low-variance confound, so {recommended} is preferred.")
    note = ("" if recommended else
            "No candidate control passed all three tests on this data. This usually means too few "
            "trials per block for any control to both remove movement and preserve signal. Collect "
            "more trials per session rather than picking the least-bad control.")
    return CalibrationResult(
        controls=results, recommended=recommended,
        valid_ranked=[r.name for r in valid_ranked], measured_rho=rho_real,
        realistic_rho_used=rho_real, threshold_d=THRESHOLD_D, n_blocks=len(blocks),
        n_trials_per_block=tuple(len(y) for (_X, y, _M) in blocks),
        n_cells_per_block=tuple(X.shape[1] for (X, _y, _M) in blocks),
        uncontrolled_preserve_auc=float(unc_preserve), note=note)


def _reason(ctl, p1, p2, p3, retain, remove, no_sig):
    if p1 and p2 and p3:
        return f"valid: passes all three tests (removes pure movement to AUC {remove:.3f})."
    bits = []
    if not p1:
        bits.append(f"FAILS no-signal test (returns AUC {0.5 + no_sig:.3f} with no signal injected)")
    if not p2:
        pct = "n/a" if np.isnan(retain) else f"{retain * 100:.0f} percent"
        bits.append(f"FAILS preserve test, retains {pct} of a movement-orthogonal signal, "
                    f"over-corrects by overfitting at this trial count")
    if not p3:
        ra = "n/a" if np.isnan(remove) else f"{remove:.3f}"
        bits.append(f"FAILS remove test, leaves a pure-movement signal at AUC {ra}, under-removes "
                    f"(ignores low-variance movement directions)")
    return f"{ctl}: " + "; ".join(bits) + "."

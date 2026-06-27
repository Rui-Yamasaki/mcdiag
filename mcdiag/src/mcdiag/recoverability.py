"""recoverability_check: can this recording resolve the decision-dynamics question at all.

Two bars from the framework:

  per-cell bar     A threshold-sized single-trial decision signal is about 0.24 standard
                   deviations of a cell's across-trial activity, which maps to a choice AUC of
                   about 0.57 (Phi(0.24 / sqrt(2)) = 0.567). Whether that signal is distinguishable
                   per cell depends on firing rate (low rates make the per-trial count discrete and
                   wash out a sub-spike signal) and on trial count (sampling noise). This is checked
                   by simulating a cell at the given rate and trial count with a spiking model
                   (Poisson by default, or a negative-binomial set by spiking_variability as a Fano
                   factor) and measuring the achievable per-cell AUC.

  population bar   Distinguishing single-trial step from ramp dynamics at the population level
                   requires on the order of 120 simultaneously recorded cells (the recovery grid in
                   the paper). This is a simple comparison of the user's simultaneous-cell count to
                   that requirement.

The check reports the gap to each bar in numbers, not just yes or no.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

#: per-cell effect size of a threshold decision signal, in across-trial SD units
THRESHOLD_EFFECT_SD = 0.24
#: simultaneous-cell requirement for population step-versus-ramp recovery (paper recovery grid)
POPULATION_CELL_REQUIREMENT = 120
#: default deliberation-window length in seconds (the window the per-cell rate is measured over)
DEFAULT_WINDOW_S = 0.5


@dataclass
class RecoverabilityResult:
    firing_rate_hz: float
    n_trials: int
    n_simultaneous_cells: int
    per_cell_auc_bar: float
    achievable_per_cell_auc: float
    per_cell_auc_lo: float
    per_cell_distinguishable: bool
    per_cell_auc_gap: float
    population_requirement: int
    population_recoverable: bool
    population_cell_gap: int
    note: str = ""

    def __str__(self):
        from .report import format_recoverability
        return format_recoverability(self)


def _sample_counts(lam, n, fano, rng):
    """Sample n integer spike counts with mean lam and the given Fano factor (variance/mean)."""
    lam = max(lam, 1e-6)
    if fano <= 1.0 + 1e-6:
        return rng.poisson(lam, n)
    # negative binomial with mean lam and variance fano * lam
    p = 1.0 / fano
    r = lam * p / (1.0 - p)
    return rng.negative_binomial(max(r, 1e-6), p, n)


def recoverability_check(firing_rate_hz, n_trials, n_simultaneous_cells, spiking_variability=None,
                         window_s=DEFAULT_WINDOW_S, effect_sd=THRESHOLD_EFFECT_SD,
                         population_requirement=POPULATION_CELL_REQUIREMENT, n_sim=600, seed=0):
    """Report whether a recording can resolve the per-cell and population decision questions.

    Parameters
    ----------
    firing_rate_hz : float
        Per-cell firing rate in the analysis window.
    n_trials : int
        Trials available per cell (balanced across the two choices is assumed).
    n_simultaneous_cells : int
        Cells recorded simultaneously in the region of interest.
    spiking_variability : float or None
        Fano factor (variance over mean of the spike count). None or 1.0 means Poisson.
    window_s : float
        Analysis window length in seconds (the per-cell rate is a count over this window).

    Returns
    -------
    RecoverabilityResult
    """
    rng = np.random.default_rng(seed)
    fano = 1.0 if spiking_variability is None else float(spiking_variability)
    bar = float(norm.cdf(effect_sd / np.sqrt(2.0)))           # 0.567 for the 0.24 SD threshold
    mu = firing_rate_hz * window_s                            # mean count in the window
    sd_count = np.sqrt(max(fano * mu, 1e-9))
    delta = effect_sd * sd_count                             # injected count difference between choices

    n_each = max(2, int(n_trials // 2))
    aucs = []
    from sklearn.metrics import roc_auc_score
    for _ in range(n_sim):
        c0 = _sample_counts(mu - delta / 2.0, n_each, fano, rng)
        c1 = _sample_counts(mu + delta / 2.0, n_each, fano, rng)
        x = np.concatenate([c0, c1]).astype(float)
        x = x + rng.normal(0, 1e-6, x.size)                  # break exact ties deterministically
        y = np.r_[np.zeros(n_each), np.ones(n_each)]
        aucs.append(roc_auc_score(y, x))
    aucs = np.array(aucs)
    achievable = float(np.median(aucs))
    lo = float(np.percentile(aucs, 5))
    # distinguishable: the rate resolves the signal near the bar AND the trial count makes it
    # reliably above chance
    distinguishable = bool((achievable >= bar - 0.01) and (lo > 0.5))

    pop_recoverable = bool(n_simultaneous_cells >= population_requirement)
    pop_gap = int(population_requirement - n_simultaneous_cells)

    note = []
    if not distinguishable:
        if achievable < bar - 0.01:
            note.append(f"firing rate too low to resolve a {effect_sd} SD signal "
                        f"(ceiling AUC {achievable:.3f} below the {bar:.3f} bar)")
        if lo <= 0.5:
            note.append(f"too few trials to separate the signal from chance "
                        f"(5th-percentile AUC {lo:.3f})")
    if not pop_recoverable:
        note.append(f"{pop_gap} cells short of the {population_requirement}-cell population bar")
    return RecoverabilityResult(
        firing_rate_hz=float(firing_rate_hz), n_trials=int(n_trials),
        n_simultaneous_cells=int(n_simultaneous_cells), per_cell_auc_bar=bar,
        achievable_per_cell_auc=achievable, per_cell_auc_lo=lo,
        per_cell_distinguishable=distinguishable, per_cell_auc_gap=float(bar - achievable),
        population_requirement=int(population_requirement),
        population_recoverable=pop_recoverable, population_cell_gap=pop_gap,
        note="; ".join(note) if note else "both bars met")

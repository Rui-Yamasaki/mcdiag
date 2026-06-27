"""Recoverability check behaves correctly at the extremes and reports the IBL population gap."""
import mcdiag


def test_low_rate_low_trials_not_distinguishable():
    # few trials: the signal cannot be separated from chance (sampling-limited)
    r = mcdiag.recoverability_check(firing_rate_hz=3.0, n_trials=40, n_simultaneous_cells=20)
    assert not r.per_cell_distinguishable
    assert r.per_cell_auc_lo <= 0.5


def test_very_low_rate_ceiling_below_bar():
    # very low rate: even with many trials, spike discreteness caps the achievable AUC below the bar
    r = mcdiag.recoverability_check(firing_rate_hz=1.0, n_trials=2000, n_simultaneous_cells=20)
    assert not r.per_cell_distinguishable
    assert r.achievable_per_cell_auc < r.per_cell_auc_bar


def test_high_rate_high_trials_distinguishable():
    r = mcdiag.recoverability_check(firing_rate_hz=40.0, n_trials=400, n_simultaneous_cells=200)
    assert r.per_cell_distinguishable
    assert r.per_cell_auc_lo > 0.5


def test_population_bar_and_ibl_gap():
    # the IBL cohort full-census maximum is 91 simultaneous good-QC cells vs the 120-cell bar
    r = mcdiag.recoverability_check(firing_rate_hz=30.0, n_trials=300, n_simultaneous_cells=91)
    assert not r.population_recoverable
    assert r.population_cell_gap == 29
    assert r.population_requirement == 120


def test_population_recoverable_when_enough_cells():
    r = mcdiag.recoverability_check(firing_rate_hz=30.0, n_trials=300, n_simultaneous_cells=150)
    assert r.population_recoverable
    assert r.population_cell_gap == -30

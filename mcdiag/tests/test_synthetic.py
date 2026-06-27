"""On pure synthetic data with known ground truth, the three calibration tests behave correctly
and the recommended control recovers the injected signal."""
import numpy as np

import mcdiag


def _synthetic_block(n_cells=24, n_trials=80, seed=1):
    rng = np.random.default_rng(seed)
    scales = np.array([3.0, 1.0, 0.5, 0.3])             # high- and low-variance movement directions
    movement = rng.standard_normal((n_trials, 4)) * scales
    w = rng.standard_normal(4)
    latent = (movement @ w)
    latent /= latent.std()
    choice = (0.6 * latent + 0.8 * rng.standard_normal(n_trials) > 0).astype(int)
    cell_move = rng.standard_normal((n_cells, 4))
    cell_choice = 0.15 * rng.standard_normal(n_cells)
    activity = (rng.standard_normal((n_cells, n_trials)) + cell_move @ movement.T
                + np.outer(cell_choice, choice * 2 - 1.0))
    return activity, choice, movement


def _synthetic_blocks(n_blocks=4, **kw):
    """Several co-recorded blocks, as the tool is used on pseudo-population data (steadier tests)."""
    blocks = [_synthetic_block(seed=10 + i, **kw) for i in range(n_blocks)]
    return ([b[0] for b in blocks], [b[1] for b in blocks], [b[2] for b in blocks])


def test_three_tests_behave_and_recommended_recovers():
    act, ch, mov = _synthetic_blocks()
    res = mcdiag.calibrate_controls(act, ch, mov, n_repeats=8)

    # a control is recommended, and it passes all three ground-truth tests
    assert res.recommended is not None, res.note
    rec = res.controls[res.recommended]
    assert rec.passes_no_signal and rec.passes_preserve and rec.passes_remove

    # no-signal test: chance with nothing injected
    assert abs(rec.no_signal_auc - 0.5) < 0.05

    # remove test: a pure-movement signal is brought to chance
    assert abs(rec.remove_auc - 0.5) < 0.06

    # preserve / recover: a threshold signal at the realistic entanglement is recovered above chance
    assert rec.realistic_auc > 0.53

    # the uncontrolled reference does rise with the injected signal (sanity on the harness)
    assert res.uncontrolled_preserve_auc > 0.55


def test_no_signal_is_chance_for_all_controls():
    act, ch, mov = _synthetic_blocks(n_blocks=4)
    res = mcdiag.calibrate_controls(act, ch, mov, n_repeats=8)
    # at the movement-orthogonal label with nothing injected, no control should invent choice signal
    for name, r in res.controls.items():
        assert abs(r.no_signal_auc - 0.5) < 0.06, f"{name} fabricates signal at d=0"


def test_pca_under_removes_on_low_variance_movement():
    # the core property: with low-variance movement directions present, a PCA control under-removes
    act, ch, mov = _synthetic_blocks(n_blocks=4)
    res = mcdiag.calibrate_controls(act, ch, mov, n_repeats=8)
    pca = res.controls["pca"]
    linear = res.controls["linear"]
    assert pca.remove_auc > linear.remove_auc            # pca leaves more movement than linear
    assert not pca.valid or pca.under_removes            # flagged either way

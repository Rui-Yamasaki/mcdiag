"""Headline validation: on the real IBL midbrain (MRN) data the tool reproduces the paper's
control verdicts. It must recommend linear, flag expanded as over-correcting, and flag pca as
under-removing. This is the reproduce-first gate.

The fixture (tests/fixtures/ibl_mrn.npz) holds the 28 decodable MRN co-recorded blocks
(deliberation-window mean rate, decorrelated trials) from the IBL Brain Wide Map open release
(openalyx.internationalbrainlab.org, brainwide tag, cached 2026-06-11).
"""
from pathlib import Path

import numpy as np
import pytest

import mcdiag

FIX = Path(__file__).resolve().parent / "fixtures" / "ibl_mrn.npz"


@pytest.fixture(scope="module")
def ibl_mrn():
    d = np.load(FIX, allow_pickle=True)
    return list(d["activity"]), list(d["choice"]), list(d["movement"])


@pytest.fixture(scope="module")
def calib(ibl_mrn):
    act, ch, mov = ibl_mrn
    return mcdiag.calibrate_controls(act, ch, mov, n_repeats=10)


def test_recommends_linear(calib):
    assert calib.recommended == "linear"


def test_expanded_over_corrects(calib):
    exp = calib.controls["expanded"]
    assert not exp.valid
    assert not exp.passes_preserve            # eats the movement-orthogonal signal
    assert "over-correct" in exp.reason


def test_pca_under_removes(calib):
    pca = calib.controls["pca"]
    linear = calib.controls["linear"]
    # pca leaves more movement than the recommended linear control (the under-removal signature)
    assert pca.remove_auc > linear.remove_auc
    assert pca.under_removes
    assert "under-remove" in pca.reason


def test_measured_rho_is_reasonable(calib):
    # the IBL MRN choice-to-movement coupling is moderate (about 0.5)
    assert 0.4 < calib.measured_rho < 0.65


def test_linear_passes_all_three(calib):
    lin = calib.controls["linear"]
    assert lin.passes_no_signal and lin.passes_preserve and lin.passes_remove and lin.valid

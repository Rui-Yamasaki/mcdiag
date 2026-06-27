"""Generate the full diagnose report on the real IBL MRN fixture (for documentation/validation).

    python examples/ibl_mrn_report.py
"""
from pathlib import Path

import numpy as np

import mcdiag

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ibl_mrn.npz"

if __name__ == "__main__":
    d = np.load(FIX, allow_pickle=True)
    act, ch, mov = list(d["activity"]), list(d["choice"]), list(d["movement"])
    # representative IBL MRN hi-FR cell rate; the population question uses the cohort max of 91
    # simultaneous good-QC cells (full-census maximum) against the 120-cell bar
    dx = mcdiag.diagnose(
        activity=act, choice=ch, movement=mov,
        firing_rate_hz=20.0, n_simultaneous_cells=91,
        calibrate_kwargs={"n_repeats": 12},
    )
    print(dx)

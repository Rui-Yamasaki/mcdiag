"""diagnose: run the calibration and the recoverability check together and emit one report."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibrate import calibrate_controls
from .recoverability import recoverability_check


@dataclass
class Diagnosis:
    calibration: object = None
    recoverability: object = None

    def __str__(self):
        parts = []
        if self.calibration is not None:
            parts.append(str(self.calibration))
        if self.recoverability is not None:
            parts.append(str(self.recoverability))
        return "\n\n".join(parts)


def diagnose(activity=None, choice=None, movement=None,
             firing_rate_hz=None, n_trials=None, n_simultaneous_cells=None,
             spiking_variability=None, calibrate_kwargs=None, recover_kwargs=None):
    """Run both checks and return a Diagnosis whose str() is the full human-readable report.

    Pass activity/choice/movement to run the control calibration. Pass firing_rate_hz/n_trials/
    n_simultaneous_cells to run the recoverability check. If the summary numbers are not given but
    full data is, they are estimated from the data (mean rate proxy from activity is not assumed,
    so firing_rate_hz should be given in Hz when known).
    """
    cal = rec = None
    if activity is not None and choice is not None and movement is not None:
        cal = calibrate_controls(activity, choice, movement, **(calibrate_kwargs or {}))
        if n_simultaneous_cells is None:
            cells = cal.n_cells_per_block
            n_simultaneous_cells = max(cells) if cells else None
        if n_trials is None:
            tr = cal.n_trials_per_block
            n_trials = int(np.median(tr)) if tr else None
    if firing_rate_hz is not None and n_trials is not None and n_simultaneous_cells is not None:
        rec = recoverability_check(firing_rate_hz, n_trials, n_simultaneous_cells,
                                   spiking_variability=spiking_variability, **(recover_kwargs or {}))
    return Diagnosis(calibration=cal, recoverability=rec)

"""mcdiag: a movement-control diagnostic for decision-signal analyses.

Two questions every lab that regresses out movement should answer before trusting a decision-signal
estimate:

  1. Which movement control is valid for my data? A control that overfits removes genuine signal
     (over-correction); a control that ignores low-variance movement leaves confound (under-removal).
     calibrate_controls injects known signals built from your own movement covariance and tells you
     which controls remove movement without eating signal.

  2. Can my recording resolve the question at all? recoverability_check compares your firing rate,
     trial count, and simultaneous-cell count to the per-cell and population recovery bars.

Public API: calibrate_controls, recoverability_check, diagnose.
"""
from .calibrate import calibrate_controls, CalibrationResult
from .recoverability import recoverability_check, RecoverabilityResult
from .diagnose import diagnose, Diagnosis
from .controls import CONTROL_NAMES

__version__ = "0.1.0"
__all__ = [
    "calibrate_controls", "CalibrationResult",
    "recoverability_check", "RecoverabilityResult",
    "diagnose", "Diagnosis", "CONTROL_NAMES", "__version__",
]

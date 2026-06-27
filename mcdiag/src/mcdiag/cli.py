"""Command line interface: mcdiag calibrate | recover | diagnose.

Reads arrays from .npy or .csv and prints the report. For calibrate and the data side of diagnose
the inputs are a single co-recorded block; for pseudo-population calibration across many blocks use
the Python API (pass lists to calibrate_controls).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .calibrate import calibrate_controls
from .diagnose import diagnose
from .recoverability import recoverability_check


def _load(path):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".csv"):
        return np.loadtxt(path, delimiter=",")
    raise ValueError(f"unsupported file type (use .npy or .csv): {path}")


def _add_calib_args(p):
    p.add_argument("--activity", required=True, help="[n_cells, n_trials] .npy or .csv")
    p.add_argument("--choice", required=True, help="[n_trials] binary .npy or .csv")
    p.add_argument("--movement", required=True, help="[n_trials, n_features] .npy or .csv")
    p.add_argument("--repeats", type=int, default=12, help="synthetic repeats per grid point")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mcdiag", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("calibrate", help="choose a valid movement control for your data")
    _add_calib_args(pc)

    pr = sub.add_parser("recover", help="check per-cell and population recoverability")
    pr.add_argument("--rate", type=float, required=True, help="firing rate (Hz)")
    pr.add_argument("--trials", type=int, required=True, help="trials per cell")
    pr.add_argument("--cells", type=int, required=True, help="simultaneous cells")
    pr.add_argument("--fano", type=float, default=None, help="Fano factor (default Poisson)")

    pd_ = sub.add_parser("diagnose", help="run both and print one report")
    _add_calib_args(pd_)
    pd_.add_argument("--rate", type=float, required=True, help="firing rate (Hz)")
    pd_.add_argument("--cells", type=int, default=None, help="simultaneous cells (default from data)")
    pd_.add_argument("--fano", type=float, default=None, help="Fano factor (default Poisson)")

    a = ap.parse_args(argv)
    if a.cmd == "calibrate":
        res = calibrate_controls(_load(a.activity), _load(a.choice), _load(a.movement),
                                 n_repeats=a.repeats)
        print(res)
    elif a.cmd == "recover":
        print(recoverability_check(a.rate, a.trials, a.cells, spiking_variability=a.fano))
    elif a.cmd == "diagnose":
        act = _load(a.activity)
        res = diagnose(activity=act, choice=_load(a.choice), movement=_load(a.movement),
                       firing_rate_hz=a.rate, n_simultaneous_cells=a.cells,
                       spiking_variability=a.fano, calibrate_kwargs={"n_repeats": a.repeats})
        print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())

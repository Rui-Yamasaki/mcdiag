"""Re-run the NB/realistic recovery grid at a higher replicate count R, writing to a NEW
results/phase1_recovery_nb_grid_R{R}.csv (the R=20 original is left untouched for provenance).

Only R changes — the EXACT deterministic seed scheme (phase1_recovery_nb_grid.cell_seeds) and the
validated engine (recover_canonical) are reused verbatim. The Poisson best-case sweep is already
R=60 in results/phase1_recovery_sweep.csv, so it is NOT regenerated here (re-running would only
reproduce it bit-identically).

  python src/rerun_recovery_highR.py --R 40
"""
from __future__ import annotations

import os
# pin BLAS to 1 thread BEFORE numpy is imported (workers do the CV; avoid oversubscription)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402

import phase1_recovery_nb_grid as G  # noqa: E402  (NB grid generator; we redirect its globals)
from phase1_recovery import load_windows  # noqa: E402
from ibl_one import PROJECT_ROOT  # noqa: E402

RESULTS = PROJECT_ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=40, help="replicates per cell (R=20 original)")
    a = ap.parse_args()

    # redirect outputs to new filenames; bump R ONLY (cell_seeds reads this module global)
    G.R = a.R
    G.OUT_CSV = RESULTS / f"phase1_recovery_nb_grid_R{a.R}.csv"
    G.LOG = RESULTS / f"phase1_recovery_nb_grid_R{a.R}.log"

    windows = load_windows()
    G.hb(f"HIGH-R RERUN: NB grid R={a.R} -> {G.OUT_CSV.name} "
         f"(seed scheme unchanged, only R; N_JOBS={G.N_JOBS}, K={G.K})")
    ok, got, ref = G.run_gate(windows)
    if not ok:
        raise SystemExit(f"STOP: NB reproduce gate FAILED at R={a.R} — not sweeping.")
    G.run_grid(windows)
    G.hb(f"HIGH-R RERUN DONE: R={a.R} -> {G.OUT_CSV}")


if __name__ == "__main__":
    main()

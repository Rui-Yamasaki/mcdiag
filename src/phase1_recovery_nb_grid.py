"""Full FR x N recovery grid under the REALISTIC regime: overdispersed (negative-binomial)
emissions + matched NB fit, at canonical sigma=0.4. This is the honest version of the Fig-2(a)
identifiability map (the cached phase1_recovery_sweep.csv is clean-Poisson, best-case).

Reuses the VALIDATED engine verbatim (phase1_recovery_hardened.recover_canonical -> emit /
model_loglik_nb / fit_step_nb / fit_ramp_nb), same axes as phase1_recovery_sweep.csv, the SAME
Fano set and R as phase1_recovery_matched_nb.csv. No canonical result is overwritten; only the new
results/phase1_recovery_nb_grid.csv is written (resumable per (FR,N) cell, heartbeat per cell).

REPRODUCE GATE (runs first): recompute FR=20, N in {160,320} (NB fit, Fano>=1.5) and require it to
match phase1_recovery_matched_nb.csv (~0.74 and ~0.82). If it does not, STOP before sweeping.

  python src/phase1_recovery_nb_grid.py            # gate -> full grid (resumable)
  python src/phase1_recovery_nb_grid.py --gate     # gate only
"""
from __future__ import annotations

import os
# pin BLAS to 1 thread BEFORE numpy is imported (workers do the CV; avoid oversubscription)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from ibl_one import PROJECT_ROOT  # noqa: E402
from phase1_recovery import load_windows  # noqa: E402
from phase1_recovery_hardened import recover_canonical  # noqa: E402  (validated engine)

# axes match phase1_recovery_sweep.csv exactly; Fano/R/K match phase1_recovery_matched_nb.csv
FRS = [2.0, 5.0, 10.0, 20.0, 40.0]
NS = [20, 40, 80, 160, 320]
FANOS = [1.0, 1.5, 2.0, 3.0]
GENS = ["step", "ramp"]
R = 20
K = 50
N_JOBS = int(os.environ.get("NB_GRID_JOBS", "10"))   # 12-core box; BLAS pinned to 1 above

OUT_CSV = PROJECT_ROOT / "results" / "phase1_recovery_nb_grid.csv"
LOG = PROJECT_ROOT / "results" / "phase1_recovery_nb_grid.log"
MATCHED = PROJECT_ROOT / "results" / "phase1_recovery_matched_nb.csv"
GATE_CELLS = [(20.0, 160), (20.0, 320)]


def hb(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n"); fh.flush()


def cell_seeds(fr, N):
    """(gen, rep) -> seed. Reuse phase1_recovery_hardened.run_matched's EXACT seeds for the
    FR=20, N in {160,320} cells so those cells equal phase1_recovery_matched_nb.csv; a separate
    non-overlapping scheme for every other cell."""
    if fr == 20.0 and N in (160, 320):
        Ni = [160, 320].index(N)
        return {(g, r): 60000 + ((Ni * 2 + gi) * R + r)
                for gi, g in enumerate(GENS) for r in range(R)}
    ci = FRS.index(fr) * len(NS) + NS.index(N)
    base = 100000 + ci * 1000
    return {(g, r): base + gi * R + r for gi, g in enumerate(GENS) for r in range(R)}


def compute_cell(fr, N, windows):
    """Return the 8 grid rows (true x fano) for one (fr,N) cell: NB-fit recovery over R reps."""
    seeds = cell_seeds(fr, N)
    jobs = [(g, fano, r) for g in GENS for fano in FANOS for r in range(R)]
    recs = Parallel(n_jobs=N_JOBS)(
        delayed(recover_canonical)(g, N, fr, seeds[(g, r)], windows, K, fano, "nb")
        for (g, fano, r) in jobs)
    df = pd.DataFrame(recs)
    rows = []
    for g in GENS:
        for fano in FANOS:
            sub = df[(df.true == g) & (df.fano == fano)]
            rows.append(dict(true=g, N=int(N), fr=float(fr), fano=float(fano),
                             recovery=float(sub.correct.mean()), R=int(len(sub))))
    return pd.DataFrame(rows)


def done_cells():
    if not OUT_CSV.exists():
        return set()
    d = pd.read_csv(OUT_CSV)
    return set(zip(d.fr.astype(float), d.N.astype(int)))


def write_cell(rows):
    rows.to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)


def agg_overdispersed(df, N):
    """Mean NB recovery over Fano>=1.5 and both generators at FR=20 for a given N."""
    s = df[(df.fr == 20.0) & (df.N == N) & (df.fano >= 1.5)]
    return float(s.recovery.mean())


def run_gate(windows):
    hb("REPRODUCE GATE: recompute FR=20, N in {160,320} (NB fit) ...")
    done = done_cells()
    parts = [pd.read_csv(OUT_CSV)] if OUT_CSV.exists() else []
    for fr, N in GATE_CELLS:
        if (fr, N) in done:
            hb(f"  gate cell FR={fr:g} N={N} already cached -> skip recompute")
            continue
        t = time.time()
        rows = compute_cell(fr, N, windows)
        write_cell(rows)
        parts.append(rows)
        hb(f"  gate cell FR={fr:g} N={N} done in {time.time()-t:.0f}s")
    grid = pd.concat(parts, ignore_index=True)
    got = {N: agg_overdispersed(grid, N) for N in (160, 320)}
    # cross-check against the canonical matched file (Fano>=1.5, fit=nb)
    m = pd.read_csv(MATCHED)
    mo = m[(m.fit == "nb") & (m.fano >= 1.5)].groupby("N")["correct"].mean()
    ref = {160: float(mo.loc[160]), 320: float(mo.loc[320])}
    hb(f"  recomputed NB recovery (Fano>=1.5): N160={got[160]:.3f}  N320={got[320]:.3f}")
    hb(f"  matched_nb.csv reference         : N160={ref[160]:.3f}  N320={ref[320]:.3f}")
    ok = all(abs(got[N] - ref[N]) <= 0.06 for N in (160, 320))
    hb(f"  GATE {'PASS' if ok else 'FAIL'} (tol 0.06; target ~0.74 / ~0.82)")
    return ok, got, ref


def run_grid(windows):
    done = done_cells()
    todo = [(fr, N) for fr in FRS for N in NS if (fr, N) not in done]
    hb(f"GRID: {len(FRS)*len(NS)} cells total, {len(done)} cached, {len(todo)} to do")
    for i, (fr, N) in enumerate(todo, 1):
        t = time.time()
        rows = compute_cell(fr, N, windows)
        write_cell(rows)
        rec = rows[rows.fano >= 1.5].recovery.mean()
        hb(f"  [{i}/{len(todo)}] cell FR={fr:g} N={N}: NB recovery(Fano>=1.5,both)="
           f"{rec:.3f}  ({time.time()-t:.0f}s)")
    hb(f"GRID complete: {len(done_cells())}/{len(FRS)*len(NS)} cells -> {OUT_CSV.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="run the reproduce gate only")
    a = ap.parse_args()
    windows = load_windows()
    hb(f"window pool: {len(windows)} RTs (median {np.median(windows)*1000:.0f} ms); "
       f"NB grid sigma=0.4, Fano={FANOS}, R={R}, K={K}")
    ok, got, ref = run_gate(windows)
    if not ok:
        raise SystemExit("STOP: NB reproduce gate FAILED — not sweeping the grid.")
    if a.gate:
        hb("gate-only mode: stopping after PASS."); return
    run_grid(windows)


if __name__ == "__main__":
    main()

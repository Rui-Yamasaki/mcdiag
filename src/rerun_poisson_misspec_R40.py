"""Fig 2b "Poisson-collapse" at R=40, on the SAME overdispersed datasets the NB-grid R=40 run saw.

Reuses phase1_recovery_nb_grid.cell_seeds at R=40 (the exact seed sequence of phase1_recovery_nb_grid_R40.csv)
and the validated recover_canonical harness, but fits with the POISSON emission model (fit="poisson").
Because recover_canonical draws win/counts/CV-split from `seed` BEFORE the fit branch, the Poisson fit sees
bit-identical trials to the NB fit. Writes ONE new file; touches nothing else.

Built-in determinism proof: recomputes the NB fit at FR20/N40 and requires an EXACT match to the existing
results/phase1_recovery_nb_grid_R40.csv (same seeds -> same data -> same nb recovery).

  python src/rerun_poisson_misspec_R40.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

import phase1_recovery_nb_grid as G  # noqa: E402  (cell_seeds; we set R=40 to match the R40 NB run)
from phase1_recovery_hardened import recover_canonical  # noqa: E402  (validated engine)
from phase1_recovery import load_windows  # noqa: E402
from ibl_one import PROJECT_ROOT  # noqa: E402

G.R = 40                                   # match the R=40 NB-grid seed sequence
RESULTS = PROJECT_ROOT / "results"
OUT = RESULTS / "phase1_recovery_poisson_misspec_R40.csv"
NB_R40 = RESULTS / "phase1_recovery_nb_grid_R40.csv"
FR, NS, FANOS, GENS, K, R = 20.0, [40, 160, 320], G.FANOS, G.GENS, G.K, G.R


def cell(N, fit):
    """recovery per (true,fano) for one FR20/N cell, R reps, given emission `fit`. SAME seeds as NB grid."""
    seeds = G.cell_seeds(FR, N)
    jobs = [(g, fano, r) for g in GENS for fano in FANOS for r in range(R)]
    recs = Parallel(n_jobs=G.N_JOBS)(
        delayed(recover_canonical)(g, N, FR, seeds[(g, r)], windows, K, fano, fit)
        for (g, fano, r) in jobs)
    df = pd.DataFrame(recs)
    rows = []
    for g in GENS:
        for fano in FANOS:
            sub = df[(df.true == g) & (df.fano == fano)]
            rows.append(dict(true=g, N=int(N), fr=FR, fano=float(fano), fit=fit,
                             recovery=float(sub.correct.mean()), R=int(len(sub))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    windows = load_windows()
    print(f"poisson-misspec regen: FR={FR}, N={NS}, R={R}, fano={FANOS}, K={K}, jobs={G.N_JOBS}", flush=True)

    # ---- bit-identity proof: nb @ FR20 N40 must EXACTLY equal the existing R40 grid ----
    t = time.time()
    nb40 = cell(40, "nb")[["true", "fano", "recovery"]].sort_values(["true", "fano"]).reset_index(drop=True)
    grid = pd.read_csv(NB_R40)
    g40 = (grid[(grid.fr == 20.0) & (grid.N == 40)][["true", "fano", "recovery"]]
           .sort_values(["true", "fano"]).reset_index(drop=True))
    mg = nb40.merge(g40, on=["true", "fano"], suffixes=("_recomputed", "_R40file"))
    mg["match"] = mg.recovery_recomputed == mg.recovery_R40file
    print("DETERMINISM PROOF (nb @ FR20 N40, recomputed vs existing R40 grid file):", flush=True)
    print(mg.to_string(index=False), flush=True)
    print(f"ALL MATCH = {bool(mg.match.all())}   ({time.time()-t:.0f}s)", flush=True)

    # ---- deliverable: POISSON fit on the same datasets, N=40/160/320 ----
    parts = []
    for N in NS:
        t = time.time()
        c = cell(N, "poisson")
        parts.append(c)
        rec = c[c.fano >= 1.5].recovery.mean()
        print(f"  poisson cell FR20 N={N}: recovery(Fano>=1.5,both)={rec:.3f}  ({time.time()-t:.0f}s)", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"wrote -> {OUT}", flush=True)
    o = out[out.fano >= 1.5]
    summ = {int(n): round(v, 3) for n, v in o.groupby("N")["recovery"].mean().items()}
    print(f"POISSON recovery (Fano>=1.5, both) by N: {summ}", flush=True)

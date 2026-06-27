"""Replication-feasibility coverage assessment for Steinmetz et al. 2019 (PURE ASSESSMENT).

Reads the cached Neuromatch binned release (data/steinmetz/steinmetz_part{0,1,2}.npz,
39 sessions, 10 ms bins, all simultaneously-recorded multi-probe) and reports, WITHOUT
building any analysis pipeline:
  - per-region cell counts + high-FR (>=10 Hz, the IBL floor) for our targets
    {MRN, SCm, SNr, GRN, IRN} and their Allen-CCF analogues (SCs, APN, PAG, RN, ZI ...);
  - per-session SIMULTANEOUS yield per region (Steinmetz is multi-probe -> simultaneous);
  - the DECORRELATED-trial budget (error + equal/zero-contrast Go trials) per session+total;
  - movement-covariate availability (wheel / face motion-energy / pupil).

No raw voltage, no figshare bulk download; the binned .npz is the compact coverage route.

  python src/steinmetz_coverage.py
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from ibl_one import PROJECT_ROOT

DATA = PROJECT_ROOT / "data" / "steinmetz"
OUT_REGION = PROJECT_ROOT / "results" / "steinmetz_coverage.csv"
OUT_SESSION = PROJECT_ROOT / "results" / "steinmetz_coverage_sessions.csv"

TARGETS = ["MRN", "SCm", "SNr", "GRN", "IRN"]          # our IBL decision-core regions
ANALOGUES = ["SCs", "SCsg", "SCig", "APN", "PAG", "RN", "ZI", "MB"]  # nearby midbrain
FR_FLOOR = 10.0                                         # Hz, same high-FR floor as IBL


def load_sessions():
    out = []
    for j in range(3):
        f = DATA / f"steinmetz_part{j}.npz"
        out += list(np.load(f, allow_pickle=True)["dat"])
    return out


def session_fr(s):
    """Mean FR (Hz) per neuron over the full recorded window (n_trials x 2.5 s)."""
    sp = s["spks"]                                      # (neurons, trials, bins)
    n_tr, n_bin = sp.shape[1], sp.shape[2]
    dur = n_tr * n_bin * s["bin_size"]                 # total seconds observed/neuron
    return sp.sum(axis=(1, 2)) / dur


def decorrelated(s):
    """Go-trial counts where choice is separable from stimulus side.
    error  = feedback_type == -1 (incorrect) & Go ; equal = contrast_left==contrast_right
    & Go ; zero = both contrasts 0 & Go. Union = decorrelated budget."""
    cl, cr = s["contrast_left"], s["contrast_right"]
    resp, fb = s["response"], s["feedback_type"]
    go = resp != 0
    err = (fb == -1) & go
    equal = (cl == cr) & go
    zero = (cl == 0) & (cr == 0) & go
    union = (err | equal)
    return dict(n_trials=len(resp), n_go=int(go.sum()), n_nogo=int((~go).sum()),
                n_err=int(err.sum()), n_equal=int(equal.sum()), n_zero=int(zero.sum()),
                n_decorr=int(union.sum()))


def main():
    sess = load_sessions()
    print(f"loaded {len(sess)} Steinmetz sessions (binned, 10 ms)\n")

    all_regions = collections.Counter()
    reg_cells = collections.Counter()          # total cells per region
    reg_hifr = collections.Counter()           # high-FR cells per region
    reg_sess = collections.defaultdict(list)   # per-session simultaneous counts per region
    reg_sess_hifr = collections.defaultdict(list)
    srows = []
    cov = dict(wheel=0, face=0, pupil=0)

    for si, s in enumerate(sess):
        ba = np.asarray([str(x) for x in s["brain_area"]])
        fr = session_fr(s)
        all_regions.update(ba.tolist())
        d = decorrelated(s)
        has = dict(wheel=np.size(s.get("wheel", [])) > 0,
                   face=np.size(s.get("face", [])) > 0,
                   pupil=np.size(s.get("pupil", [])) > 0)
        for k in cov:
            cov[k] += int(has[k])
        # per-region tallies for this (simultaneous) session
        sess_reg = {}
        for r in set(ba.tolist()):
            m = ba == r
            n = int(m.sum()); nhi = int((fr[m] >= FR_FLOOR).sum())
            reg_cells[r] += n; reg_hifr[r] += nhi
            reg_sess[r].append(n); reg_sess_hifr[r].append(nhi)
            sess_reg[r] = (n, nhi)
        tgt = {r: sess_reg.get(r, (0, 0)) for r in TARGETS}
        srows.append(dict(sess=si, mouse=s["mouse_name"], date=s["date_exp"],
                          n_neurons=len(ba), n_regions=len(set(ba.tolist())),
                          **{f"{r}_cells": tgt[r][0] for r in TARGETS},
                          **{f"{r}_hiFR": tgt[r][1] for r in TARGETS},
                          **d, **{f"has_{k}": int(has[k]) for k in cov}))

    sdf = pd.DataFrame(srows)
    sdf.to_csv(OUT_SESSION, index=False)

    # region coverage table
    rows = []
    for r in TARGETS + ANALOGUES:
        cells = reg_cells.get(r, 0)
        if cells == 0 and r in ANALOGUES:
            continue
        sc = np.array(reg_sess.get(r, [0]))
        schi = np.array(reg_sess_hifr.get(r, [0]))
        rows.append(dict(region=r, target=(r in TARGETS), total_cells=cells,
                         hiFR_cells=reg_hifr.get(r, 0), n_sessions=len(reg_sess.get(r, [])),
                         max_simult=int(sc.max()) if len(sc) else 0,
                         median_simult=float(np.median(sc)) if len(sc) else 0.0,
                         max_simult_hiFR=int(schi.max()) if len(schi) else 0,
                         median_simult_hiFR=float(np.median(schi)) if len(schi) else 0.0))
    rdf = pd.DataFrame(rows).sort_values(["target", "total_cells"], ascending=[False, False])
    rdf.to_csv(OUT_REGION, index=False)

    # ---- report ----
    print("== TARGET-REGION COVERAGE (high-FR floor = %.0f Hz; Steinmetz = simultaneous) ==" % FR_FLOOR)
    print(rdf[rdf.target][["region", "total_cells", "hiFR_cells", "n_sessions",
                           "max_simult", "median_simult", "max_simult_hiFR"]].to_string(index=False))
    print("\n== nearby midbrain analogues present ==")
    nb = rdf[~rdf.target]
    print(nb[["region", "total_cells", "hiFR_cells", "n_sessions", "max_simult"]].to_string(index=False)
          if len(nb) else "  (none of the listed analogues present)")

    print("\n== DECORRELATED-TRIAL BUDGET (Go trials, choice separable from stimulus) ==")
    tot = sdf[["n_err", "n_equal", "n_zero", "n_decorr"]].sum()
    print(f"  TOTAL across {len(sdf)} sessions: error={tot.n_err}, equal-contrast={tot.n_equal}, "
          f"zero-contrast={tot.n_zero}, decorrelated(union)={tot.n_decorr}")
    print(f"  per session: decorr median {sdf.n_decorr.median():.0f} "
          f"(IQR {sdf.n_decorr.quantile(.25):.0f}-{sdf.n_decorr.quantile(.75):.0f}), "
          f"error median {sdf.n_err.median():.0f}, n_trials median {sdf.n_trials.median():.0f}")

    print("\n== MOVEMENT COVARIATES (sessions with non-empty) ==")
    print(f"  wheel {cov['wheel']}/{len(sess)}, face-motion-energy {cov['face']}/{len(sess)}, "
          f"pupil {cov['pupil']}/{len(sess)}  (no full DLC body pose in release)")

    print("\n== SIMULTANEITY (MRN priority) ==")
    mrn_sess = sdf[sdf.MRN_cells > 0]
    if len(mrn_sess):
        print(f"  MRN present in {len(mrn_sess)}/{len(sdf)} sessions; simultaneous MRN cells/session "
              f"median {mrn_sess.MRN_cells.median():.0f}, max {mrn_sess.MRN_cells.max():.0f}; "
              f"hi-FR median {mrn_sess.MRN_hiFR.median():.0f}, max {mrn_sess.MRN_hiFR.max():.0f}")
        print(f"  total MRN cells {reg_cells.get('MRN',0)} (hi-FR {reg_hifr.get('MRN',0)})")
    else:
        print("  !! MRN absent from all sessions")

    print(f"\nSaved region coverage -> {OUT_REGION}")
    print(f"Saved per-session table -> {OUT_SESSION}")


if __name__ == "__main__":
    main()

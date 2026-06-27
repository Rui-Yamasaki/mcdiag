"""Reproduce-first anchor checks for the referee-response tasks (PURE; cached data only).

Re-runs the ACTUAL pipeline functions (not CSV echoes) and prints my-number vs manuscript.

  ./.venv/Scripts/python.exe src/referee_reproduce.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ibl_one import PROJECT_ROOT
import audit_corrections as AC

R = PROJECT_ROOT / "results"


def anchor_cascade():
    """Task B anchor: IBL dataset-wide BH cascade 301->98, 131->0, 59->0 (column sums)."""
    print("\n===== Task B anchor: IBL dataset-wide BH cascade =====")
    ic = pd.read_csv(AC.IBL_CELLS)
    if "p_move" not in ic:
        ic["p_move"] = np.where(ic["has_dlc"], ic["p_move_full"], ic["p_move_wheel"])
    ibl = AC.cascade_counts(ic, ["GRN", "MRN", "SNr", "SCm", "IRN"])
    s = ibl[["choice_raw", "choice_fdr", "move_surv_raw", "move_surv_fdr",
             "triple_raw", "triple_fdr"]].sum()
    print(f"  choice-selective       raw->FDR : {s.choice_raw}->{s.choice_fdr}   (manuscript 301->98)")
    print(f"  movement-independent   raw->FDR : {s.move_surv_raw}->{s.move_surv_fdr}   (manuscript 131->0)")
    print(f"  triple-coded           raw->FDR : {s.triple_raw}->{s.triple_fdr}   (manuscript 59->0)")
    return ibl


def anchor_census():
    """Task C anchor: sampled 35-session all-QC maxima MRN 56, SCm 79, global 79."""
    print("\n===== Task C anchor: sampled all-QC simultaneity maxima =====")
    d = pd.read_csv(R / "phase2_census_units.csv")
    n_sess, n_ins = d["eid"].nunique(), d["pid"].nunique()
    print(f"  sampled census: {n_ins} insertions / {n_sess} sessions")
    glob_max = 0
    for r in ["MRN", "SCm", "IRN", "SNr", "GRN"]:
        sub = d[d["region"] == r]
        if sub.empty:
            print(f"  {r:4s}: (absent)"); continue
        g = sub.groupby("eid").size()
        glob_max = max(glob_max, int(g.max()))
        print(f"  {r:4s}: {g.size:2d} region-sessions, median {g.median():.0f}, max {int(g.max())}")
    print(f"  GLOBAL MAX = {glob_max}   (manuscript: MRN 56, SCm 79, global max 79)")


def anchor_pooled():
    """Task A (part) anchor: IBL MRN pooled residual +0.059 SD CI[0.001,0.117]."""
    print("\n===== Task A anchor (pooled): IBL MRN error-trial decision effect =====")
    ibl = pd.read_csv(AC.IBL_FEAT)
    r = AC.pooled_effect(ibl, "MRN", "err")
    print(f"  IBL MRN: {r['eff']:+.4f} SD  95%CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
          f"({r['n_cells']} cells)   (manuscript +0.059 SD [0.001, 0.117])")


if __name__ == "__main__":
    anchor_cascade()
    anchor_census()
    anchor_pooled()

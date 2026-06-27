"""TASK B — FDR scope: dataset-wide vs within-region BH vs hierarchical (Benjamini-Bogomolov).

The manuscript applies BH dataset-wide (max correction burden). A referee argues this may be what
produces "0 triple-coded." We re-run the cascade under less aggressive corrections and report
whether the zero holds. PURE; uses the existing per-cell p-values.

Cascade definitions (mirror docs/Table S4 / audit_corrections.cascade_counts):
  choice-selective     = p_delib significant
  movement-independent = choice-selective AND p_move significant
  stim/prior-independent = choice-selective AND p_stim significant
  triple-coded         = choice-selective AND p_early (leading) AND p_move AND p_stim significant
Each filter's "significant" is decided by the SAME q<0.05 BH, but the FAMILY changes:
  (1) dataset-wide BH  : one family = all hi-FR cells in the dataset (the published choice).
  (2) within-region BH : family = the cells of one region (each region & each filter its own family).
  (3) hierarchical (BB): screen regions by a Simes region-level p, BH-select regions, then
                         within-region BH at the BB-adjusted level q*|R|/m.

  ./.venv/Scripts/python.exe src/referee_fdr_scope.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from ibl_one import PROJECT_ROOT
import audit_corrections as AC

R = PROJECT_ROOT / "results"
OUT = R / "referee_response"
OUT.mkdir(parents=True, exist_ok=True)
Q = AC.FDR_Q
FILTERS = ["p_delib", "p_early", "p_move", "p_stim"]
IBL_REGIONS = ["MRN", "SCm", "SNr", "GRN", "IRN"]
ST_REGIONS = ["MRN", "SC", "SNr"]


def _bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); m = np.isfinite(p)
    if m.any():
        q[m] = false_discovery_control(p[m], method="bh")
    return q


def load_ibl():
    c = pd.read_csv(AC.IBL_CELLS)
    if "p_move" not in c:
        c["p_move"] = np.where(c["has_dlc"], c["p_move_full"], c["p_move_wheel"])
    return c[["cell", "region", "p_delib", "p_early", "p_move", "p_stim"]].copy()


def load_steinmetz():
    cache = OUT / "steinmetz_cascade_pvals.csv"
    if cache.exists():
        return pd.read_csv(cache)
    print("  recomputing Steinmetz per-cell cascade p-values (2000-perm AUC tests)...")
    c = AC.steinmetz_cascade_percell()
    c = c[["cell", "region", "p_delib", "p_early", "p_move", "p_stim"]].copy()
    c.to_csv(cache, index=False)
    return c


# ---------- significance under each scheme ----------
def sig_datasetwide(c):
    """q<Q computed over ALL cells (one family per filter)."""
    out = {}
    for filt in FILTERS:
        out[filt] = _bh(c[filt].to_numpy()) < Q
    return pd.DataFrame(out, index=c.index)


def sig_withinregion(c):
    """q<Q computed within each region (region x filter = a family)."""
    flags = {f: np.zeros(len(c), bool) for f in FILTERS}
    for reg, idx in c.groupby("region").groups.items():
        idx = list(idx)
        for filt in FILTERS:
            q = _bh(c.loc[idx, filt].to_numpy())
            for j, ix in enumerate(idx):
                flags[filt][c.index.get_loc(ix)] = q[j] < Q
    return pd.DataFrame(flags, index=c.index)


def simes(p):
    p = np.sort(np.asarray(p, float)); m = len(p)
    return float(np.min(p * m / np.arange(1, m + 1))) if m else 1.0


def sig_hierarchical_bb(c):
    """Benjamini-Bogomolov: stage-1 screen regions by Simes(p_delib); BH-select regions at Q;
    stage-2 within each selected region, BH each filter at level Q*|R|/m. Non-selected regions
    contribute no survivors. (Choice screen drives region selection; all filters use the BB level
    inside selected regions.)"""
    regions = list(c["region"].unique())
    m = len(regions)
    reg_p = {r: simes(c.loc[c.region == r, "p_delib"].to_numpy()) for r in regions}
    order = sorted(regions, key=lambda r: reg_p[r])
    pv = np.array([reg_p[r] for r in order])
    bh = pv * m / np.arange(1, m + 1)
    # largest k with pv[k]<=Q*k/m  -> select that many smallest
    below = np.where(pv <= Q * np.arange(1, m + 1) / m)[0]
    ksel = (below.max() + 1) if below.size else 0
    selected = set(order[:ksel])
    qlevel = Q * len(selected) / m if selected else 0.0
    flags = {f: np.zeros(len(c), bool) for f in FILTERS}
    for reg in selected:
        idx = list(c.index[c.region == reg])
        for filt in FILTERS:
            q = _bh(c.loc[idx, filt].to_numpy())
            for j, ix in enumerate(idx):
                flags[filt][c.index.get_loc(ix)] = q[j] < qlevel
    return pd.DataFrame(flags, index=c.index), selected, qlevel


def counts(c, sig, regions, scheme, dataset):
    """Per-region survivor counts for choice / movement-indep / stim-indep / triple."""
    choice = sig["p_delib"]
    move = choice & sig["p_move"]
    stim = choice & sig["p_stim"]
    triple = choice & sig["p_early"] & sig["p_move"] & sig["p_stim"]
    rows = []
    for reg in regions:
        m = (c.region == reg).to_numpy()
        rows.append(dict(dataset=dataset, scheme=scheme, region=reg, hiFR=int(m.sum()),
                         choice_fdr=int((choice & m).sum()),
                         move_indep_fdr=int((move & m).sum()),
                         stim_indep_fdr=int((stim & m).sum()),
                         triple_fdr=int((triple & m).sum())))
    tot = dict(dataset=dataset, scheme=scheme, region="ALL", hiFR=int(len(c)),
               choice_fdr=int(choice.sum()), move_indep_fdr=int(move.sum()),
               stim_indep_fdr=int(stim.sum()), triple_fdr=int(triple.sum()))
    rows.append(tot)
    return rows


def run(c, regions, dataset):
    rows = []
    rows += counts(c, sig_datasetwide(c), regions, "dataset-wide BH", dataset)
    rows += counts(c, sig_withinregion(c), regions, "within-region BH", dataset)
    bb, sel, ql = sig_hierarchical_bb(c)
    print(f"  [{dataset}] Benjamini-Bogomolov: selected regions {sorted(sel)} "
          f"(within-region BH level q={ql:.4f})")
    rows += counts(c, bb, regions, "hierarchical BB", dataset)
    return rows


def main():
    print("=== TASK B: FDR scope (dataset-wide vs within-region vs hierarchical BB) ===\n")
    ibl = load_ibl()
    st = load_steinmetz()
    rows = run(ibl, IBL_REGIONS, "IBL") + run(st, ST_REGIONS, "Steinmetz")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fdr_scope.csv", index=False)

    for dataset in ("IBL", "Steinmetz"):
        print(f"\n----- {dataset} -----")
        for scheme in ("dataset-wide BH", "within-region BH", "hierarchical BB"):
            sub = df[(df.dataset == dataset) & (df.scheme == scheme)]
            print(f"  [{scheme}]")
            print("    " + sub[["region", "hiFR", "choice_fdr", "move_indep_fdr",
                                 "stim_indep_fdr", "triple_fdr"]].to_string(index=False).replace("\n", "\n    "))

    print("\n========== DECISION NUMBER ==========")
    for dataset in ("IBL", "Steinmetz"):
        for scheme in ("within-region BH", "hierarchical BB"):
            sub = df[(df.dataset == dataset) & (df.scheme == scheme) & (df.region != "ALL")]
            tri = sub[sub.triple_fdr > 0]
            if len(tri):
                where = ", ".join(f"{r.region}={r.triple_fdr}" for _, r in tri.iterrows())
                print(f"  {dataset} {scheme}: triple-coded SURVIVE -> {where}")
            else:
                print(f"  {dataset} {scheme}: triple-coded = 0 in EVERY region (zero holds)")
    print(f"\nSaved -> {OUT / 'fdr_scope.csv'}")


if __name__ == "__main__":
    main()

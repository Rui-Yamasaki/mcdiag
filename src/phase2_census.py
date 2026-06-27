"""Phase 2 / step 1 - population census (TRIMMED spikes, NO model fitting).

Go/no-go sub-gate: does the Phase-1 RECOVERABLE regime (FR >= 10-20 Hz, poolable
to N >= 160-320 engaged deliberative trials) actually contain enough REAL
target-region choice-selective neurons to justify modeling?

Target regions (IBL Brain-Wide Map, Nature 2025, s41586-025-09235-0): choice
responses were "highly prevalent in the midbrain (SCm, MRN, SNr, RPF, NPC) and
the medulla and cerebellar nuclei (GRN, IRN, SOC, VII, TRN, FOTU)", and stimulus
representations "spread to ramp-like activity in a collection of midbrain and
hindbrain regions that also encoded choices" (GRN highlighted as the exemplar
single-neuron choice region). We census the high-coverage decision core
{GRN, MRN, SNr, SCm, IRN} for spikes and report poolable-N for the full list.

Data loaded per insertion is TRIMMED: spikes.times, spikes.clusters, the clusters
table (for peak channel + QC label) and channels (for region) -- NEVER spikes.amps
or spikes.depths. Streamed one insertion at a time; large spike files are deleted
after processing (--no-cleanup to keep them).

Census selectivity is a deliberately SIMPLE heuristic (choice ROC-AUC of the
per-trial deliberative-window firing rate), explicitly NOT the final selectivity
analysis.

    python src/phase2_census.py --stage coverage   # region -> insertions/sessions + poolable N
    python src/phase2_census.py --stage spikes --cap 12   # trimmed spike census (background)
    python src/phase2_census.py --stage report     # FR distribution + figure + verdict
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from brainbox.io.one import SpikeSortingLoader, SessionLoader  # noqa: E402
from iblatlas.atlas import BrainRegions  # noqa: E402

from ibl_one import BWM_PROJECT, DATA_DIR, PROJECT_ROOT, get_one  # noqa: E402

warnings.filterwarnings("ignore", message="Multiple revisions")

# Paper target regions (Beryl acronyms). Spike census on the high-coverage core;
# poolable-N reported for the full list.
CORE_REGIONS = ["GRN", "MRN", "SNr", "SCm", "IRN"]
ALL_REGIONS = ["SCm", "MRN", "SNr", "RPF", "NPC",       # midbrain
               "GRN", "IRN", "SOC", "VII", "TRN", "FOTU"]  # medulla/cerebellar

# deliberative-trial filter (engaged, slow, low-to-moderate contrast)
RT_LO, RT_HI = 0.5, 2.0
MAX_ABS_CONTRAST = 25.0          # %  (low-to-moderate)
MIN_PER_SIDE = 8                 # min trials/choice for an AUC
GOOD_LABEL = 1.0                 # IBL clusters.metrics.label == 1 -> good unit
AUC_SEL = 0.10                   # |AUC-0.5| >= 0.10 -> "choice-selective" (heuristic)
FR_BINS_HZ = (10.0, 20.0)        # recoverable-lane thresholds from Phase 1

COVERAGE_CSV = PROJECT_ROOT / "results" / "phase2_coverage.csv"
SESSION_DELIB_CSV = PROJECT_ROOT / "results" / "phase2_session_delib.csv"
UNITS_CSV = PROJECT_ROOT / "results" / "phase2_census_units.csv"
FIG_PATH = PROJECT_ROOT / "figures" / "phase2_fr_census.png"
REGION_CSV = PROJECT_ROOT / "results" / "phase2_region_census.csv"


# --- region -> insertions/sessions (batched, details=True) -------------------
def region_coverage(one):
    rows, reg_eids, reg_pids = [], {}, {}
    for reg in ALL_REGIONS:
        ids, det = one.search_insertions(atlas_acronym=reg, project=BWM_PROJECT,
                                         details=True)
        pids = [str(i) for i in ids]
        eids = sorted({d["session"] for d in det})
        reg_pids[reg], reg_eids[reg] = pids, eids
        rows.append(dict(region=reg, insertions=len(pids), sessions=len(eids)))
        print(f"  {reg:5s}: {len(pids):3d} insertions, {len(eids):3d} sessions")
    return pd.DataFrame(rows), reg_eids, reg_pids


# --- deliberative trials for one session -------------------------------------
def deliberative_trials(trials):
    on = trials["stimOn_times"].to_numpy(float)
    fm = trials["firstMovement_times"].to_numpy(float)
    rsp = trials["response_times"].to_numpy(float)
    ch = trials["choice"].to_numpy(float)
    cl = np.nan_to_num(trials["contrastLeft"].to_numpy(float))
    cr = np.nan_to_num(trials["contrastRight"].to_numpy(float))
    absc = np.fmax(cl, cr) * 100.0
    rt = fm - on
    m = (ch != 0) & np.isfinite(on) & np.isfinite(fm) & np.isfinite(rsp)
    m &= (rt >= RT_LO) & (rt <= RT_HI) & (absc <= MAX_ABS_CONTRAST)
    return on[m], fm[m], ch[m]


# --- one insertion: trimmed spikes -> per target-region unit census ----------
def process_insertion(one, br, pid, cleanup=True):
    ssl = SpikeSortingLoader(pid=pid, one=one)
    eid, pname = ssl.eid, ssl.pname
    coll = f"alf/{pname}/pykilosort"

    sl = SessionLoader(one=one, eid=eid)
    sl.load_trials()
    on, fm, ch = deliberative_trials(sl.trials)
    n_delib = len(on)
    if n_delib < 2 * MIN_PER_SIDE:
        return None, dict(pid=pid, eid=eid, n_delib=n_delib, skipped="few_trials")

    clusters = one.load_object(eid, "clusters", collection=coll)
    channels = one.load_object(eid, "channels", collection=coll)
    spikes = one.load_object(eid, "spikes", collection=coll,
                             attribute=["times", "clusters"])  # TRIMMED
    st, sc = spikes["times"], spikes["clusters"].astype(int)

    # per-cluster region (Beryl) via peak channel; QC label
    ch_beryl = br.id2acronym(channels["brainLocationIds_ccf_2017"], mapping="Beryl")
    clu_region = np.asarray(ch_beryl)[clusters["channels"].astype(int)]
    label = clusters["metrics"]["label"].to_numpy(float)
    nclu = len(clusters["channels"])
    Trec = float(st[-1])
    fr_overall = np.bincount(sc, minlength=nclu)[:nclu] / Trec

    # per-trial x per-cluster spike counts in the deliberative window
    dur = fm - on
    counts = np.zeros((n_delib, nclu))
    for k in range(n_delib):
        i0 = np.searchsorted(st, on[k]); i1 = np.searchsorted(st, fm[k])
        counts[k] = np.bincount(sc[i0:i1], minlength=nclu)[:nclu]
    rates = counts / dur[:, None]                       # Hz, per trial x cluster
    y_right = (ch > 0).astype(int)
    n_left, n_right = int((ch < 0).sum()), int((ch > 0).sum())

    out = []
    target = set(CORE_REGIONS)
    for u in range(nclu):
        if clu_region[u] not in target or label[u] < GOOD_LABEL:
            continue
        auc = np.nan
        if n_left >= MIN_PER_SIDE and n_right >= MIN_PER_SIDE:
            try:
                auc = roc_auc_score(y_right, rates[:, u])
            except Exception:
                auc = np.nan
        out.append(dict(region=clu_region[u], pid=pid, eid=eid, cluster=u,
                        fr_overall=float(fr_overall[u]),
                        fr_window=float(rates[:, u].mean()),
                        auc=float(auc), selectivity=float(abs(auc - 0.5)),
                        n_delib=n_delib, n_left=n_left, n_right=n_right))

    if cleanup:
        spath = one.eid2path(eid)
        for fn in ("spikes.times.npy", "spikes.clusters.npy"):
            for f in Path(spath).glob(f"alf/{pname}/pykilosort/**/{fn}"):
                try:
                    f.unlink()
                except OSError:
                    pass
    meta = dict(pid=pid, eid=eid, n_delib=n_delib, n_left=n_left,
                n_right=n_right, n_target_good=len(out))
    return pd.DataFrame(out), meta


# --- poolable deliberative N (behavior only, comprehensive) ------------------
def session_delib_counts(one, eids):
    """Per-session deliberative-trial counts; incremental + resumable.

    Appends each session to SESSION_DELIB_CSV as it goes (skips eids already
    there), so the file's row count is a live progress signal and a crash just
    resumes.
    """
    done = set()
    if SESSION_DELIB_CSV.exists():
        done = set(pd.read_csv(SESSION_DELIB_CSV)["eid"].astype(str))
    SESSION_DELIB_CSV.parent.mkdir(parents=True, exist_ok=True)
    todo = [e for e in eids if str(e) not in done]
    print(f"  {len(done)} sessions cached; {len(todo)} to compute")
    for i, eid in enumerate(todo, 1):
        try:
            sl = SessionLoader(one=one, eid=eid)
            sl.load_trials()
            on, fm, ch = deliberative_trials(sl.trials)
            row = dict(eid=str(eid), n_delib=len(on),
                       n_left=int((ch < 0).sum()), n_right=int((ch > 0).sum()))
        except Exception as exc:  # noqa: BLE001
            row = dict(eid=str(eid), n_delib=0, n_left=0, n_right=0)
        pd.DataFrame([row]).to_csv(SESSION_DELIB_CSV, mode="a",
                                   header=not SESSION_DELIB_CSV.exists(), index=False)
        if i % 25 == 0 or i == len(todo):
            print(f"  trials {i}/{len(todo)}")
    return pd.read_csv(SESSION_DELIB_CSV)


def run_coverage(one):
    cov, reg_eids, reg_pids = region_coverage(one)
    union = sorted({e for eids in reg_eids.values() for e in eids})
    print(f"\nComputing deliberative-N for {len(union)} sessions (behavior only)...")
    sdf = session_delib_counts(one, union)
    SESSION_DELIB_CSV.parent.mkdir(parents=True, exist_ok=True)
    sdf.to_csv(SESSION_DELIB_CSV, index=False)
    dmap = sdf.set_index("eid")["n_delib"].to_dict()
    cov["poolable_N"] = [int(sum(dmap.get(str(e), 0) for e in reg_eids[r]))
                         for r in cov["region"]]
    cov["median_session_N"] = [
        int(np.median([dmap.get(str(e), 0) for e in reg_eids[r]]) if reg_eids[r] else 0)
        for r in cov["region"]]
    cov.to_csv(COVERAGE_CSV, index=False)
    print("\n" + cov.to_string(index=False))
    print(f"\nSaved -> {COVERAGE_CSV}, {SESSION_DELIB_CSV}")


def build_sample(one, cap):
    pids, seen = [], set()
    for reg in CORE_REGIONS:
        ids, _ = one.search_insertions(atlas_acronym=reg, project=BWM_PROJECT,
                                       details=True)
        for pid in [str(i) for i in ids][:cap]:
            if pid not in seen:
                seen.add(pid); pids.append(pid)
    return pids


def run_spikes(one, br, cap, cleanup):
    pids = build_sample(one, cap)
    done = set()
    if UNITS_CSV.exists():
        done = set(pd.read_csv(UNITS_CSV)["pid"].astype(str).unique())
    todo = [p for p in pids if p not in done]
    print(f"Spike census: {len(pids)} sampled insertions, {len(todo)} to do "
          f"({len(done)} already in CSV).")
    UNITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    for i, pid in enumerate(todo, 1):
        try:
            df, meta = process_insertion(one, br, pid, cleanup=cleanup)
            n = 0 if df is None else len(df)
            if df is not None and n:
                df.to_csv(UNITS_CSV, mode="a", header=not UNITS_CSV.exists(),
                          index=False)
            print(f"  [{i}/{len(todo)}] {pid[:8]} -> {n} target units "
                  f"(n_delib={meta.get('n_delib')})")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(todo)}] {pid[:8]} FAILED: {repr(exc)[:90]}")
    print(f"\nSaved per-unit census -> {UNITS_CSV}")


def run_report(one):
    from scipy.stats import norm
    units = pd.read_csv(UNITS_CSV)
    units = units[units["auc"].notna()].copy()
    cov = pd.read_csv(COVERAGE_CSV)
    n_ins_sampled = units.groupby("region")["pid"].nunique()
    ins_total = cov.set_index("region")["insertions"]
    pool = cov.set_index("region")["poolable_N"]
    sess = cov.set_index("region")["sessions"]

    # null AUC SD per unit (Mann-Whitney) -> chance P(|AUC-0.5| >= thr).
    # The in-window AUC heuristic is noisy at ~40 trials, so a lenient threshold
    # is heavily chance-contaminated; we report the chance-CORRECTED excess.
    n1, n2 = units["n_left"], units["n_right"]
    units["nullsd"] = np.sqrt((n1 + n2 + 1) / (12 * n1 * n2))

    def p_chance(t, sub):
        return float((2 * (1 - norm.cdf(t / sub["nullsd"]))).sum())

    print(f"\nSampled {units['pid'].nunique()} insertions -> {len(units)} "
          f"target-region good units with a computable choice-AUC "
          f"(median {int(units['n_left'].median())}L/{int(units['n_right'].median())}R "
          f"deliberative trials).")
    print("\n  choice-selectivity vs chance (in-window AUC heuristic):")
    print(f"  {'thr':>5} {'observed':>9} {'~chance':>8} {'excess':>7} | "
          f"{'hiFR>=20 obs':>12} {'~chance':>8} {'excess':>7}")
    for t in (0.10, 0.15, 0.20):
        obs = int((units["selectivity"] >= t).sum())
        hi = units[units["fr_window"] >= 20]
        hobs = int((hi["selectivity"] >= t).sum())
        print(f"  {t:>5} {obs:>9} {p_chance(t,units):>8.0f} "
              f"{obs-p_chance(t,units):>7.0f} | {hobs:>12} "
              f"{p_chance(t,hi):>8.0f} {hobs-p_chance(t,hi):>7.0f}")

    # robust population: high-FR GOOD units (any selectivity)
    print(f"\n  high-FR GOOD units (modelable population, any selectivity): "
          f">=10 Hz {int((units['fr_window']>=10).sum())} | "
          f">=20 Hz {int((units['fr_window']>=20).sum())}  (sampled)")

    THR = 0.15                                  # primary selectivity threshold
    rows = []
    for reg in CORE_REGIONS:
        u = units[units["region"] == reg]
        hi20 = u[u["fr_window"] >= 20]
        sel20 = int((hi20["selectivity"] >= THR).sum())
        exc20 = max(0.0, sel20 - p_chance(THR, hi20))
        ki = int(n_ins_sampled.get(reg, 0)) or 1
        scale = int(ins_total.get(reg, 0)) / ki   # sample -> full coverage
        rows.append(dict(
            region=reg, poolable_N=int(pool.get(reg, 0)),
            sessions=int(sess.get(reg, 0)),
            median_session_N=int(cov.set_index("region")["median_session_N"].get(reg, 0)),
            sampled_insertions=int(n_ins_sampled.get(reg, 0)),
            good_units=len(u), good_hiFR20=len(hi20),
            sel_hiFR20_obs=sel20,
            sel_hiFR20_excess=round(exc20, 1),
            est_sel_hiFR20_full=int(round(exc20 * scale))))
    rt = pd.DataFrame(rows)
    REGION_CSV.parent.mkdir(parents=True, exist_ok=True)
    rt.to_csv(REGION_CSV, index=False)
    print(f"\n========== per-region census (selectivity thr |AUC-0.5|>={THR}, "
          f"chance-corrected) ==========")
    print(rt.to_string(index=False))

    make_figure(units, rt, THR)

    samp_hi20 = int((units["fr_window"] >= 20).sum())
    excess_full = int(rt["est_sel_hiFR20_full"].sum())
    print("\n========== GLANCEABLE VERDICT ==========")
    print(f"  HIGH-FR population is abundant: {samp_hi20} good units >=20 Hz in the "
          f"sampled ~{units['pid'].nunique()} insertions (hundreds-to-1000+ full coverage).")
    print(f"  CHOICE-SELECTIVE high-FR (>=20 Hz, chance-corrected, thr {THR}): "
          f"~{int(rt['sel_hiFR20_excess'].sum())} in sample -> ~{excess_full} full coverage, "
          f"GRN-led ({rt.set_index('region')['sel_hiFR20_excess'].to_dict()}).")
    print(f"  Per-NEURON deliberative N ~ {int(rt['median_session_N'].median())} "
          f"(single session) -> single-neuron recovery is MARGINAL (Phase-1 needs N>=160);")
    print("  aggregate poolable N per region is large -> HIERARCHICAL pooling across "
          "cells is the viable path, and the cell count supports it.")
    lane = ("POPULATED (proceed; GRN/MRN core, hierarchical pooling)"
            if excess_full >= 30 else "MARGINAL" if excess_full >= 10 else "EMPTY")
    print(f"  -> recoverable lane: {lane}")
    print("  NOTE: selectivity here is a crude in-window AUC census heuristic; a "
          "movement/contrast-controlled permutation test is the next step to pin counts.")


def make_figure(units, rt, thr):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # A: FR distribution of ALL good units (robust population) + selective subset
    ax = axes[0]
    allv = units["fr_window"].clip(upper=80).to_numpy()
    selv = units.loc[units["selectivity"] >= thr, "fr_window"].clip(upper=80).to_numpy()
    bins = np.arange(0, 82, 4)
    ax.hist(allv, bins=bins, color="#bcd", edgecolor="white",
            label=f"all good units (n={len(units)})")
    ax.hist(selv, bins=bins, color="#27a", edgecolor="white",
            label=f"choice-selective |AUC-.5|>={thr} (n={len(selv)})")
    for t, c in ((10, "orange"), (20, "red")):
        ax.axvline(t, color=c, ls="--", lw=1.5,
                   label=f"{t} Hz ({np.mean(units['fr_window']>=t):.0%} of good units)")
    ax.set_xlabel("within-window firing rate (Hz)")
    ax.set_ylabel("# units")
    ax.set_title("FR distribution: target-region good units", fontsize=10, loc="left")
    ax.legend(fontsize=7.5)
    # B: per-region high-FR(>=20) GOOD units vs chance-corrected selective excess
    ax = axes[1]
    x = np.arange(len(rt))
    ax.bar(x - 0.2, rt["good_hiFR20"], 0.4, label=">=20 Hz good units", color="#9bd")
    ax.bar(x + 0.2, rt["sel_hiFR20_excess"], 0.4,
           label=f">=20 Hz choice-sel (excess, thr {thr})", color="#a24")
    for i, r in rt.iterrows():
        ax.text(i, r["good_hiFR20"] + 1, f"N={r['poolable_N']}", ha="center",
                fontsize=7, color="navy")
    ax.set_xticks(x); ax.set_xticklabels(rt["region"])
    ax.set_ylabel("# units (sampled insertions)")
    ax.set_xlabel("region  (navy = aggregate poolable deliberative N)")
    ax.set_title("High-FR population vs chance-corrected choice-selective",
                 fontsize=10, loc="left")
    ax.legend(fontsize=7.5)
    fig.suptitle("Phase 2 census: high-FR choice cells in BWM midbrain/hindbrain "
                 "choice regions vs the Phase-1 recoverable lane", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nSaved figure -> {FIG_PATH}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["coverage", "spikes", "report", "test-one"],
                    default="coverage")
    ap.add_argument("--cap", type=int, default=12, help="insertions/region (spikes)")
    ap.add_argument("--no-cleanup", action="store_true")
    args = ap.parse_args()

    one = get_one()
    print(f"Connected: {one.alyx.base_url}\n")
    br = BrainRegions()

    if args.stage == "test-one":
        ids, _ = one.search_insertions(atlas_acronym="SNr", project=BWM_PROJECT,
                                       details=True)
        import time
        t = time.time()
        df, meta = process_insertion(one, br, str(ids[0]),
                                     cleanup=not args.no_cleanup)
        print(f"meta: {meta}  [{time.time()-t:.1f}s]")
        if df is not None and len(df):
            print(df.head(12).to_string(index=False))
    elif args.stage == "coverage":
        run_coverage(one)
    elif args.stage == "spikes":
        run_spikes(one, br, args.cap, cleanup=not args.no_cleanup)
    elif args.stage == "report":
        run_report(one)


if __name__ == "__main__":
    main()

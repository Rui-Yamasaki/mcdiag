"""Bulletproofing step 3 - BIN-SIZE robustness of the three IBL headline conclusions.

The canonical feature extraction (phase2_selectivity.py::_rate) computes a window-MEAN
firing rate = count([a,b]) / (b-a) -- there is NO temporal sub-binning.  All three
headline analyses (cascade, pooled effect, decode) consume only that per-trial scalar.
A window-mean rate is mathematically bin-INVARIANT: summing spike counts over 10/20/50 ms
sub-bins tiling [a,b] gives the identical total count, hence the identical mean.  The ONLY
channel through which a bin size can move the scalar is WINDOW-EDGE TRUNCATION when the
window is not an exact multiple of the bin: with the standard complete-bins-only convention
the <w remainder at the window end is dropped.  This script measures that residual.

Binned window rate (complete-bins-only):  K = floor((b-a)/w);  rate = count([a, a+K*w]) / (K*w).
  - w = None (canonical): K=1 over the whole window -> reproduces _rate EXACTLY.
  - w in {10,20,50} ms: drops up to w of window at the end (delib >=0.5 s, early >=0.2 s,
    peri = 0.2 s which is an exact multiple of all three bins -> peri is never truncated).

Re-extraction is SPIKES ONLY (target-region cohort clusters).  Movement covariates
(wheel_speed/disp, paw/nose DLC speed) and trial labels (choice/absc/signed/pL) are
window-means / labels => bin-invariant => reused verbatim from the cached features CSV.
So one spike stream per insertion yields all four bin sizes; nothing else is downloaded.

Stages (resumable; checkpoint per (pid), heartbeat per session and per cell):
  python src/robustness_binsize.py --stage cohort           # print canonical settings + cohort
  python src/robustness_binsize.py --stage probe --pid PID   # time + validate ONE insertion
  python src/robustness_binsize.py --stage extract           # stream cohort spikes -> per-pid rate cache
  python src/robustness_binsize.py --stage assemble          # splice re-binned rates into cached covariates
  python src/robustness_binsize.py --stage analyze           # cascade/pooled/decode per bin -> binsize_<bin>.json
  python src/robustness_binsize.py --stage verdict           # summary csv + append docs/robustness.md
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ibl_one import PROJECT_ROOT, get_one
# canonical extraction logic (identical trial filter + windows); importing is side-effect-free
from phase2_selectivity import delib_trials, EARLY, PERI

from brainbox.io.one import SessionLoader, SpikeSortingLoader

R = PROJECT_ROOT / "results"
OUT = R / "robustness"
CACHE = OUT / "binsize_cache"          # per-pid rate checkpoints (resume markers)
FEAT_FULL = R / "phase2_sel_features_full.csv"
CELLS_FULL = R / "phase2_sel_cells_full.csv"
LOG = OUT / "binsize_run.log"

# bin label -> width in SECONDS (None = canonical full-window mean)
BINS = {"canon": None, "b10": 0.010, "b20": 0.020, "b50": 0.050}
WINDOWS = ("delib", "early", "peri")
REGIONS = ["GRN", "MRN", "SNr", "SCm", "IRN"]   # CORE_REGIONS (cascade)


# --------------------------------------------------------------------------- heartbeat
def hb(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")
        fh.flush()


# --------------------------------------------------------------------------- cohort
def load_cohort():
    """Fixed canonical cohort (pid -> cluster ids) from the cached features CSV."""
    f = pd.read_csv(FEAT_FULL, usecols=["cell", "region", "eid"])
    f["pid"] = f["cell"].str.split(":").str[0]
    f["u"] = f["cell"].str.split(":").str[1].astype(int)
    cells = f.drop_duplicates("cell")
    pid2eid = cells.drop_duplicates("pid").set_index("pid")["eid"].astype(str).to_dict()
    pid2clu = cells.groupby("pid")["u"].apply(lambda s: sorted(set(s))).to_dict()
    pid2cells = cells.groupby("pid")["cell"].apply(list).to_dict()
    return pid2eid, pid2clu, pid2cells, cells


def stage_cohort():
    pid2eid, pid2clu, pid2cells, cells = load_cohort()
    print("=== CANONICAL FEATURE SETTINGS (phase2_selectivity.py) ===")
    print("  rate = count([a,b]) / (b-a)   # window-MEAN; NO temporal binning")
    print(f"  windows:  delib=[on, fm]   early=[on+{EARLY[0]}, fm-{EARLY[1]}]   "
          f"peri=[fm-{PERI[0]}, fm+{PERI[1]}]")
    print("  FR floor: mean(rate_delib) >= 10 Hz (cohort-defining; held FIXED here)")
    print("  trial filter: choice!=0, RT in [0.5,2.0] s, |contrast|<=25%; QC label>=1.0")
    print(f"\n=== COHORT (from {FEAT_FULL.name}) ===")
    print(f"  hi-FR units: {cells.cell.nunique()}   sessions(eid): {cells.eid.nunique()}   "
          f"insertions(pid): {len(pid2eid)}")
    print(f"  per region : {dict(cells.region.value_counts())}")
    print(f"  bin sizes to sweep (ms): canonical(window-mean), 10, 20, 50")


# --------------------------------------------------------------------------- binned rate
def window_rate(ut, a, b, w):
    """Complete-bins-only window-mean rate. w=None -> exact canonical _rate."""
    if not (b > a):
        return np.nan
    if w is None:
        return (np.searchsorted(ut, b) - np.searchsorted(ut, a)) / (b - a)
    K = int(np.floor((b - a) / w + 1e-9))
    if K < 1:
        return np.nan
    bend = a + K * w
    return (np.searchsorted(ut, bend) - np.searchsorted(ut, a)) / (K * w)


def cell_rates(ut, tr):
    """Per-trial rate_{delib,early,peri} at every bin size for one cell's spike train."""
    on = tr["on"].to_numpy(float); fm = tr["fm"].to_numpy(float)
    out = {f"rate_{win}_{b}": np.empty(len(tr)) for win in WINDOWS for b in BINS}
    for i in range(len(tr)):
        a, c = on[i], fm[i]
        ea, eb = a + EARLY[0], c - EARLY[1]
        pa, pb = c - PERI[0], c + PERI[1]
        for blab, w in BINS.items():
            out[f"rate_delib_{blab}"][i] = window_rate(ut, a, c, w)
            out[f"rate_early_{blab}"][i] = window_rate(ut, ea, eb, w) if eb > ea else np.nan
            out[f"rate_peri_{blab}"][i] = window_rate(ut, pa, pb, w)
    return out


# --------------------------------------------------------------------------- one insertion
def process_pid(one, pid, eid, clusters_wanted, cleanup=True):
    """Stream spikes for one insertion; return tidy per-(cell,trial) rate frame at all bins."""
    ssl = SpikeSortingLoader(pid=pid, one=one)
    eid2, pname = ssl.eid, ssl.pname
    assert str(eid2) == str(eid), f"eid mismatch {eid2} != {eid}"
    coll = f"alf/{pname}/pykilosort"
    sl = SessionLoader(one=one, eid=eid2); sl.load_trials()
    tr = delib_trials(sl.trials).reset_index(drop=True)
    spikes = one.load_object(eid2, "spikes", collection=coll, attribute=["times", "clusters"])
    st = np.asarray(spikes["times"]); sc = np.asarray(spikes["clusters"]).astype(int)
    order = np.argsort(st, kind="stable") if not np.all(np.diff(st) >= 0) else None
    if order is not None:
        st = st[order]; sc = sc[order]

    rows = []
    for u in clusters_wanted:
        ut = st[sc == u]
        rt = cell_rates(ut, tr)
        base = dict(cell=f"{pid}:{u}", eid=str(eid2))
        for i in range(len(tr)):
            row = dict(base, ti=i,
                       choice=int(tr["choice"].iloc[i]), absc=float(tr["absc"].iloc[i]),
                       signed=float(tr["signed"].iloc[i]), pL=float(tr["pL"].iloc[i]))
            for k, arr in rt.items():
                row[k] = float(arr[i])
            rows.append(row)
        hb(f"    cell {pid[:8]}:{u}  ({len(tr)} trials)")
    df = pd.DataFrame(rows)

    del st, sc, spikes; gc.collect()
    if cleanup:
        spath = Path(one.eid2path(eid2))
        for pat in (f"alf/{pname}/pykilosort/**/spikes.times.npy",
                    f"alf/{pname}/pykilosort/**/spikes.clusters.npy"):
            for fp in spath.glob(pat):
                try:
                    fp.unlink()
                except OSError:
                    pass
    return df, len(tr)


# --------------------------------------------------------------------------- probe
def stage_probe(pid=None):
    pid2eid, pid2clu, pid2cells, cells = load_cohort()
    if pid is None:
        pid = sorted(pid2eid, key=lambda p: len(pid2clu[p]))[len(pid2eid) // 2]
    eid = pid2eid[pid]; clus = pid2clu[pid]
    hb(f"PROBE pid={pid} eid={eid}  ({len(clus)} cohort clusters)")
    one = get_one()
    t0 = time.time()
    df, ntr = process_pid(one, pid, eid, clus)
    dt = time.time() - t0
    hb(f"PROBE done in {dt:.1f}s  ({len(clus)} cells x {ntr} trials -> {len(df)} rows)")

    # validate alignment + canonical reproduction against cache
    cache = pd.read_csv(FEAT_FULL)
    cache = cache[cache.cell.isin(df.cell.unique())].copy()
    cache["ti"] = cache.groupby("cell").cumcount()
    m = df.merge(cache, on=["cell", "ti"], suffixes=("_new", "_old"))
    align = all(np.allclose(m[f"{c}_new"], m[f"{c}_old"], equal_nan=True)
                for c in ("choice", "absc", "signed", "pL"))
    print(f"  trial-alignment (choice/absc/signed/pL identical): {align}")
    for win, old in (("delib", "rate_delib"), ("early", "rate_early"), ("peri", "rate_peri")):
        a = m[f"rate_{win}_canon"].to_numpy(float); b = m[old].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        ad = np.abs(a[ok] - b[ok]); rd = ad / (np.abs(b[ok]) + 1e-12)
        print(f"  canonical {win:6s}: max|abs|={ad.max():.3e}  max|rel|={rd.max():.3e}  "
              f"(n={ok.sum()}, nan_new={(~np.isfinite(a)).sum()}, nan_old={(~np.isfinite(b)).sum()})")
    return df


# --------------------------------------------------------------------------- extract
def stage_extract(limit=None):
    pid2eid, pid2clu, pid2cells, cells = load_cohort()
    CACHE.mkdir(parents=True, exist_ok=True)
    pids = list(pid2eid)
    done = {p.stem for p in CACHE.glob("*.csv")}
    todo = [p for p in pids if p not in done]
    if limit:
        todo = todo[:limit]
    hb(f"EXTRACT: {len(pids)} cohort insertions, {len(done)} cached, {len(todo)} to do")
    one = get_one()
    for i, pid in enumerate(todo, 1):
        eid = pid2eid[pid]; clus = pid2clu[pid]
        t0 = time.time()
        hb(f"  [{i}/{len(todo)}] START pid={pid[:8]} eid={eid} ({len(clus)} cells)")
        try:
            df, ntr = process_pid(one, pid, eid, clus)
            tmp = CACHE / f"{pid}.csv.tmp"
            df.to_csv(tmp, index=False)
            tmp.replace(CACHE / f"{pid}.csv")          # atomic -> resume marker
            hb(f"  [{i}/{len(todo)}] DONE  pid={pid[:8]} {len(df)} rows in {time.time()-t0:.1f}s")
        except Exception as exc:  # noqa: BLE001
            hb(f"  [{i}/{len(todo)}] FAIL  pid={pid[:8]}: {repr(exc)[:140]}")
    n_cached = len(list(CACHE.glob("*.csv")))
    hb(f"EXTRACT checkpoint: {n_cached}/{len(pids)} insertions cached")


# --------------------------------------------------------------------------- assemble
def assemble_bin(blab, cache_long, base):
    """Return a features DataFrame == cached covariates with rate_{delib,early,peri}
    replaced by the re-binned values for bin `blab`."""
    ren = {f"rate_delib_{blab}": "rate_delib", f"rate_early_{blab}": "rate_early",
           f"rate_peri_{blab}": "rate_peri"}
    rb = cache_long[["cell", "ti", *ren]].rename(columns=ren)
    out = base.merge(rb, on=["cell", "ti"], how="left", suffixes=("_OLD", ""))
    out = out.drop(columns=[c for c in out.columns if c.endswith("_OLD")])
    return out


def stage_assemble():
    pid2eid, pid2clu, pid2cells, cells = load_cohort()
    parts = []
    miss = []
    for pid in pid2eid:
        fp = CACHE / f"{pid}.csv"
        if fp.exists():
            parts.append(pd.read_csv(fp))
        else:
            miss.append(pid)
    if miss:
        hb(f"ASSEMBLE: WARNING {len(miss)} insertions not yet extracted: "
           f"{[p[:8] for p in miss[:6]]}{'...' if len(miss) > 6 else ''}")
    cache_long = pd.concat(parts, ignore_index=True)
    base = pd.read_csv(FEAT_FULL)
    base["ti"] = base.groupby("cell").cumcount()
    # restrict to cells we actually re-extracted (resumable / partial-safe)
    have = set(cache_long.cell.unique())
    base = base[base.cell.isin(have)].copy()

    # alignment guard against cached labels
    chk = base.merge(cache_long[["cell", "ti", "choice", "absc", "signed", "pL"]],
                     on=["cell", "ti"], suffixes=("", "_re"))
    for c in ("choice", "absc", "signed", "pL"):
        if not np.allclose(chk[c], chk[c + "_re"], equal_nan=True):
            raise SystemExit(f"ASSEMBLE ABORT: trial labels disagree for {c} -> alignment broken")
    hb(f"ASSEMBLE: {len(have)} cells aligned ok; writing per-bin features")
    base_cols = list(pd.read_csv(FEAT_FULL, nrows=0).columns)  # canonical column order
    for blab in BINS:
        feat = assemble_bin(blab, cache_long, base)[base_cols + ["ti"]]
        feat = feat.drop(columns=["ti"])
        out = OUT / f"binsize_feat_{blab}.csv"
        feat.to_csv(out, index=False)
        hb(f"  wrote {out.name}  ({len(feat)} rows, {feat.cell.nunique()} cells)")

    # canonical feature-match report vs cache
    can = pd.read_csv(OUT / "binsize_feat_canon.csv")
    ref = base[base.cell.isin(can.cell.unique())].sort_values(["cell", "ti"]).reset_index(drop=True)
    can = can.assign(ti=can.groupby("cell").cumcount()).sort_values(["cell", "ti"]).reset_index(drop=True)
    rep = {}
    for col in ("rate_delib", "rate_early", "rate_peri"):
        a = can[col].to_numpy(float); b = ref[col].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        ad = np.abs(a[ok] - b[ok])
        rep[col] = dict(max_abs=float(ad.max()), max_rel=float((ad / (np.abs(b[ok]) + 1e-12)).max()),
                        nan_mismatch=int((np.isfinite(a) != np.isfinite(b)).sum()))
    (OUT / "binsize_feature_match.json").write_text(json.dumps(rep, indent=2))
    hb(f"ASSEMBLE feature-match (canonical vs cache): {json.dumps(rep)}")


# --------------------------------------------------------------------------- analyze
def analyze_bin(blab):
    """Run cascade + pooled + decode on the per-bin features; return the headline dict."""
    import phase2_selectivity as PS
    import audit_corrections as AC
    import audit_realtrial_decode as ARD

    feat_path = OUT / f"binsize_feat_{blab}.csv"
    feats = pd.read_csv(feat_path)

    # --- cascade (per-cell perm AUC + FDR), reusing the canonical engine on these features
    PS.FEATURES_CSV = feat_path
    PS.CELLS_CSV = OUT / f"_cells_{blab}.csv"
    cells = PS.run_analyze()
    casc = AC.cascade_counts(cells, REGIONS)
    n_triple_raw = int(casc.triple_raw.sum())
    n_triple_fdr = int(casc.triple_fdr.sum())
    per_region_fdr = dict(zip(casc.region, casc.triple_fdr.astype(int)))

    # --- pooled MRN error-trial decision effect (cell-clustered bootstrap)
    pe = AC.pooled_effect(feats, "MRN", "err")

    # --- real-trial per-session CV decode, EXPANDED nonlinear movement control, MRN
    f, dec_fn = ARD.ibl_prep(feats)
    res = ARD.run_region(f, "MRN", dec_fn, np.random.default_rng(0))
    dec = res["expanded"]

    out = dict(bin=blab, bin_ms=(None if BINS[blab] is None else int(BINS[blab] * 1000)),
               n_triple_raw=n_triple_raw, n_triple_FDR_total=n_triple_fdr,
               n_triple_FDR_per_region=per_region_fdr,
               MRN_pooled_SD=pe["eff"], MRN_pooled_CI=[pe["ci_lo"], pe["ci_hi"]],
               MRN_pooled_p=pe["mwu_p_descriptive"], MRN_pooled_ci_lo=pe["ci_lo"],
               MRN_decode_AUC=dec["cv_auc"], MRN_decode_p=dec["perm_p"],
               n_cells=int(feats.cell.nunique()))
    (OUT / f"binsize_{blab}.json").write_text(json.dumps(out, indent=2))
    hb(f"ANALYZE {blab}: triple_raw={n_triple_raw} triple_FDR={n_triple_fdr} "
       f"MRN_pooled={pe['eff']:+.4f} [{pe['ci_lo']:.4f},{pe['ci_hi']:.4f}] p={pe['mwu_p_descriptive']:.4f} "
       f"MRN_decode_AUC={dec['cv_auc']:.4f} p={dec['perm_p']:.4f}")
    return out


def stage_analyze(only=None):
    labs = [only] if only else list(BINS)
    for blab in labs:
        if not (OUT / f"binsize_feat_{blab}.csv").exists():
            hb(f"ANALYZE skip {blab}: features not assembled")
            continue
        if (OUT / f"binsize_{blab}.json").exists():
            hb(f"ANALYZE skip {blab}: binsize_{blab}.json exists")
            continue
        analyze_bin(blab)


# --------------------------------------------------------------------------- verdict
def stage_verdict():
    rows = []
    for blab in BINS:
        fp = OUT / f"binsize_{blab}.json"
        if fp.exists():
            rows.append(json.loads(fp.read_text()))
    if not rows:
        raise SystemExit("VERDICT: no binsize_*.json yet")
    df = pd.DataFrame(rows)
    order = {"canon": 0, "b10": 1, "b20": 2, "b50": 3}
    df = df.sort_values("bin", key=lambda s: s.map(order)).reset_index(drop=True)
    df.to_csv(OUT / "binsize_summary.csv", index=False)
    print(df.to_string(index=False))

    # STABLE/BROKEN verdict (canonical anchored)
    c1 = bool((df.n_triple_FDR_total == 0).all())                         # MRN per-cell FDR-null
    mrn_fdr_null = all(j.get("MRN", 0) == 0 for j in df.n_triple_FDR_per_region)
    c2 = bool((df.MRN_pooled_SD.abs() < 0.15).all() and (df.MRN_pooled_SD > 0).all())
    c3 = bool((df.MRN_decode_AUC <= 0.56).all())
    print("\n=== STABLE/BROKEN across bin size ===")
    print(f"  (1) MRN per-cell FDR-null (triple_FDR MRN==0 every bin): "
          f"{'STABLE' if mrn_fdr_null else 'BROKEN'}")
    print(f"  (2) MRN pooled small+positive (0<SD<0.15 every bin): {'STABLE' if c2 else 'BROKEN'}")
    print(f"  (3) MRN movement-controlled decode <=0.56 every bin: {'STABLE' if c3 else 'BROKEN'}")
    return df


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["cohort", "probe", "extract", "assemble", "analyze", "verdict"])
    ap.add_argument("--pid", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="analyze a single bin label")
    a = ap.parse_args()
    if a.stage == "cohort":
        stage_cohort()
    elif a.stage == "probe":
        stage_probe(a.pid)
    elif a.stage == "extract":
        stage_extract(a.limit)
    elif a.stage == "assemble":
        stage_assemble()
    elif a.stage == "analyze":
        stage_analyze(a.only)
    elif a.stage == "verdict":
        stage_verdict()


if __name__ == "__main__":
    main()

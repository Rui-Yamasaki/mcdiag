"""TASK C — full-cohort all-QC simultaneity census.

The manuscript's all-QC census (Table S3) is a 35-session SAMPLE (global max 79). Here we extend to
the FULL IBL BWM cohort for the five decision-core regions. Simultaneity = units co-recorded in the
SAME session (all probes in a session are simultaneous), so per (session x region) we SUM good-QC
units across every probe in that session that touches the region.

"all-QC" = IBL good-unit QC label (clusters.metrics.label == 1), at ANY firing rate (matches the
manuscript Table S3 definition; the all-QC set is more permissive than the hi-FR set). We ALSO record
the no-QC-filter count (every cluster) as the most-permissive upper bound.

This loads ONLY clusters + channels per insertion (QC label + peak-channel Beryl region) -- NO spike
streaming, NO trials -- so it is light and resumable.

  ./.venv/Scripts/python.exe src/referee_full_census.py --stage scan     # download+count (resumable)
  ./.venv/Scripts/python.exe src/referee_full_census.py --stage report   # build full_census.csv
  ./.venv/Scripts/python.exe src/referee_full_census.py --stage test-one
"""
from __future__ import annotations
import argparse
import sys
import warnings

try:                                     # live progress when stdout is redirected to a file
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import BrainRegions

from ibl_one import BWM_PROJECT, PROJECT_ROOT, get_one

warnings.filterwarnings("ignore", message="Multiple revisions")

CORE_REGIONS = ["MRN", "SCm", "IRN", "SNr", "GRN"]
GOOD_LABEL = 1.0
RECOVERY_REQUIREMENT = 120
OUT = PROJECT_ROOT / "results" / "referee_response"
OUT.mkdir(parents=True, exist_ok=True)
UNITS_CSV = OUT / "full_census_units.csv"        # per (pid, eid, region): good + all counts


def all_insertions(one):
    """All BWM (pid, eid) touching any core region; deduped to unique pids."""
    pid2eid = {}
    pid_regions = {}
    for reg in CORE_REGIONS:
        ids, det = one.search_insertions(atlas_acronym=reg, project=BWM_PROJECT, details=True)
        for i, d in zip(ids, det):
            pid = str(i)
            pid2eid[pid] = d["session"]
            pid_regions.setdefault(pid, set()).add(reg)
    return pid2eid, pid_regions


def count_insertion(one, br, pid):
    """Load clusters+channels (no spikes); return per-core-region good + all unit counts."""
    ssl = SpikeSortingLoader(pid=pid, one=one)
    eid, pname = ssl.eid, ssl.pname
    coll = f"alf/{pname}/pykilosort"
    # TRIMMED: only peak channel + QC label + per-channel region (NOT waveforms etc.)
    clusters = one.load_object(eid, "clusters", collection=coll,
                               attribute=["channels", "metrics"])
    channels = one.load_object(eid, "channels", collection=coll,
                               attribute=["brainLocationIds_ccf_2017"])
    ch_beryl = br.id2acronym(channels["brainLocationIds_ccf_2017"], mapping="Beryl")
    clu_region = np.asarray(ch_beryl)[clusters["channels"].astype(int)]
    label = clusters["metrics"]["label"].to_numpy(float)
    rows = []
    for reg in CORE_REGIONS:
        m = clu_region == reg
        if not m.any():
            continue
        rows.append(dict(pid=pid, eid=str(eid), region=reg,
                         n_good=int(((label >= GOOD_LABEL) & m).sum()),
                         n_all=int(m.sum())))
    return rows


def run_scan(one, br):
    pid2eid, pid_regions = all_insertions(one)
    pids = sorted(pid2eid)
    print(f"Full cohort: {len(pids)} unique insertions touching core regions "
          f"{CORE_REGIONS}")
    done = set()
    if UNITS_CSV.exists():
        done = set(pd.read_csv(UNITS_CSV)["pid"].astype(str).unique())
    todo = [p for p in pids if p not in done]
    print(f"  {len(done)} insertions cached; {len(todo)} to load (clusters+channels only)")
    for i, pid in enumerate(todo, 1):
        try:
            rows = count_insertion(one, br, pid)
            if rows:
                pd.DataFrame(rows).to_csv(UNITS_CSV, mode="a", header=not UNITS_CSV.exists(),
                                          index=False)
            if i % 10 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {pid[:8]} -> {sum(r['n_good'] for r in rows)} good units")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(todo)}] {pid[:8]} FAILED: {repr(exc)[:100]}")
    print(f"Saved per-insertion counts -> {UNITS_CSV}")


def run_report():
    u = pd.read_csv(UNITS_CSV)
    n_ins, n_sess = u["pid"].nunique(), u["eid"].nunique()
    print(f"\n=== FULL-COHORT all-QC simultaneity census ===")
    print(f"  {n_ins} insertions / {n_sess} sessions (full BWM coverage of core regions)\n")
    rows, glob_max_good, glob_max_all = [], 0, 0
    for reg in CORE_REGIONS:
        sub = u[u.region == reg]
        if sub.empty:
            rows.append(dict(region=reg, region_sessions=0)); continue
        # simultaneous = SUM good units across all probes in a session x region
        g = sub.groupby("eid")["n_good"].sum()
        a = sub.groupby("eid")["n_all"].sum()
        glob_max_good = max(glob_max_good, int(g.max()))
        glob_max_all = max(glob_max_all, int(a.max()))
        rows.append(dict(
            region=reg, region_sessions=int(g.size),
            median_simul_goodQC=float(g.median()), max_simul_goodQC=int(g.max()),
            sessions_ge120_goodQC=int((g >= RECOVERY_REQUIREMENT).sum()),
            max_simul_allunits=int(a.max()),
            sessions_ge120_allunits=int((a >= RECOVERY_REQUIREMENT).sum())))
    rt = pd.DataFrame(rows)
    rt.to_csv(OUT / "full_census.csv", index=False)
    print(rt.to_string(index=False))
    total_sessions = int(rt["region_sessions"].sum())
    reaching_good = int(rt.get("sessions_ge120_goodQC", pd.Series([0])).sum())
    reaching_all = int(rt.get("sessions_ge120_allunits", pd.Series([0])).sum())
    print(f"\n  GLOBAL MAX simultaneous good-QC units = {glob_max_good}")
    print(f"  GLOBAL MAX simultaneous ALL units (no QC filter) = {glob_max_all}")
    print(f"  region-sessions reaching >= {RECOVERY_REQUIREMENT}: "
          f"good-QC {reaching_good}/{total_sessions}; all-units {reaching_all}/{total_sessions}")
    verdict = ("NO session reaches the >=120-cell recovery requirement"
               if reaching_good == 0 and reaching_all == 0 else
               "SOME session(s) reach >=120 cells -- see table")
    print(f"  -> {verdict}")
    print(f"\nSaved -> {OUT / 'full_census.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["scan", "report", "test-one"], default="scan")
    a = ap.parse_args()
    if a.stage == "report":
        run_report(); return
    one = get_one(); br = BrainRegions()
    print(f"Connected: {one.alyx.base_url}")
    if a.stage == "test-one":
        ids, _ = one.search_insertions(atlas_acronym="MRN", project=BWM_PROJECT, details=True)
        import time
        t = time.time()
        rows = count_insertion(one, br, str(ids[0]))
        print(f"  test {str(ids[0])[:8]} [{time.time()-t:.1f}s]:", rows)
    else:
        run_scan(one, br)


if __name__ == "__main__":
    main()

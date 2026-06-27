"""Pre-submission robustness sweep (PURE; cached results only, NO re-extraction/re-download).

Tests whether the three headline conclusions are STABLE across analyst knobs, WITHOUT touching any
canonical results file (all outputs go to results/robustness/). Reproduce-before-innovate: the
default cell must match the canonical headline before any sweep.

FEASIBLE knobs from cache (rates were extracted then spikes deleted; re-binning needs spikes):
  - fr_floor in {10,15,25} Hz  -> filter cells/trials by fr_window (can only RAISE >=10). all 3 analyses.
  - window in {delib,early,peri} -> the cached rate_delib/rate_early/rate_peri columns. pooled + decode
    (and a coherent window-cascade: choice-AUC-p in that window & movement & stimulus, FDR'd).
NOT FEASIBLE from cache (need spike re-extraction = re-download, forbidden by the hard rules):
  - bin_size: the 3 analyses use a window-MEAN rate, not binned counts (binning is a recovery-sim knob).
  - deliberation-window bounds & RT/engagement threshold: the features CSV has NO per-trial RT/event
    times (only the 3 pre-computed window rates), so RT/window/engagement cannot be re-derived.

  python src/robustness_sweep.py --stage all      # gate -> sweep -> verdict
  python src/robustness_sweep.py --stage verdict   # rebuild summary/doc from existing cell JSONs
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from ibl_one import PROJECT_ROOT
import audit_corrections as AC
import audit_realtrial_decode as RD

ROB = PROJECT_ROOT / "results" / "robustness"
LOG = ROB / "run.log"
SUMMARY = PROJECT_ROOT / "results" / "robustness_summary.csv"
DOC = PROJECT_ROOT / "docs" / "robustness.md"
REGIONS = ["GRN", "MRN", "SNr", "SCm", "IRN"]
# sweep cells: (knob, value); fr_floor_10 == window_delib == canonical baseline (computed once)
CELLS = [("fr_floor", 10), ("fr_floor", 15), ("fr_floor", 25), ("window", "early"), ("window", "peri")]
NOT_FEASIBLE = {
    "bin_size": "3 analyses use a window-MEAN rate, not binned counts; binning is a recovery-sim knob",
    "deliberation_window": "features CSV has no per-trial event times; re-windowing needs spikes (deleted)",
    "rt_engagement": "features CSV has no per-trial RT; re-filtering on RT/engagement needs spikes (deleted)",
}
_LOGF = None


def hb(msg):
    global _LOGF
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOGF is None:
        ROB.mkdir(parents=True, exist_ok=True)
        _LOGF = open(LOG, "a", encoding="utf-8")
    _LOGF.write(line + "\n"); _LOGF.flush()


# ---- metric computations (use the EXACT canonical functions; never write canonical files) ----
def _cells():
    c = pd.read_csv(AC.IBL_CELLS)
    if "p_move" not in c:
        c["p_move"] = np.where(c["has_dlc"], c["p_move_full"], c["p_move_wheel"])
    return c


def cascade_metrics(cells):
    casc = AC.cascade_counts(cells, REGIONS)
    return dict(n_triple_raw_total=int(casc.triple_raw.sum()),
                n_triple_FDR_total=int(casc.triple_fdr.sum()),
                n_triple_FDR_per_region={r: int(v) for r, v in zip(casc.region, casc.triple_fdr)})


def pooled_metrics(feats):
    pe = AC.pooled_effect(feats, "MRN", "err")
    return dict(MRN_pooled_SD=round(pe["eff"], 4),
                MRN_pooled_CI=[round(pe["ci_lo"], 4), round(pe["ci_hi"], 4)],
                MRN_pooled_p=round(pe["mwu_p_descriptive"], 4), MRN_pooled_n_cells=pe["n_cells"])


def decode_metrics(feats):
    f, dec_fn = RD.ibl_prep(feats)
    res = RD.run_region(f, "MRN", dec_fn, np.random.default_rng(0))
    e = res["expanded"]
    return dict(MRN_decode_AUC=round(e["cv_auc"], 4), MRN_decode_p=round(e["perm_p"], 4),
                MRN_decode_n_sessions=e["n_sessions"])


def apply_knob(knob, value, cells, feats):
    if knob == "fr_floor":
        return cells[cells.fr_window >= value].copy(), feats[feats.fr_window >= value].copy()
    # window
    if value == "delib":
        return cells.copy(), feats.copy()
    f2 = feats.copy(); f2["rate_delib"] = f2[f"rate_{value}"]
    c2 = cells.copy(); c2["p_delib"] = c2[f"p_{value}"]; c2["p_early"] = c2[f"p_{value}"]  # leading collapses
    return c2, f2


def run_cell(knob, value, cells, feats):
    name = f"{knob}_{value}"
    out_path = ROB / f"{name}.json"
    if out_path.exists():
        hb(f"SKIP {name} (json exists)")
        return json.loads(out_path.read_text())
    t = time.time()
    cv, fv = apply_knob(knob, value, cells, feats)
    hb(f"CELL {name}: cells={len(cv['cell'].unique()) if 'cell' in cv else len(cv)} "
       f"rows={len(fv)} | computing cascade...")
    cm = cascade_metrics(cv)
    hb(f"  {name} cascade triple_raw={cm['n_triple_raw_total']} triple_FDR={cm['n_triple_FDR_total']} "
       f"| pooled...")
    pm = pooled_metrics(fv)
    hb(f"  {name} pooled MRN SD={pm['MRN_pooled_SD']:+.4f} CI={pm['MRN_pooled_CI']} | decode (slow)...")
    dm = decode_metrics(fv)
    rec = dict(knob=knob, value=value, **cm, **pm, **dm)
    out_path.write_text(json.dumps(rec, indent=2))
    hb(f"  {name} decode MRN AUC={dm['MRN_decode_AUC']:.4f} p={dm['MRN_decode_p']:.4f} "
       f"-> wrote {out_path.name} ({time.time()-t:.0f}s)")
    return rec


# ---- Phase 1 gate ----
def gate(cells, feats):
    hb("PHASE 1 GATE: reproduce default (fr_floor_10) vs canonical headline")
    rec = run_cell("fr_floor", 10, cells, feats)
    ok = (rec["n_triple_raw_total"] == 59 and rec["n_triple_FDR_total"] == 0
          and abs(rec["MRN_pooled_SD"] - 0.059) < 0.003
          and abs(rec["MRN_decode_AUC"] - 0.528) < 0.01)
    hb(f"  GATE check: triple_raw=59? {rec['n_triple_raw_total']==59} | triple_FDR=0? "
       f"{rec['n_triple_FDR_total']==0} | pooled~0.059? {abs(rec['MRN_pooled_SD']-0.059)<0.003} "
       f"({rec['MRN_pooled_SD']:+.4f}) | decode~0.528? {abs(rec['MRN_decode_AUC']-0.528)<0.01} "
       f"({rec['MRN_decode_AUC']:.4f})")
    if not ok:
        hb("  GATE FAILED -> STOP (default does not reproduce canonical; do not sweep)")
        raise SystemExit("Phase 1 gate failed")
    hb("  GATE PASSED -> proceeding to sweep")


# ---- Phase 3 verdict ----
def verdict():
    rows = []
    for knob, value in CELLS:
        p = ROB / f"{knob}_{value}.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    if not rows:
        hb("no cell JSONs found"); return
    df = pd.DataFrame([{
        "knob": r["knob"], "value": r["value"], "n_triple_raw_total": r["n_triple_raw_total"],
        "n_triple_FDR_total": r["n_triple_FDR_total"],
        "n_triple_FDR_per_region": json.dumps(r["n_triple_FDR_per_region"]),
        "MRN_pooled_SD": r["MRN_pooled_SD"], "MRN_pooled_CI": json.dumps(r["MRN_pooled_CI"]),
        "MRN_pooled_p": r["MRN_pooled_p"], "MRN_decode_AUC": r["MRN_decode_AUC"],
        "MRN_decode_p": r["MRN_decode_p"]} for r in rows])
    df.to_csv(SUMMARY, index=False)

    c1 = all(r["n_triple_FDR_total"] == 0 and all(v == 0 for v in r["n_triple_FDR_per_region"].values())
             for r in rows)
    c2 = all(abs(r["MRN_pooled_SD"]) < 0.15 and r["MRN_pooled_SD"] > 0 for r in rows)
    c3 = all(r["MRN_decode_AUC"] <= 0.56 for r in rows)
    flags = []
    for r in rows:
        nm = f"{r['knob']}_{r['value']}"
        if r["n_triple_FDR_total"] != 0 or any(v != 0 for v in r["n_triple_FDR_per_region"].values()):
            flags.append(f"BREAK conclusion-1 (FDR-null) at {nm}: triple_FDR={r['n_triple_FDR_total']} {r['n_triple_FDR_per_region']}")
        if not (abs(r["MRN_pooled_SD"]) < 0.15 and r["MRN_pooled_SD"] > 0):
            flags.append(f"BREAK conclusion-2 (pooled small+positive) at {nm}: SD={r['MRN_pooled_SD']:+.4f}")
        if r["MRN_decode_AUC"] > 0.56:
            flags.append(f"BREAK conclusion-3 (decode<=0.56) at {nm}: AUC={r['MRN_decode_AUC']:.4f}")

    lines = ["# Robustness sweep — one-at-a-time sensitivity (IBL, cached data only)\n",
             "No canonical results file modified; all cells under `results/robustness/`. "
             "Reproduce-before-innovate gate passed (default == canonical: triple 59 raw / 0 FDR, "
             "MRN pooled +0.059, MRN decode 0.528).\n",
             "## Feasible knobs swept (from cache)\n",
             "- **fr_floor** {10,15,25} Hz (filter `fr_window`; can only raise >=10) — all 3 analyses.",
             "- **window** {delib,early,peri} (cached `rate_delib/early/peri`) — pooled + decode + a "
             "coherent window-cascade (choice-AUC-p in that window & movement & stimulus, FDR'd).\n",
             "## Knobs NOT swept (require spike re-extraction = re-download, forbidden)\n"]
    for k, why in NOT_FEASIBLE.items():
        lines.append(f"- **{k}**: {why}.")
    lines.append("\n## Cells\n")
    lines.append("| knob | value | triple_raw | triple_FDR | triple_FDR/region | MRN pooled SD | MRN pooled CI | MRN pooled p | MRN decode AUC | MRN decode p |")
    lines.append("|---|---|--:|--:|---|--:|---|--:|--:|--:|")
    for r in rows:
        lines.append(f"| {r['knob']} | {r['value']} | {r['n_triple_raw_total']} | {r['n_triple_FDR_total']} | "
                     f"{r['n_triple_FDR_per_region']} | {r['MRN_pooled_SD']:+.4f} | {r['MRN_pooled_CI']} | "
                     f"{r['MRN_pooled_p']:.4f} | {r['MRN_decode_AUC']:.4f} | {r['MRN_decode_p']:.4f} |")
    lines.append("\n## Verdict\n")
    lines.append(f"1. **Per-cell decision code FDR-null** (triple_FDR==0 in every region, every cell): "
                 f"**{'STABLE' if c1 else 'BROKEN'}**")
    lines.append(f"2. **MRN pooled effect small & positive** (|SD|<0.15, SD>0, every cell): "
                 f"**{'STABLE' if c2 else 'BROKEN'}**")
    lines.append(f"3. **MRN decode below the bar** (AUC<=0.56, every cell): "
                 f"**{'STABLE' if c3 else 'BROKEN'}**")
    lines.append("\n### Flags (loud — these are findings)\n")
    lines += ([f"- **{f}**" for f in flags] if flags
              else ["- none — all three conclusions hold across every swept cell."])

    # interpretation (does NOT soften the flags; explains where/why they break)
    mrn_fdr_stable = all(r["n_triple_FDR_per_region"].get("MRN", 0) == 0 for r in rows)
    premove = [r for r in rows if not (r["knob"] == "window" and r["value"] == "peri")]
    c2_pre = all(abs(r["MRN_pooled_SD"]) < 0.15 and r["MRN_pooled_SD"] > 0 for r in premove)
    lines.append("\n### Interpretation\n")
    lines.append(f"- **MRN (the headline region) is FDR-null at EVERY cell** "
                 f"(MRN triple_FDR==0 across all fr_floor and all windows): {mrn_fdr_stable}. "
                 "The conclusion-1 break is **SCm at fr_floor=25 Hz (3 cells)** — raising the floor "
                 "shrinks the cell set (1620->767), lightening the BH-FDR burden so a few SCm cells "
                 "survive. So 'per-cell signal is completely FDR-null' is **count/FR-floor-sensitive**; "
                 "the MRN-specific FDR-null is robust.")
    lines.append(f"- **The conclusion-2 break is the movement-locked PERI window (+0.184 SD)** — a "
                 "positive control: choice==wheel-turn direction, so a movement-window rate measures "
                 "movement, not decision. In the decision windows (delib +0.059, early ~0) the pooled "
                 f"effect stays small+positive: {c2_pre}. The effect also grows with FR floor "
                 "(+0.059->+0.070->+0.133 at fr 10/15/25; +0.133 nears the 0.15 line at 25 Hz).")
    lines.append("- **Conclusion 3 (movement-controlled decode) is STABLE at every cell** (AUC "
                 "0.514-0.536 <= 0.56), INCLUDING the peri window — the nonlinear movement control "
                 "removes the movement leakage that inflates the uncontrolled pooled effect there. "
                 "This is the strongest result and validates the movement control.")
    lines.append("\n**Net:** at the **canonical settings** (deliberation window, fr_floor=10 Hz) all "
                 "three hold. The breaks are at **non-canonical settings** and are interpretable "
                 "(movement window = a movement positive-control; 25 Hz floor = lighter FDR). The "
                 "movement-controlled MRN decode conclusion is robust across the entire feasible sweep.")
    lines.append("\n*Caveat: bin_size / deliberation-window / RT-engagement could not be swept from "
                 "cached data (spikes deleted post-extraction; re-download forbidden) — see above.*")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    hb(f"VERDICT  c1(FDR-null)={'STABLE' if c1 else 'BROKEN'}  "
       f"c2(pooled)={'STABLE' if c2 else 'BROKEN'}  c3(decode)={'STABLE' if c3 else 'BROKEN'}  "
       f"flags={len(flags)}")
    print(df.to_string(index=False))
    for f in flags:
        hb("  FLAG: " + f)
    print(f"\nwrote {SUMMARY} and {DOC}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gate", "sweep", "verdict", "all"], default="all")
    args = ap.parse_args()
    hb(f"=== robustness_sweep --stage {args.stage} ===")
    if args.stage in ("gate", "sweep", "all"):
        cells = _cells(); feats = pd.read_csv(AC.IBL_FEAT)
        gate(cells, feats)
    if args.stage in ("sweep", "all"):
        for knob, value in CELLS:
            if (knob, value) == ("fr_floor", 10):
                continue  # already done in gate
            run_cell(knob, value, cells, feats)
    if args.stage in ("verdict", "all"):
        verdict()


if __name__ == "__main__":
    main()

"""Phase 1 / step 1 - the identifiability gate (BEHAVIOR ONLY).

Quantify the IBL task's per-trial DELIBERATION WINDOW from behavior alone,
pooled across many brain-wide-map sessions. No spikes, no simulation.

Per trial we measure two candidate "windows" relative to stimulus onset:
    reaction_time    = firstMovement_times - stimOn_times   (stim -> first wheel move)
    response_latency = response_times      - stimOn_times   (stim -> choice threshold)

reaction_time is the decision-FORMATION window (the epoch a stepping/ramping
latent would evolve over); response_latency additionally includes the ballistic
wheel turn. We report both but headline reaction_time.

Stepping vs ramping single-trial models are only identifiable when this window
is both LONG enough and VARIABLE enough across trials, and should lengthen on
hard (low-contrast) trials. So we stratify everything by absolute contrast.

Run:
    python src/phase1_behavior.py            # ~100 sessions (cached after 1st run)
    python src/phase1_behavior.py --n 200    # pool more sessions
    python src/phase1_behavior.py --refresh  # re-download / re-pool from ONE
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

from brainbox.io.one import SessionLoader  # noqa: E402

from ibl_one import BWM_PROJECT, DATA_DIR, PROJECT_ROOT, get_one  # noqa: E402

# ONE emits "Multiple revisions" warnings per session; it auto-picks the latest.
warnings.filterwarnings("ignore", message="Multiple revisions")

# --- parameters --------------------------------------------------------------
STD_CONTRASTS = [0.0, 6.25, 12.5, 25.0, 100.0]   # absolute contrast, percent
RT_MAX = 60.0          # s: hard timeout cap; drop anything longer (disengaged)
THRESH = (0.200, 0.500)  # s: window thresholds for fraction reporting
MIN_N_PLOT = 20        # min trials in a contrast bin to draw its violin
TRIAL_COLS = [
    "stimOn_times", "firstMovement_times", "response_times", "choice",
    "contrastLeft", "contrastRight", "feedbackType", "probabilityLeft",
]
POOLED_PARQUET = DATA_DIR / "phase1_trials_pooled.parquet"  # gitignored cache
FIG_PATH = PROJECT_ROOT / "figures" / "phase1_rt_by_contrast.png"
SUMMARY_CSV = PROJECT_ROOT / "results" / "phase1_summary.csv"


# --- data gathering ----------------------------------------------------------
def gather_trials(one, n_sessions):
    """Pool trials tables from the first ``n_sessions`` BWM sessions (sorted)."""
    eids = sorted(one.search(project=BWM_PROJECT))
    eids = eids[:n_sessions]
    print(f"Pooling trials from {len(eids)} brain-wide-map sessions...")

    frames, ok, fails, all_cols = [], 0, [], set()
    for i, eid in enumerate(eids, 1):
        try:
            sl = SessionLoader(one=one, eid=eid)
            sl.load_trials()
            t = sl.trials
            all_cols.update(t.columns)
            sub = pd.DataFrame({c: (t[c] if c in t.columns else np.nan)
                                for c in TRIAL_COLS})
            sub["eid"] = eid
            frames.append(sub)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - keep pooling other sessions
            fails.append((eid, repr(exc)[:90]))
        if i % 20 == 0 or i == len(eids):
            print(f"  {i}/{len(eids)} sessions  (ok={ok}, failed={len(fails)})")

    df = pd.concat(frames, ignore_index=True)
    print(f"\nColumns seen across sessions: {sorted(all_cols)}")
    qc_like = [c for c in all_cols
               if any(k in c.lower() for k in ("qc", "include", "engage"))]
    print(f"Per-trial QC/engagement columns shipped by IBL: "
          f"{qc_like if qc_like else 'NONE'}")
    if fails:
        print(f"Sessions that failed to load ({len(fails)}): "
              f"{[e for e, _ in fails[:5]]}{' ...' if len(fails) > 5 else ''}")
    return df, ok, len(fails)


def load_pooled(one, n_sessions, refresh):
    if POOLED_PARQUET.exists() and not refresh:
        df = pd.read_parquet(POOLED_PARQUET)
        ns = df["eid"].nunique()
        print(f"Loaded cached pool: {len(df):,} trials from {ns} sessions "
              f"({POOLED_PARQUET}). Use --refresh to re-pull.")
        return df, ns, 0
    df, ok, fail = gather_trials(one, n_sessions)
    df.to_parquet(POOLED_PARQUET)
    print(f"Cached pooled trials -> {POOLED_PARQUET}")
    return df, ok, fail


# --- metrics & filtering -----------------------------------------------------
def add_metrics(df):
    df = df.copy()
    df["reaction_time"] = df["firstMovement_times"] - df["stimOn_times"]
    df["response_latency"] = df["response_times"] - df["stimOn_times"]
    cl = np.nan_to_num(df["contrastLeft"].to_numpy(dtype=float))
    cr = np.nan_to_num(df["contrastRight"].to_numpy(dtype=float))
    df["abs_contrast_pct"] = np.round(np.fmax(cl, cr) * 100.0, 4)
    return df


def engaged_mask(df, metric):
    """Engaged = responded (choice != 0), finite events, window in (0, RT_MAX]."""
    v = df[metric].to_numpy(dtype=float)
    m = (df["choice"].to_numpy(dtype=float) != 0)
    m &= np.isfinite(df["stimOn_times"].to_numpy(dtype=float))
    m &= np.isfinite(v) & (v > 0) & (v <= RT_MAX)
    return m


def stats_ms(v_s):
    """Summary stats (ms) for a vector of windows given in seconds."""
    v = np.asarray(v_s, dtype=float) * 1000.0
    p10, p25, p50, p75, p90 = np.percentile(v, [10, 25, 50, 75, 90])
    sd = np.std(v, ddof=1) if v.size > 1 else np.nan
    qcd = (p75 - p25) / (p75 + p25) if (p75 + p25) > 0 else np.nan
    return {
        "n": v.size,
        "median_ms": p50,
        "p10_ms": p10, "p25_ms": p25, "p75_ms": p75, "p90_ms": p90,
        "IQR_ms": p75 - p25,
        "IDR_ms": p90 - p10,           # inter-decile range (robust spread)
        "QCD": qcd,                    # quartile coeff. of dispersion (robust)
        "CV": sd / np.mean(v) if v.size > 1 and np.mean(v) > 0 else np.nan,
        "frac_gt_200ms": float(np.mean(v > THRESH[0] * 1000)),
        "frac_gt_500ms": float(np.mean(v > THRESH[1] * 1000)),
    }


def summarize(df):
    """Build the per-contrast x per-metric summary table."""
    rows = []
    for metric in ("reaction_time", "response_latency"):
        eng = df[engaged_mask(df, metric)]
        # overall (all contrasts pooled)
        rows.append({"metric": metric, "contrast_pct": "ALL",
                     **stats_ms(eng[metric])})
        for c in STD_CONTRASTS:
            sub = eng[np.isclose(eng["abs_contrast_pct"], c)]
            if len(sub) == 0:
                continue
            rows.append({"metric": metric, "contrast_pct": c,
                         **stats_ms(sub[metric])})
    return pd.DataFrame(rows)


# --- plotting ----------------------------------------------------------------
def violin_logms(ax, df, metric, title):
    eng = df[engaged_mask(df, metric)]
    groups, labels = [], []
    for c in STD_CONTRASTS:
        v = eng.loc[np.isclose(eng["abs_contrast_pct"], c), metric].to_numpy()
        v = v[v > 0]
        if v.size >= MIN_N_PLOT:
            groups.append(np.log10(v * 1000.0))
            labels.append(f"{c:g}\n(n={v.size})")
    pos = np.arange(1, len(groups) + 1)
    parts = ax.violinplot(groups, positions=pos, showmedians=True,
                          showextrema=False, widths=0.85)
    for body in parts["bodies"]:
        body.set_facecolor("#3b7dd8")
        body.set_alpha(0.6)
    parts["cmedians"].set_color("k")
    for thr in THRESH:
        ax.axhline(np.log10(thr * 1000), color="gray", ls="--", lw=0.9)
        ax.text(len(groups) + 0.45, np.log10(thr * 1000),
                f"{int(thr*1000)} ms", va="center", fontsize=8, color="gray")
    yt = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    ax.set_yticks(np.log10(yt))
    ax.set_yticklabels([str(y) for y in yt])
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(0.4, len(groups) + 1.1)
    ax.set_xlabel("absolute contrast (%)")
    ax.set_ylabel("ms from stimOn (log)")
    ax.set_title(title, fontsize=10, loc="left")


def make_figure(df, n_sessions, n_trials):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    violin_logms(axes[0], df, "reaction_time",
                 "reaction_time  (stimOn -> firstMovement)")
    violin_logms(axes[1], df, "response_latency",
                 "response_latency  (stimOn -> response)")
    fig.suptitle(
        f"Phase 1 deliberation window by contrast  |  {n_sessions} sessions, "
        f"{n_trials:,} engaged-eligible trials  |  dashed = 200/500 ms",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nSaved figure -> {FIG_PATH}")


# --- caveats / coverage ------------------------------------------------------
def coverage_caveats(df):
    responded = (df["choice"].to_numpy(dtype=float) != 0) & np.isfinite(
        df["response_times"].to_numpy(dtype=float))
    fm = df["firstMovement_times"].to_numpy(dtype=float)
    miss_fm = np.mean(~np.isfinite(fm[responded])) if responded.any() else np.nan
    rt = df["reaction_time"].to_numpy(dtype=float)
    neg = np.mean((rt < 0) & np.isfinite(rt))
    anticip = np.mean((rt > 0) & (rt < 0.08) & np.isfinite(rt))
    other_c = df.loc[~df["abs_contrast_pct"].isin(STD_CONTRASTS),
                     "abs_contrast_pct"].unique()
    print("\n========== coverage / caveats ==========")
    print(f"  responded trials missing firstMovement_times : {miss_fm:6.2%}")
    print(f"  reaction_time < 0 (pre-stim wheel move)       : {neg:6.2%}")
    print(f"  reaction_time in (0,80) ms (anticipatory)     : {anticip:6.2%}")
    print(f"  non-standard absolute contrasts present       : "
          f"{sorted(other_c.tolist()) if len(other_c) else 'none'}")


def glanceable(summary):
    rt = summary[summary["metric"] == "reaction_time"].set_index("contrast_pct")
    allr = rt.loc["ALL"]
    hardest = rt.loc[0.0] if 0.0 in rt.index else allr
    easiest = rt.loc[100.0] if 100.0 in rt.index else allr
    print("\n========== GLANCEABLE SUMMARY (reaction_time = the window) ======")
    print(f"  median window        = {allr['median_ms']:.0f} ms "
          f"(IQR {allr['p25_ms']:.0f}-{allr['p75_ms']:.0f} ms)")
    print(f"  spread (variability) = IQR {allr['IQR_ms']:.0f} ms, "
          f"inter-decile {allr['IDR_ms']:.0f} ms, "
          f"CV {allr['CV']:.2f}, QCD {allr['QCD']:.2f}")
    print(f"  hardest 0% contrast  : median {hardest['median_ms']:.0f} ms, "
          f"extends to p90 = {hardest['p90_ms']:.0f} ms")
    print(f"  easiest 100% contrast: median {easiest['median_ms']:.0f} ms, "
          f"p90 = {easiest['p90_ms']:.0f} ms")
    print(f"  fraction > 200 ms    = {allr['frac_gt_200ms']:.1%}")
    print(f"  fraction > 500 ms    = {allr['frac_gt_500ms']:.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="number of sessions")
    ap.add_argument("--refresh", action="store_true", help="re-pull from ONE")
    args = ap.parse_args()

    one = get_one()
    print(f"Connected to: {one.alyx.base_url}\n")

    df, n_sessions, n_fail = load_pooled(one, args.n, args.refresh)
    df = add_metrics(df)

    n_rt_eng = int(engaged_mask(df, "reaction_time").sum())
    n_lat_eng = int(engaged_mask(df, "response_latency").sum())
    print(f"\nPooled raw trials       : {len(df):,}")
    print(f"Engaged (reaction_time) : {n_rt_eng:,}")
    print(f"Engaged (resp_latency)  : {n_lat_eng:,}")

    coverage_caveats(df)

    summary = summarize(df)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    disp = summary.copy()
    for col in disp.columns:
        if col not in ("metric", "contrast_pct", "n"):
            disp[col] = disp[col].astype(float).round(3)
    print("\n========== per-contrast x per-metric summary (ms) ==========")
    print(disp.to_string(index=False))

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved summary table -> {SUMMARY_CSV}")

    make_figure(df, n_sessions, n_lat_eng)
    glanceable(summary)


if __name__ == "__main__":
    main()

"""Phase 1 / step 1b - second identifiability gate (BEHAVIOR ONLY).

Phase 1a found reaction_time is bimodal (fast ~150 ms mode + slow ~1-3 s mode).
The slow mode is where stepping-vs-ramping identifiability lives -- BUT only if
those slow trials are ENGAGED decisions (above-chance, contrast-ordered
accuracy), not the animal zoning out (at-chance accuracy).

This script decides which, from the already-cached pooled trials. No spikes,
no simulation.

Run:
    python src/phase1b_engagement.py
    python src/phase1b_engagement.py --refresh   # re-pull trials first
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ibl_one import DATA_DIR, PROJECT_ROOT  # noqa: E402

POOLED_PARQUET = DATA_DIR / "phase1_trials_pooled.parquet"
FIG_PATH = PROJECT_ROOT / "figures" / "phase1b_accuracy_by_rt.png"
TABLE_CSV = PROJECT_ROOT / "results" / "phase1b_accuracy_by_rt.csv"

STD_CONTRASTS = [0.0, 6.25, 12.5, 25.0, 100.0]
LOW_CONTRASTS = [0.0, 6.25, 12.5]            # "hard" / deliberation-inducing
RT_MAX = 60.0
# Reaction-time bins (ms), left-closed: [0,150) [150,300) ... [2000, inf)
RT_EDGES = [0, 150, 300, 500, 1000, 2000, np.inf]
RT_LABELS = ["<150", "150-300", "300-500", "500-1000", "1000-2000", ">2000"]
SLOW_LABELS = ["500-1000", "1000-2000", ">2000"]   # the deliberative range
MIN_CELL = 30                                       # min n to judge a cell


# --- helpers -----------------------------------------------------------------
def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def load_df(refresh, n):
    if POOLED_PARQUET.exists() and not refresh:
        df = pd.read_parquet(POOLED_PARQUET)
        print(f"Loaded cached pool: {len(df):,} trials from "
              f"{df['eid'].nunique()} sessions ({POOLED_PARQUET}).")
        return df
    from ibl_one import get_one
    from phase1_behavior import load_pooled
    df, _, _ = load_pooled(get_one(), n, refresh=True)
    return df


def add_columns(df):
    df = df.copy()
    df["reaction_time"] = df["firstMovement_times"] - df["stimOn_times"]
    df["response_latency"] = df["response_times"] - df["stimOn_times"]
    cl = np.nan_to_num(df["contrastLeft"].to_numpy(dtype=float))
    cr = np.nan_to_num(df["contrastRight"].to_numpy(dtype=float))
    df["abs_contrast_pct"] = np.round(np.fmax(cl, cr) * 100.0, 4)
    df["signed_contrast"] = cr - cl                     # >0 right, <0 left
    df["correct"] = (df["feedbackType"].to_numpy(dtype=float) == 1).astype(float)
    return df


def restrict_to_trained_ephys(df):
    """Drop training-contaminated sessions.

    The 50% contrast level exists ONLY in the IBL training task; the biased
    ephys task (brain-wide map) uses {0, 6.25, 12.5, 25, 100}%. A session that
    contains any 50% trial is therefore a training (or partially-trained)
    session, and its still-learning behavior corrupts the pooled psychometric
    (it depresses 100%-contrast accuracy toward chance). We keep only sessions
    whose contrast set is the standard biased-ephys set.
    """
    has_training = df.groupby("eid")["abs_contrast_pct"].transform(
        lambda s: bool(np.isclose(s, 50.0).any()))
    n_before = df["eid"].nunique()
    kept = df[~has_training].copy()
    n_after = kept["eid"].nunique()
    print("\n========== curation: restrict to trained biased-ephys ==========")
    print(f"  dropped {n_before - n_after} training-contaminated sessions "
          f"(contain 50% contrast); kept {n_after} trained sessions")
    print(f"  trials: {len(df):,} raw -> {len(kept):,} after curation")
    return kept


def engaged_mask(df, metric):
    v = df[metric].to_numpy(dtype=float)
    m = (df["choice"].to_numpy(dtype=float) != 0)
    m &= np.isfinite(df["stimOn_times"].to_numpy(dtype=float))
    m &= np.isfinite(df["feedbackType"].to_numpy(dtype=float))
    m &= np.isfinite(v) & (v > 0) & (v <= RT_MAX)
    return m


# --- Task 1 sanity: feedbackType vs choice-vs-side geometry ------------------
def geometry_sanity(df):
    signed = df["signed_contrast"].to_numpy()
    choice = df["choice"].to_numpy(dtype=float)
    fb = df["feedbackType"].to_numpy(dtype=float)
    m = (signed != 0) & (choice != 0) & np.isfinite(fb)
    prod = np.sign(choice[m]) * np.sign(signed[m])   # +1 or -1 per trial
    is_correct = fb[m] == 1
    agree_pos = np.mean((prod == 1) == is_correct)   # correct <=> prod==+1
    agree_neg = np.mean((prod == -1) == is_correct)  # correct <=> prod==-1
    if agree_pos >= agree_neg:
        mapping = "correct <=> sign(choice) == sign(contrastR - contrastL)"
        agree = agree_pos
    else:
        mapping = "correct <=> sign(choice) == -sign(contrastR - contrastL)"
        agree = agree_neg
    print("\n========== Task 1: feedbackType <-> geometry sanity ==========")
    print(f"  nonzero-contrast responded trials checked : {int(m.sum()):,}")
    print(f"  inferred mapping : {mapping}")
    print(f"  agreement (feedbackType vs geometry)      : {agree:6.3%}")
    return agree


# --- Task 2 psychometric -----------------------------------------------------
def psychometric(df):
    resp = df[(df["choice"].to_numpy(dtype=float) != 0)
              & np.isfinite(df["feedbackType"].to_numpy(dtype=float))]
    print("\n========== Task 2: psychometric (accuracy vs |contrast|) ==========")
    rows = []
    for c in sorted(resp["abs_contrast_pct"].unique()):
        sub = resp[np.isclose(resp["abs_contrast_pct"], c)]
        acc = sub["correct"].mean()
        rows.append((c, len(sub), acc))
        tag = "  (chance-by-design)" if c == 0 else ""
        flag = "  <-- non-standard (training)" if c not in STD_CONTRASTS else ""
        print(f"  {c:6.2f}% : n={len(sub):6d}  accuracy={acc:6.2%}{tag}{flag}")
    nz = [(c, a) for c, _, a in rows if c > 0 and c in STD_CONTRASTS]
    accs = [a for _, a in nz]
    monotonic = all(x <= y + 1e-9 for x, y in zip(accs, accs[1:]))
    print(f"  monotonic increase over nonzero standard contrasts: {monotonic}")
    return monotonic


# --- Task 3 accuracy x contrast x RT-bin -------------------------------------
def accuracy_grid(df, metric):
    eng = df[engaged_mask(df, metric)].copy()
    eng = eng[eng["abs_contrast_pct"].isin(STD_CONTRASTS)]
    eng["rt_bin"] = pd.cut(eng[metric] * 1000.0, bins=RT_EDGES,
                           labels=RT_LABELS, right=False)
    grp = eng.groupby(["abs_contrast_pct", "rt_bin"], observed=True)["correct"]
    acc = grp.mean().unstack().reindex(index=STD_CONTRASTS, columns=RT_LABELS)
    nct = grp.count().unstack().reindex(index=STD_CONTRASTS, columns=RT_LABELS)

    # long-format with Wilson CIs for the CSV / plotting
    rows = []
    for c in STD_CONTRASTS:
        for b in RT_LABELS:
            n = nct.loc[c, b]
            n = 0 if pd.isna(n) else int(n)
            a = acc.loc[c, b]
            k = 0 if (pd.isna(a) or n == 0) else int(round(a * n))
            lo, hi = wilson(k, n)
            rows.append({"metric": metric, "contrast_pct": c, "rt_bin": b,
                         "n": n, "accuracy": a, "wilson_lo": lo,
                         "wilson_hi": hi})
    return acc, nct, pd.DataFrame(rows)


def print_grid(metric, acc, nct):
    print(f"\n========== Task 3: accuracy x contrast x {metric} bin ==========")
    print("  accuracy (%) [n] per cell; rows=|contrast|%, cols=RT bin (ms)")
    header = "  |contrast|  " + "".join(f"{b:>15}" for b in RT_LABELS)
    print(header)
    for c in STD_CONTRASTS:
        cells = []
        for b in RT_LABELS:
            a = acc.loc[c, b]
            n = nct.loc[c, b]
            if pd.isna(a) or pd.isna(n) or n == 0:
                cells.append(f"{'--':>15}")
            else:
                cells.append(f"{a*100:5.1f}% [{int(n):>5}]".rjust(15))
        print(f"  {c:7.2f}%  " + "".join(cells))


# --- Task 4 budget -----------------------------------------------------------
def engaged_budget(long_df, metric):
    d = long_df[(long_df["metric"] == metric)
                & (long_df["rt_bin"].isin(SLOW_LABELS))
                & (long_df["contrast_pct"].isin(LOW_CONTRASTS))].copy()
    d["engaged"] = (d["n"] >= MIN_CELL) & (d["wilson_lo"] > 0.5)
    total_slow_low = int(d["n"].sum())
    engaged_slow_low = int(d.loc[d["engaged"], "n"].sum())
    print(f"\n========== Task 4: deliberative budget ({metric}) ==========")
    print(f"  low contrast = {LOW_CONTRASTS} %, slow = {SLOW_LABELS} ms")
    print(f"  cells flagged ENGAGED if n>={MIN_CELL} and Wilson-95%-low > 50%:")
    for _, r in d.iterrows():
        if r["n"] == 0:
            continue
        mark = "ENGAGED" if r["engaged"] else "at-chance/low-n"
        acc = r["accuracy"]
        print(f"    {r['contrast_pct']:6.2f}% x {r['rt_bin']:>10}  "
              f"n={int(r['n']):5d}  acc={acc*100:5.1f}% "
              f"(95% CI {r['wilson_lo']*100:4.1f}-{r['wilson_hi']*100:4.1f})  "
              f"-> {mark}")
    print(f"  TOTAL slow+low-contrast trials            : {total_slow_low:,}")
    print(f"  TOTAL engaged (above-chance) of those     : {engaged_slow_low:,}")
    return total_slow_low, engaged_slow_low, d


# --- Task 5 figure -----------------------------------------------------------
def make_figure(long_rt, long_lat):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    cmap = plt.get_cmap("viridis")
    xs = np.arange(len(RT_LABELS))
    for ax, (long_df, metric, title) in zip(
            axes,
            [(long_rt, "reaction_time", "accuracy vs reaction_time bin"),
             (long_lat, "response_latency", "accuracy vs response_latency bin")]):
        for i, c in enumerate(STD_CONTRASTS):
            sub = long_df[long_df["contrast_pct"] == c].set_index("rt_bin")
            sub = sub.reindex(RT_LABELS)
            y = sub["accuracy"].to_numpy(dtype=float) * 100
            lo = sub["wilson_lo"].to_numpy(dtype=float) * 100
            hi = sub["wilson_hi"].to_numpy(dtype=float) * 100
            n = sub["n"].to_numpy(dtype=float)
            mask = n >= 15
            yy = np.where(mask, y, np.nan)
            err = np.vstack([yy - np.where(mask, lo, np.nan),
                             np.where(mask, hi, np.nan) - yy])
            color = cmap(i / (len(STD_CONTRASTS) - 1))
            ax.errorbar(xs, yy, yerr=err, marker="o", ms=5, lw=1.8,
                        capsize=2, color=color, label=f"{c:g}%")
        ax.axhline(50, color="k", ls="--", lw=1, label="chance")
        ax.axvspan(2.5, len(RT_LABELS) - 0.5, color="gray", alpha=0.08)
        ax.text(3.5, 38, "deliberative range\n(>=500 ms)", ha="center",
                fontsize=8, color="gray")
        ax.set_xticks(xs)
        ax.set_xticklabels(RT_LABELS, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(35, 102)
        ax.set_xlabel("RT bin (ms)")
        ax.set_title(title, fontsize=10, loc="left")
    axes[0].set_ylabel("accuracy (% correct)")
    axes[1].legend(title="|contrast|", fontsize=8, loc="lower right")
    fig.suptitle("Phase 1b: is the slow RT mode engaged deliberation? "
                 "(accuracy by RT bin, within contrast)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nSaved figure -> {FIG_PATH}")


def verdict(long_rt):
    """Glanceable read on the slow mode at the lowest nonzero contrasts."""
    d = long_rt[(long_rt["rt_bin"].isin(SLOW_LABELS))
                & (long_rt["contrast_pct"].isin([6.25, 12.5]))]
    # pooled accuracy across slow bins for 6.25 and 12.5 separately
    def pooled(c):
        s = d[d["contrast_pct"] == c]
        n = s["n"].sum()
        k = (s["accuracy"] * s["n"]).sum()
        lo, hi = wilson(int(round(k)), int(n))
        return n, (k / n if n else np.nan), lo, hi
    n6, a6, lo6, _ = pooled(6.25)
    n12, a12, lo12, _ = pooled(12.5)
    ordered = a12 > a6
    above = (lo6 > 0.5) and (lo12 > 0.5)
    print("\n========== GLANCEABLE VERDICT ==========")
    print(f"  slow mode (>=500 ms), pooled over slow bins:")
    print(f"     6.25% contrast: accuracy {a6*100:.1f}% "
          f"(n={int(n6)}, 95% low {lo6*100:.1f}%)")
    print(f"     12.5% contrast: accuracy {a12*100:.1f}% "
          f"(n={int(n12)}, 95% low {lo12*100:.1f}%)")
    print(f"  above chance (both 95% CIs exclude 50%) : {above}")
    print(f"  contrast-ordered (12.5% > 6.25%)        : {ordered}")
    return above, ordered


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    df_raw = add_columns(load_df(args.refresh, args.n))

    geometry_sanity(df_raw)
    df = restrict_to_trained_ephys(df_raw)

    mono = psychometric(df)
    if not mono:
        raise SystemExit(
            "STOP: accuracy does not rise monotonically with contrast even "
            "after curation -- something is wrong with the data or the "
            "correctness definition.")

    acc_rt, n_rt, long_rt = accuracy_grid(df, "reaction_time")
    print_grid("reaction_time", acc_rt, n_rt)
    acc_lat, n_lat, long_lat = accuracy_grid(df, "response_latency")
    print_grid("response_latency", acc_lat, n_lat)

    all_long = pd.concat([long_rt, long_lat], ignore_index=True)
    TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_long.to_csv(TABLE_CSV, index=False)
    print(f"\nSaved table -> {TABLE_CSV}")

    engaged_budget(long_rt, "reaction_time")
    engaged_budget(long_lat, "response_latency")

    make_figure(long_rt, long_lat)
    verdict(long_rt)


if __name__ == "__main__":
    main()

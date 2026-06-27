"""Steinmetz population step-vs-ramp recovery pre-flight (PURE SIM, reuses the §3.5 engine).

IBL's §3.5 verdict (population model dead) was driven by lack of SIMULTANEITY: its 243 MRN
cells were a pseudo-population across 61 sessions (<=53 co-recorded, ~9-25 shared trials), and
recovery needed >=120 simultaneous cells x >=100 shared trials. The Steinmetz feasibility gate
shows GENUINELY SIMULTANEOUS MRN recordings: 11 independent sessions, best 45 hi-FR cells x 35
decorrelated trials (alt 34 x 124), median ~20 hi-FR x 54 trials. This reopens recoverability
as a QUANTITATIVE question, answered here BEFORE building any model.

Reuses phase2_mrn_recovery_preflight.recover_once (shared-latent population HMM, NB emission,
held-out CV-LL, ORACLE per-cell tuning = generous upper bound). No Steinmetz download: the real
per-session yields come from results/steinmetz_coverage_sessions.csv.

Two possible rescuers tested:
  (a) per-cell effect: Steinmetz's true per-cell effect is unknown until the replication decode,
      so coupling is SWEPT from IBL (per-cell AUC ~0.53 / 0.10 SD) up to optimistic (AUC 0.60+);
  (b) AGGREGATION: 11 sessions are independent + simultaneous, so per-session step-vs-ramp
      verdicts can be legitimately combined (majority vote / pooled evidence) -- impossible for
      IBL's pseudo-pool. Does combining 11 weak readouts give a reliable verdict?

  python src/run_steinmetz_preflight.py --stage all          # full
  python src/run_steinmetz_preflight.py --stage all --quick  # smoke
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ibl_one import PROJECT_ROOT  # noqa: E402
from phase1_recovery import DT, load_windows  # noqa: E402
import phase2_mrn_recovery_preflight as E  # noqa: E402  (the validated §3.5 engine)

N_JOBS = E.N_JOBS
GSIGMA = E.MRN_GSIGMA                       # keep IBL coupling-tail shape; sweep scale only
# coupling sweep: gscale -> per-cell AUC ~ {0.53, 0.54, 0.55, 0.57, 0.60, 0.62}
G_SCALE = [0.015, 0.022, 0.032, 0.045, 0.065, 0.090]
FANO = 2.0
COV_SESS = PROJECT_ROOT / "results" / "steinmetz_coverage_sessions.csv"

FIG_PATH = PROJECT_ROOT / "figures" / "steinmetz_population_preflight.png"
PERSESS_CSV = PROJECT_ROOT / "results" / "steinmetz_population_preflight.csv"
AGG_CSV = PROJECT_ROOT / "results" / "steinmetz_population_aggregation.csv"
CKPT_RAW = PROJECT_ROOT / "results" / "_steinpop_raw.csv"
CKPT_CAL = PROJECT_ROOT / "results" / "_steinpop_calib.csv"


# ---- calibration: gscale -> per-cell AUC + effect SD + pop decode AUC -------------
def calibrate(gscale, windows, N=243, ntr=1500, seed=0):
    rng = np.random.default_rng(seed)
    b, g = E.draw_cells(rng, N, gscale, GSIGMA)
    d = rng.choice([-1.0, 1.0], ntr)
    Tb = np.maximum(3, np.round(rng.choice(windows, ntr) / DT).astype(int))
    X = np.empty((ntr, N))
    for j in range(ntr):
        tc = E.sim_trial("step" if rng.random() < 0.5 else "ramp", rng, int(Tb[j]),
                         d[j], b, g, FANO)
        X[j] = tc.sum(1) / (tc.shape[1] * DT)
    y = (d > 0).astype(int)
    auc = np.array([roc_auc_score(y, X[:, i]) for i in range(N)])
    typ = float(np.mean(np.maximum(auc, 1 - auc)))
    eff = float(np.mean([abs(X[y == 1, i].mean() - X[y == 0, i].mean()) /
                         (X[:, i].std() + 1e-9) for i in range(N)]))
    k = int(0.7 * ntr)
    sc = StandardScaler().fit(X[:k])
    clf = LogisticRegression(C=0.05, max_iter=400).fit(sc.transform(X[:k]), y[:k])
    pop = float(roc_auc_score(y[k:], clf.predict_proba(sc.transform(X[k:]))[:, 1]))
    return dict(gscale=gscale, percell_auc=typ, percell_eff_sd=eff, pop_auc=pop)


# ---- session configs ----------------------------------------------------------------
def load_mrn_sessions():
    """The 11 genuinely-simultaneous Steinmetz MRN sessions (hi-FR cells x decorrelated)."""
    s = pd.read_csv(COV_SESS)
    m = s[s.MRN_cells > 0].copy()
    m["label"] = m.mouse.astype(str) + "_" + m.date.astype(str)
    m = m.sort_values("MRN_hiFR", ascending=False)
    return [(r.label, int(r.MRN_hiFR), int(r.n_decorr)) for _, r in m.iterrows()]


# synthetic "named scales" the mission calls out (median-cell / median-hiFR sessions)
SYNTH_SCALES = [("median_72x46", 72, 46), ("median_hiFR_20x46", 20, 46)]


def run_block(configs, gscales, windows, R, tag, seed0):
    """For each (label, cells, trials) x gscale x R reps x {step,ramp}: recover_once.
    Stores every draw (correct + d_cv) -> raw material for per-session AND aggregation."""
    jobs, sid = [], 0
    for (label, nc, nt) in configs:
        nc = max(2, int(nc))
        for gs in gscales:
            for tg in ("step", "ramp"):
                for _ in range(R):
                    jobs.append((label, nc, nt, gs, tg, seed0 + sid)); sid += 1
    print(f"  [{tag}] {len(jobs)} recoveries ({len(configs)} cfg x {len(gscales)} gscale x 2 x R={R})")
    t = time.time()
    res = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(_one)(label, nc, nt, gs, tg, sd, windows)
        for (label, nc, nt, gs, tg, sd) in jobs)
    print(f"  [{tag}] done in {time.time()-t:.0f}s")
    return pd.DataFrame([r for r in res if r is not None])


def _one(label, nc, nt, gscale, tg, seed, windows):
    r = E.recover_once(tg, nc, nt, FANO, gscale, GSIGMA, seed, windows)
    if r is None:
        return None
    r["label"] = label
    return r


# ---- aggregation across independent simultaneous sessions ---------------------------
def aggregate(raw, sess_labels, gscale, n_ens, rng, topk=None):
    """Bootstrap the legitimate combine: one ground truth (does MRN step or ramp?), each of the
    independent simultaneous sessions gives a noisy verdict; combine by MAJORITY VOTE and by
    POOLED EVIDENCE sign(sum d_cv). Draws resampled from the stored per-session reps."""
    labels = sess_labels[:topk] if topk else sess_labels
    pools = {}
    for lab in labels:
        sub = raw[(raw.label == lab) & (np.isclose(raw.gscale, gscale))]
        pools[lab] = {tg: sub[sub.true == tg][["correct", "d_cv"]].to_numpy()
                      for tg in ("step", "ramp")}
    pools = {k: v for k, v in pools.items() if len(v["step"]) and len(v["ramp"])}
    if not pools:
        return dict(maj=np.nan, pool=np.nan, maj_step=np.nan, maj_ramp=np.nan)
    labs = list(pools)
    ok = dict(maj=0, pool=0)
    per = {"step": dict(maj=0, n=0), "ramp": dict(maj=0, n=0)}
    for i in range(n_ens):
        tg = "step" if i % 2 == 0 else "ramp"          # balanced ground truth
        nstep = 0; sdcv = 0.0
        for lab in labs:
            arr = pools[lab][tg]
            c, dcv = arr[rng.integers(len(arr))]
            pred = tg if c == 1 else ("ramp" if tg == "step" else "step")
            nstep += (pred == "step"); sdcv += dcv
        maj = "step" if nstep > len(labs) / 2 else ("ramp" if nstep < len(labs) / 2
                                                    else ("step" if sdcv > 0 else "ramp"))
        ok["maj"] += (maj == tg); ok["pool"] += (("step" if sdcv > 0 else "ramp") == tg)
        per[tg]["maj"] += (maj == tg); per[tg]["n"] += 1
    return dict(maj=ok["maj"] / n_ens, pool=ok["pool"] / n_ens,
                maj_step=per["step"]["maj"] / max(per["step"]["n"], 1),
                maj_ramp=per["ramp"]["maj"] / max(per["ramp"]["n"], 1))


# ---- summarise ----------------------------------------------------------------------
def per_session_summary(raw, cal):
    rows = []
    a2g = dict(zip(cal.gscale.round(4), cal.percell_auc))
    for (label, nc, nt, gs), gg in raw.groupby(["label", "N_cells", "N_trials", "gscale"]):
        acc = gg.correct.mean(); n = len(gg)
        rows.append(dict(label=label, N_cells=int(nc), N_trials=int(nt), gscale=gs,
                         percell_auc=a2g.get(round(gs, 4), np.nan),
                         recovery=acc, se=np.sqrt(acc * (1 - acc) / max(n, 1)), n=n,
                         rec_step=gg[gg.true == "step"].correct.mean(),
                         rec_ramp=gg[gg.true == "ramp"].correct.mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["all", "report"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if args.fresh:
        for p in (CKPT_RAW, CKPT_CAL):
            p.unlink(missing_ok=True)

    windows = load_windows()
    sessions = load_mrn_sessions()
    print(f"Steinmetz MRN sessions: {len(sessions)} (hi-FR x decorr) "
          f"{[(l.split('_')[0], c, t) for l, c, t in sessions]}\n")

    gscales = G_SCALE if not args.quick else [0.015, 0.065]
    R = 4 if args.quick else 50
    n_ens = 50 if args.quick else 600

    # ---- calibration ----
    if CKPT_CAL.exists() and not args.quick:
        cal = pd.read_csv(CKPT_CAL)
        print("loaded calibration checkpoint")
    else:
        print("== calibration: gscale -> per-cell AUC / effect / pop AUC ==")
        cal = pd.DataFrame([calibrate(gs, windows) for gs in gscales])
        if not args.quick:
            cal.to_csv(CKPT_CAL, index=False)
    for _, r in cal.iterrows():
        print(f"  gscale {r.gscale:.3f}: per-cell AUC {r.percell_auc:.3f}, "
              f"effect {r.percell_eff_sd:.3f} SD, pop AUC {r.pop_auc:.3f}")

    # ---- easy-regime sanity (reuse engine; strong coupling, large N) ----
    print("\n== EASY-regime sanity (strong coupling, large N) ==")
    easy = run_block([("easy", 80, 120)], [E.EASY_GSCALE], windows,
                     4 if args.quick else 40, "easy", 10_000)
    easy_rec = easy.correct.mean()
    print(f"  EASY recovery = {easy_rec:.1%} (step {easy[easy.true=='step'].correct.mean():.0%}"
          f" / ramp {easy[easy.true=='ramp'].correct.mean():.0%}) -> "
          f"{'RECOVERS' if easy_rec >= 0.85 else 'METHOD SUSPECT'}")
    if easy_rec < 0.85:
        print("  !! easy regime failed -> STOP, do not trust hard regime"); return

    # ---- per-session recovery sweep (11 real sessions + synthetic named scales) ----
    print("\n== per-session recovery: Steinmetz scales x coupling sweep ==")
    configs = [(l, c, t) for l, c, t in sessions] + SYNTH_SCALES
    if CKPT_RAW.exists() and not args.quick:
        raw = pd.read_csv(CKPT_RAW); print(f"  loaded raw checkpoint ({len(raw)} draws)")
    else:
        raw = run_block(configs, gscales, windows, R, "persess", 20_000)
        if not args.quick:
            raw.to_csv(CKPT_RAW, index=False)
    ps = per_session_summary(raw, cal)
    ps.to_csv(PERSESS_CSV, index=False)

    # ---- aggregation across the 11 independent simultaneous sessions ----
    print("\n== across-session aggregation (11 independent simultaneous MRN sessions) ==")
    sess_labels = [l for l, _, _ in sessions]
    rng = np.random.default_rng(7)
    agg_rows = []
    for gs in gscales:
        pa = float(cal[np.isclose(cal.gscale, gs)].percell_auc.iloc[0])
        ag = aggregate(raw, sess_labels, gs, n_ens, rng)
        # best single-session recovery at this coupling (the strongest individual session)
        best = ps[(np.isclose(ps.gscale, gs))].recovery.max()
        # recovery vs #sessions (best-K), for the curve
        ksrec = {k: aggregate(raw, sess_labels, gs, n_ens, rng, topk=k)["maj"]
                 for k in (1, 3, 5, 7, 11)}
        agg_rows.append(dict(gscale=gs, percell_auc=pa, agg_majority=ag["maj"],
                             agg_pooled=ag["pool"], agg_maj_step=ag["maj_step"],
                             agg_maj_ramp=ag["maj_ramp"], best_single=best,
                             **{f"k{k}": ksrec[k] for k in ksrec}))
        print(f"  per-cell AUC {pa:.3f} (gscale {gs:.3f}): best-single {best:.1%} | "
              f"11-sess majority {ag['maj']:.1%} (step {ag['maj_step']:.0%}/ramp "
              f"{ag['maj_ramp']:.0%}) | pooled {ag['pool']:.1%}")
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(AGG_CSV, index=False)

    make_figure(ps, agg, cal, easy_rec, sessions)
    verdict(ps, agg)


def make_figure(ps, agg, cal, easy_rec, sessions):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    chance, usable = 50.0, 80.0

    # A: per-session recovery at the 3 named scales vs coupling
    a = ax[0, 0]
    named = {"Tatum_2017-12-07": "best 45x35", "Muller_2017-01-07": "alt 34x124",
             "median_72x46": "median 72x46", "median_hiFR_20x46": "med hiFR 20x46"}
    for lab, txt in named.items():
        g = ps[ps.label == lab].sort_values("percell_auc")
        if len(g):
            a.plot(g.percell_auc, g.recovery * 100, "o-", label=txt)
    a.axhline(chance, color="k", ls="--", lw=1); a.axhline(usable, color="gray", ls=":", lw=1)
    a.axvline(0.529, color="navy", lw=1, ls="--"); a.text(0.529, 52, "IBL", color="navy", fontsize=7)
    a.set_xlabel("per-cell choice AUC (coupling)"); a.set_ylabel("per-session recovery (%)")
    a.set_title("Per-session recovery vs per-cell effect (Steinmetz scales)", fontsize=10, loc="left")
    a.legend(fontsize=8); a.set_ylim(40, 100)

    # B: aggregation -- 11-session majority & pooled vs best single, vs coupling
    a = ax[0, 1]
    a.plot(agg.percell_auc, agg.best_single * 100, "o-", color="#c44", label="best single session")
    a.plot(agg.percell_auc, agg.agg_majority * 100, "s-", color="#1a5", label="11-session majority")
    a.plot(agg.percell_auc, agg.agg_pooled * 100, "^-", color="#27a", label="11-session pooled evidence")
    a.axhline(chance, color="k", ls="--", lw=1); a.axhline(usable, color="gray", ls=":", lw=1)
    a.set_xlabel("per-cell choice AUC (coupling)"); a.set_ylabel("recovery (%)")
    a.set_title("Aggregation rescue: 11 independent sessions vs best single", fontsize=10, loc="left")
    a.legend(fontsize=8); a.set_ylim(40, 102)

    # C: recovery vs #sessions aggregated, at IBL vs optimistic coupling
    a = ax[1, 0]
    ks = [1, 3, 5, 7, 11]
    for gs, txt, col in [(0.015, "IBL AUC~0.53", "#c44"), (0.032, "AUC~0.55", "#e80"),
                         (0.065, "AUC~0.60", "#1a5")]:
        row = agg[np.isclose(agg.gscale, gs)]
        if len(row):
            a.plot(ks, [row.iloc[0][f"k{k}"] * 100 for k in ks], "o-", color=col, label=txt)
    a.axhline(chance, color="k", ls="--", lw=1); a.axhline(usable, color="gray", ls=":", lw=1)
    a.set_xlabel("# independent sessions aggregated (majority)"); a.set_ylabel("recovery (%)")
    a.set_title("Meta-analytic gain vs # sessions", fontsize=10, loc="left")
    a.legend(fontsize=8); a.set_ylim(40, 102)

    # D: all 11 sessions -- recovery at IBL coupling vs hi-FR x trials (scatter)
    a = ax[1, 1]
    gs0 = 0.015
    pts = []
    for lab, c, t in sessions:
        r = ps[(ps.label == lab) & np.isclose(ps.gscale, gs0)]
        if len(r):
            pts.append((c, t, r.recovery.iloc[0]))
    if pts:
        cc, tt, rr = zip(*pts)
        scil = a.scatter(cc, tt, c=np.array(rr) * 100, s=90, cmap="RdYlGn", vmin=50, vmax=85,
                         edgecolor="k")
        for c, t, r in pts:
            a.annotate(f"{r*100:.0f}", (c, t), fontsize=7, ha="center", va="center")
        fig.colorbar(scil, ax=a, fraction=0.046, pad=0.02).set_label("recovery (%)")
    a.set_xlabel("hi-FR MRN cells (session)"); a.set_ylabel("decorrelated trials (session)")
    a.set_title("11 MRN sessions at IBL coupling (per-cell AUC~0.53)", fontsize=10, loc="left")

    fig.suptitle("Steinmetz population step-vs-ramp recovery pre-flight "
                 "(genuinely simultaneous MRN; NB Fano 2; oracle tuning)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nSaved figure -> {FIG_PATH}")


def verdict(ps, agg):
    print("\n========== GLANCEABLE VERDICT (Steinmetz population arm) ==========")
    # per-session at the 3 named scales, IBL coupling vs optimistic
    for gs, tag in [(0.015, "IBL (AUC~0.53)"), (0.065, "optimistic (AUC~0.60)")]:
        print(f"  per-session recovery @ {tag}:")
        for lab, txt in [("Tatum_2017-12-07", "best 45x35"), ("Muller_2017-01-07", "alt 34x124"),
                         ("median_72x46", "median 72x46")]:
            r = ps[(ps.label == lab) & np.isclose(ps.gscale, gs)]
            if len(r):
                rr = r.iloc[0]
                print(f"    {txt:16s}: {rr.recovery:.1%} +/- {rr.se:.1%}")
    # threshold per-cell AUC for usable per-session at best scale
    best = ps[ps.label == "Tatum_2017-12-07"].sort_values("percell_auc")
    thr = best[best.recovery >= 0.80].percell_auc.min() if (best.recovery >= 0.80).any() else None
    print(f"  per-cell-AUC threshold for >=80% per-session (best 45x35 session): "
          f"{thr if thr is not None else '>0.62 (not reached in sweep)'}")
    # aggregation (require BALANCE: high overall but step~0/ramp~100 = bias, not recovery)
    print("  11-session AGGREGATION (majority vote; balance guards against bias-amplification):")
    for _, r in agg.iterrows():
        bal = min(r.agg_maj_step, r.agg_maj_ramp)
        usable = (r.agg_majority >= 0.80) and (bal >= 0.70)
        flag = "USABLE" if usable else ("biased" if r.agg_majority >= 0.70 and bal < 0.60
                                        else "weak")
        print(f"    per-cell AUC {r.percell_auc:.3f}: majority {r.agg_majority:.1%} "
              f"(step {r.agg_maj_step:.0%}/ramp {r.agg_maj_ramp:.0%}), pooled {r.agg_pooled:.1%}"
              f"  (best single {r.best_single:.1%})  [{flag}]")
    ibl = agg[np.isclose(agg.gscale, 0.015)].iloc[0]
    opt = agg[np.isclose(agg.gscale, 0.065)].iloc[0]

    def usable_balanced(r):
        return (r.agg_majority >= 0.80) and (min(r.agg_maj_step, r.agg_maj_ramp) >= 0.70)
    print("\n  ---- READ ----")
    agg_usable_ibl = usable_balanced(ibl)
    agg_usable_opt = usable_balanced(opt)
    if agg_usable_ibl:
        print("  PASS at IBL-level per-cell effect via AGGREGATION: combining 11 independent")
        print("  simultaneous sessions yields a usable population step-vs-ramp verdict where no")
        print("  single session does -> the population arm is LIVE on Steinmetz's real data.")
    elif agg_usable_opt:
        print("  CONDITIONAL PASS: aggregation reaches usable only if Steinmetz's per-cell effect")
        print("  exceeds IBL's (per-cell AUC >~0.57-0.60) -> gated on the replication decode.")
    else:
        print("  FAIL: even aggregating 11 simultaneous sessions at optimistic per-cell effects")
        print("  stays below usable -> even the best available simultaneous midbrain dataset can't")
        print("  support single-trial step-vs-ramp; §3.5 closes harder with a quantified bar.")


if __name__ == "__main__":
    main()

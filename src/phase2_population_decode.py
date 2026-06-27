"""Tier-1 #1 - population-decode rebuttal + gate on a population decision model.

"Absence of a strong single-cell signal != absence of a distributed population signal."
This asks whether a CROSS-VALIDATED MULTIVARIATE decoder pulls MORE stimulus- and
movement-INDEPENDENT choice out of the joint population than the per-cell analysis did.

Reuses the cached full-coverage features (results/phase2_sel_features_full.csv); NO new
spike loading, no downloads.

Method (per region {MRN, SCm, IRN, GRN, SNr}):
  - DECORRELATED trials only (0%-contrast + error), where choice is separable from stimulus.
  - PSEUDO-POPULATION decode (cells are not simultaneous): pseudo-trials are built by
    drawing, per cell, a deliberative-window rate from a trial of the target choice. Per
    cell, trials are split into train/test FIRST (no leakage). Movement (wheel + DLC
    paw/nose) and stimulus/prior (signed contrast, block pL) are regressed out WITHIN THE
    CV FOLD (confound betas fit on train, applied to test) -- this matters: a regression
    intercept makes residuals mean-zero, so global residualisation + a train/test split
    induces a spurious train<->test ANTI-correlation that flips the decoder. Within-fold
    residualisation avoids it. Features standardised on train; L2 logistic; ROC-AUC on
    held-out test pseudo-trials; permutation null shuffles choice within each cell.
    (Tests whether pooling weak per-cell signals via an optimal readout beats the
    per-cell ceiling; cannot exploit non-recorded simultaneous noise correlations.)

  python src/phase2_population_decode.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ibl_one import PROJECT_ROOT  # noqa: E402
from phase2_census import CORE_REGIONS  # noqa: E402

FEATURES_CSV = PROJECT_ROOT / "results" / "phase2_sel_features_full.csv"
DECODE_CSV = PROJECT_ROOT / "results" / "phase2_population_decode.csv"
CURVE_CSV = PROJECT_ROOT / "results" / "phase2_population_decode_curve.csv"
FIG_PATH = PROJECT_ROOT / "figures" / "phase2_population_decode.png"

MIN_SIDE = 4                # min decorrelated trials per choice side to include a cell
N_TR, N_TE = 300, 120       # pseudo-trials per class (train / test)
R_OBS = 40                  # reps averaged into the observed decode estimate
N_PERM, R_SUB = 80, 12      # null = N_PERM means, each averaged over R_SUB shuffled reps
RIDGE = 1.0                 # confound-regression ridge (stabilise tiny per-cell fits)
LOGIT_C = 0.05              # strong L2 -> robust signal-pooling, less overfit at high K
CURVE_REPS = 30             # cell-subsample repeats per K (smoother decode-vs-N curve)
RATE = "rate_delib"


def prep(f, errors_only=False):
    """Per eligible cell: raw deliberative rate, confound matrix [wheel, body, signed,
    pL], and chose_side, on decorrelated trials. errors_only=True drops 0%-contrast
    trials (the strict control: error trials remove BOTH the stimulus and the bulk of
    the block-prior confound, since a stimulus is present)."""
    f = f.copy()
    f["chose"] = -np.sign(f["choice"])
    f["stim"] = np.sign(f["signed"])
    f["pL"] = np.nan_to_num(f["pL"].to_numpy(float), nan=0.5)
    paw = f["paw_speed"].to_numpy(float); nose = f["nose_speed"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        body = np.nanmean(np.c_[paw, nose], axis=1)      # nan when no DLC -> 0
    f["body"] = np.nan_to_num(body, nan=0.0)
    f["is0"] = f["absc"] == 0
    f["err"] = (f["absc"] > 0) & (f["chose"] != f["stim"])
    dec = f[f["err"]] if errors_only else f[f["is0"] | f["err"]]
    data, region = {}, {}
    for cell, g in dec.groupby("cell"):
        chose = g["chose"].to_numpy()
        if (chose < 0).sum() < MIN_SIDE or (chose > 0).sum() < MIN_SIDE:
            continue
        C = np.c_[g["wheel_speed"].to_numpy(float), g["body"].to_numpy(float),
                  g["signed"].to_numpy(float), g["pL"].to_numpy(float)]
        C = np.nan_to_num(C, nan=0.0)
        data[cell] = (g[RATE].to_numpy(float), C, chose)
        region[cell] = g["region"].iloc[0]
    return data, region


def _ridge_resid(rate, C, idx_fit, idx_apply):
    """Fit rate ~ [1,C] (ridge) on idx_fit; return residuals on idx_apply."""
    Mf = np.c_[np.ones(len(idx_fit)), C[idx_fit]]
    A = Mf.T @ Mf + RIDGE * np.eye(Mf.shape[1])
    beta = np.linalg.solve(A, Mf.T @ rate[idx_fit])
    Ma = np.c_[np.ones(len(idx_apply)), C[idx_apply]]
    return rate[idx_apply] - Ma @ beta


def _split_pools(data, cells, rng, perm):
    """Per cell: split decorrelated trials by choice into train/test, residualise
    confounds within-fold, return train/test residual pools keyed by choice."""
    trp, tep = {}, {}
    for cell in cells:
        rate, C, chose = data[cell]
        ch = chose.copy()
        if perm:
            rng.shuffle(ch)
        iL, iR = np.where(ch < 0)[0], np.where(ch > 0)[0]
        rng.shuffle(iL); rng.shuffle(iR)
        kL, kR = max(1, int(round(0.7 * len(iL)))), max(1, int(round(0.7 * len(iR))))
        tri = np.r_[iL[:kL], iR[:kR]]
        tei = np.r_[iL[kL:], iR[kR:]]
        if iL[kL:].size == 0 or iR[kR:].size == 0:        # guarantee both test classes
            tei = np.r_[iL[-1:], iR[-1:]]
        rtr = _ridge_resid(rate, C, tri, tri)
        rte = _ridge_resid(rate, C, tri, tei)
        ctr, cte = ch[tri], ch[tei]
        trp[cell] = {-1: rtr[ctr < 0], 1: rtr[ctr > 0]}
        tep[cell] = {-1: rte[cte < 0], 1: rte[cte > 0]}
    return trp, tep


def _pseudo(pool, cells, rng, n):
    K = len(cells)
    X = np.empty((2 * n, K))
    for j, cell in enumerate(cells):
        X[:n, j] = rng.choice(pool[cell][-1], n)
        X[n:, j] = rng.choice(pool[cell][1], n)
    return X, np.r_[np.zeros(n, int), np.ones(n, int)]


def decode(data, cells, rng, perm=False, reps=R_OBS):
    aucs = []
    for _ in range(reps):
        trp, tep = _split_pools(data, cells, rng, perm)
        Xtr, ytr = _pseudo(trp, cells, rng, N_TR)
        Xte, yte = _pseudo(tep, cells, rng, N_TE)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=LOGIT_C, max_iter=400)
        clf.fit(sc.transform(Xtr), ytr)
        aucs.append(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))
    return np.array(aucs)


def per_cell_auc(data, cells, rng):
    """Univariate ceiling: cross-validated single-cell choice AUC (within-fold residual,
    averaged over splits)."""
    out = []
    for cell in cells:
        rate, C, chose = data[cell]
        accs = []
        for _ in range(10):
            iL, iR = np.where(chose < 0)[0], np.where(chose > 0)[0]
            rng.shuffle(iL); rng.shuffle(iR)
            kL, kR = max(1, int(round(0.7 * len(iL)))), max(1, int(round(0.7 * len(iR))))
            tei = np.r_[iL[kL:], iR[kR:]]
            if iL[kL:].size == 0 or iR[kR:].size == 0:
                tei = np.r_[iL[-1:], iR[-1:]]
            tri = np.r_[iL[:kL], iR[:kR]]
            rte = _ridge_resid(rate, C, tri, tei)
            yte = (chose[tei] > 0).astype(int)
            if yte.min() != yte.max():
                accs.append(roc_auc_score(yte, rte))
        if accs:
            out.append(np.mean(accs))
    return np.array(out)


def region_stats(data, cells, rng):
    obs_mean = decode(data, cells, rng, perm=False, reps=R_OBS)
    om = obs_mean.mean()
    bmean = [rng.choice(obs_mean, len(obs_mean)).mean() for _ in range(2000)]
    lo, hi = np.percentile(bmean, [2.5, 97.5])
    nm = np.array([decode(data, cells, rng, perm=True, reps=R_SUB).mean()
                   for _ in range(N_PERM)])
    p = (np.sum(nm >= om) + 1) / (len(nm) + 1)
    return dict(auc=om, ci_lo=lo, ci_hi=hi, null=nm.mean(),
                null_hi=np.percentile(nm, 97.5), p=p)


def main():
    f = pd.read_csv(FEATURES_CSV)
    data_b, reg_b = prep(f, errors_only=False)        # 0% + error
    data_e, reg_e = prep(f, errors_only=True)         # error-only (strict: drops prior)
    br_b = {r: [c for c in data_b if reg_b[c] == r] for r in CORE_REGIONS}
    br_e = {r: [c for c in data_e if reg_e[c] == r] for r in CORE_REGIONS}
    rng = np.random.default_rng(0)
    print(f"Eligible cells (>= {MIN_SIDE}/side): 0%+error {len(data_b)}, "
          f"error-only {len(data_e)}")
    print("  0%+error per region:", {r: len(v) for r, v in br_b.items()})
    print("  error-only per region:", {r: len(v) for r, v in br_e.items()})

    rows, curve = [], []
    for r in CORE_REGIONS:
        cb, ce = br_b[r], br_e[r]
        if len(cb) < 5:
            print(f"  {r}: too few cells ({len(cb)}) - skipped"); continue
        full = region_stats(data_b, cb, rng)
        pc = per_cell_auc(data_b, cb, rng)
        err = region_stats(data_e, ce, rng) if len(ce) >= 5 else None
        rows.append(dict(region=r, n_cells=len(cb), n_cells_err=len(ce),
                         decode_auc=full["auc"], ci_lo=full["ci_lo"], ci_hi=full["ci_hi"],
                         null_auc=full["null"], null_hi=full["null_hi"], perm_p=full["p"],
                         err_auc=err["auc"] if err else np.nan,
                         err_ci_lo=err["ci_lo"] if err else np.nan,
                         err_ci_hi=err["ci_hi"] if err else np.nan,
                         err_null=err["null"] if err else np.nan,
                         err_p=err["p"] if err else np.nan,
                         percell_mean_auc=pc.mean(), percell_p95_auc=np.percentile(pc, 95)))
        es = (f"err-only {err['auc']:.3f} [{err['ci_lo']:.3f},{err['ci_hi']:.3f}] "
              f"p={err['p']:.3g} (n={len(ce)})") if err else "err-only n/a"
        print(f"  {r}: 0%+err {full['auc']:.3f} [{full['ci_lo']:.3f},{full['ci_hi']:.3f}] "
              f"(null {full['null']:.3f}, p={full['p']:.3g}) | {es} | "
              f"per-cell mean {pc.mean():.3f} p95 {np.percentile(pc,95):.3f}")
        for K in [k for k in (1, 2, 5, 10, 20, 50, 100, 200, len(cb)) if k <= len(cb)]:
            accs = [decode(data_b, list(rng.choice(cb, K, replace=False)), rng, reps=1)[0]
                    for _ in range(CURVE_REPS)]
            curve.append(dict(region=r, K=K, auc=float(np.mean(accs)),
                              auc_sd=float(np.std(accs))))

    rt = pd.DataFrame(rows); cu = pd.DataFrame(curve)
    DECODE_CSV.parent.mkdir(parents=True, exist_ok=True)
    rt.to_csv(DECODE_CSV, index=False); cu.to_csv(CURVE_CSV, index=False)
    make_figure(rt, cu)

    print("\n========== GLANCEABLE VERDICT ==========")
    sig_full = list(rt[rt["perm_p"] < 0.05]["region"])
    sig_err = list(rt[rt["err_p"] < 0.05]["region"])
    gain = rt["decode_auc"].mean() - rt["percell_mean_auc"].mean()
    print(f"  0%+error decode above chance: {sig_full}  "
          f"(mean AUC {rt['decode_auc'].mean():.3f} vs per-cell mean "
          f"{rt['percell_mean_auc'].mean():.3f}, gain {gain:+.3f})")
    print(f"  STRICT error-only (prior removed) above chance: {sig_err}")
    print("  decode-vs-N accumulates with K (distributed, not outlier-driven): "
          "see curve CSV")
    # robust = significant in BOTH modes AND error-only AUC clearly above chance
    robust = list(rt[(rt["perm_p"] < 0.05) & (rt["err_p"] < 0.05)
                     & (rt["err_auc"] > 0.60)]["region"])
    print(f"  ROBUST (sig in BOTH 0%+error AND error-only, err-AUC>0.6): {robust}")
    if len(robust) >= 2:
        state = "STRONG distributed (>=2 regions robust)"
    elif len(robust) == 1:
        state = (f"REAL but region-specific - distributed & per-cell-cryptic in "
                 f"{robust[0]}; apparent signal elsewhere was prior-driven")
    else:
        state = "WEAK (no region robust to the strict prior control)"
    print(f"  -> distributed stimulus/movement/prior-independent choice signal is: {state}")


def make_figure(rt, cu):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    x = np.arange(len(rt)); w = 0.27
    ax.bar(x - w, rt["decode_auc"], w, yerr=[rt["decode_auc"] - rt["ci_lo"],
           rt["ci_hi"] - rt["decode_auc"]], capsize=3, color="#27a",
           label="decode (0%+error)")
    ax.bar(x, rt["err_auc"], w, yerr=[rt["err_auc"] - rt["err_ci_lo"],
           rt["err_ci_hi"] - rt["err_auc"]], capsize=3, color="#1a5",
           label="decode (error-only, prior-removed)")
    ax.bar(x + w, rt["percell_mean_auc"], w, color="#bbb", label="per-cell mean (CV)")
    ax.plot(x - w, rt["null_hi"], "r_", ms=16, mew=2, label="perm null (97.5%)")
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(rt["region"]); ax.set_ylim(0.40, None)
    ax.set_ylabel("choice decode AUC (held-out)")
    ax.set_title("Stimulus/movement-independent choice decode per region",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8)
    ax = axes[1]
    for r, g in cu.groupby("region"):
        ax.plot(g["K"], g["auc"], "o-", label=r)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_xlabel("# cells pooled (K)")
    ax.set_ylabel("decode AUC")
    ax.set_title("Decode vs pooled cell count", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    fig.suptitle("Phase 2 population decode: distributed stimulus/movement-independent "
                 "choice signal", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nSaved figure -> {FIG_PATH}")


if __name__ == "__main__":
    main()

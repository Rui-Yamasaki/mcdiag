"""Phase 2 / step 3 - the fork-deciding confound test: genuine CHOICE vs STIMULUS.

Step 2 found contrast is the dominant confound (at <=25% contrast, choice ~ stimulus).
The clean separation uses trials where choice is DECORRELATED from stimulus:
  - ERROR trials (choice != stimulus side) at low-to-moderate contrast  -> the decisive
    dissociation: on errors, does the cell's rate follow the CHOICE made or the STIMULUS
    shown?
  - 0%-CONTRAST trials (no stimulus side; choice driven by prior/internal) -> does rate
    predict choice, controlling block prior?

No new spike loading: reuses results/phase2_sel_features.csv (per-cell x per-trial rate
in the deliberative window + wheel, extracted from TRIMMED spikes in step 2). All
deliberative trials (<=25% contrast, engaged) are present, so 0% and error trials are
already there. NO model fitting. No git.

Mapping (Phase 1b geometry, 100% verified): correct <=> sign(choice) == -sign(signed),
signed = contrastRight - contrastLeft. So chose_side = -sign(choice) (+1 = chose right),
stim_side = sign(signed).

  python src/phase2_choice_vs_stim.py            # analyze + figure + verdict
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import false_discovery_control  # noqa: E402

from ibl_one import PROJECT_ROOT  # noqa: E402
from phase2_census import CORE_REGIONS  # noqa: E402

FEATURES_CSV = PROJECT_ROOT / "results" / "phase2_sel_features.csv"
CELLS_CSV = PROJECT_ROOT / "results" / "phase2_choicestim_cells.csv"
REGION_CSV = PROJECT_ROOT / "results" / "phase2_choicestim_region.csv"
FIG_PATH = PROJECT_ROOT / "figures" / "phase2_choice_vs_stim.png"

N_PERM = 1000
FDR_Q = 0.05
MIN_SIDE = 3                 # min trials per choice on a decorrelated subset
MIN_DECORR = 6               # min (error + 0%) trials to identify a choice coefficient
RATE = "rate_delib"


def _lstsq_choice_beta(rate, X):
    beta, *_ = np.linalg.lstsq(X, rate, rcond=None)
    return beta[1]            # choice (chose_side) coefficient


def joint_choice_test(rate, chose, stim, pL, rng):
    """Choice coefficient controlling stimulus side + prior, with a permutation null
    that shuffles choice WITHIN stimulus-side strata (preserves stimulus structure)."""
    n = len(rate)
    X = np.column_stack([np.ones(n), chose, stim, pL])
    obs = _lstsq_choice_beta(rate, X)
    strata = [np.where(stim == s)[0] for s in (-1, 0, 1)]
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        ch = chose.copy()
        for idx in strata:
            if idx.size > 1:
                ch[idx] = rng.permutation(ch[idx])
        null[k] = _lstsq_choice_beta(rate, np.column_stack([np.ones(n), ch, stim, pL]))
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)
    return float(obs), float(p)


def analyze():
    f = pd.read_csv(FEATURES_CSV)
    f["chose"] = -np.sign(f["choice"])               # +1 = chose right
    f["stim"] = np.sign(f["signed"])                 # +1 = stimulus right
    f["pL"] = np.nan_to_num(f["pL"].to_numpy(float), nan=0.5)
    f["is0"] = f["absc"] == 0
    f["err"] = (f["absc"] > 0) & (f["chose"] != f["stim"])
    f["cor"] = (f["absc"] > 0) & (f["chose"] == f["stim"])

    rng = np.random.default_rng(0)
    rec = []
    for cell, g in f.groupby("cell"):
        r = g[RATE].to_numpy(float)
        chose, stim, pL = g["chose"].to_numpy(), g["stim"].to_numpy(), g["pL"].to_numpy()
        n_dec = int((g["err"] | g["is0"]).sum())
        # (2,4) joint regression: choice coding controlling stimulus + prior
        if n_dec >= MIN_DECORR and len(np.unique(chose)) == 2:
            beta_ch, p_ch = joint_choice_test(r, chose.astype(float),
                                              stim.astype(float), pL, rng)
        else:
            beta_ch, p_ch = np.nan, np.nan
        # (3) decisive error-trial test: does rate follow CHOICE or STIMULUS?
        cor, err = g["cor"].to_numpy(), g["err"].to_numpy()
        cR, cL = cor & (chose > 0), cor & (chose < 0)
        eR, eL = err & (chose > 0), err & (chose < 0)
        corr_diff = err_diff = np.nan
        if cR.sum() >= MIN_SIDE and cL.sum() >= MIN_SIDE \
                and eR.sum() >= MIN_SIDE and eL.sum() >= MIN_SIDE:
            corr_diff = r[cR].mean() - r[cL].mean()   # choice pref on correct (=stim pref)
            err_diff = r[eR].mean() - r[eL].mean()    # choice-organised rate on errors
        rec.append(dict(cell=cell, region=g["region"].iloc[0], eid=g["eid"].iloc[0],
                        n_trials=len(g), n_err=int(err.sum()), n0=int(g["is0"].sum()),
                        fr_window=g["fr_window"].iloc[0],
                        choice_beta=beta_ch, p_choice=p_ch,
                        corr_diff=corr_diff, err_diff=err_diff))
    c = pd.DataFrame(rec)
    m = c["p_choice"].notna()
    c.loc[m, "q_choice"] = false_discovery_control(c.loc[m, "p_choice"].to_numpy(),
                                                   method="bh")
    c["sig_choice_raw"] = c["p_choice"] < 0.05
    c["sig_choice_fdr"] = c["q_choice"] < FDR_Q
    c["follows_choice"] = np.sign(c["corr_diff"]) == np.sign(c["err_diff"])
    CELLS_CSV.parent.mkdir(parents=True, exist_ok=True)
    c.to_csv(CELLS_CSV, index=False)
    return f, c


def report(f, c):
    print("========== Task 1: decorrelated trial budget (HONEST) ==========")
    print(f"  cells={c['cell' if 'cell' in c else 'region'].nunique() if False else len(c)} | "
          f"error trials total={int(f['err'].sum())} | 0%-contrast total={int(f['is0'].sum())}")
    print(f"  per-cell error trials: median {int(c['n_err'].median())} "
          f"(IQR {int(c['n_err'].quantile(.25))}-{int(c['n_err'].quantile(.75))}); "
          f"per-cell 0% trials: median {int(c['n0'].median())}")
    n_jointable = int(c["p_choice"].notna().sum())
    n_errtest = int(c["corr_diff"].notna().sum())
    print(f"  cells usable for joint choice-vs-stim regression: {n_jointable}/{len(c)}")
    print(f"  cells usable for the error-trial decisive test (>={MIN_SIDE}/side "
          f"correct AND error): {n_errtest}/{len(c)}")

    # Task 1/4 PER-CELL: genuine stimulus-independent choice coding
    print("\n========== Task 2/4: choice coding controlling stimulus+prior ==========")
    chance = round(0.05 * n_jointable)
    print(f"  choice-beta perm p<0.05: {int(c['sig_choice_raw'].sum())} "
          f"(chance ~{chance} -> excess ~{int(c['sig_choice_raw'].sum())-chance}); "
          f"FDR q<{FDR_Q}: {int(c['sig_choice_fdr'].sum())}")

    # Task 3 DECISIVE population error analysis
    e = c[c["corr_diff"].notna()].copy()
    follow = int((np.sign(e["corr_diff"]) == np.sign(e["err_diff"])).sum())
    nuse = len(e)
    # sign test (binomial, two-sided) vs 0.5
    from scipy.stats import binomtest, spearmanr
    sign_p = binomtest(follow, nuse, 0.5).pvalue if nuse else np.nan
    rho, rho_p = (spearmanr(e["corr_diff"], e["err_diff"]) if nuse > 3 else (np.nan, np.nan))
    print("\n========== Task 3: DECISIVE error-trial choice-vs-stimulus ==========")
    print(f"  usable cells: {nuse} | follow CHOICE on errors: {follow} "
          f"({follow/nuse:.0%}) vs follow STIMULUS: {nuse-follow} "
          f"({(nuse-follow)/nuse:.0%})  | sign-test p={sign_p:.3g}")
    print(f"  population corr(err_diff, corr_diff) = {rho:+.3f} (p={rho_p:.3g})  "
          f"[positive -> follows CHOICE, negative -> follows STIMULUS]")

    # higher-power trial-level pooled test: z-score rate within each cell's error
    # trials, align by the cell's CORRECT-trial choice preference, pool all error
    # trials, test whether choice-congruent errors have higher rate (-> choice coding)
    zc, zi, per_cell = [], [], []
    for cell, g in f.groupby("cell"):
        cor, err, chose = g["cor"].to_numpy(), g["err"].to_numpy(), g["chose"].to_numpy()
        cR, cL = cor & (chose > 0), cor & (chose < 0)
        if cR.sum() < MIN_SIDE or cL.sum() < MIN_SIDE or err.sum() < 2 * MIN_SIDE:
            continue
        r = g[RATE].to_numpy(float)
        pref = np.sign(r[cR].mean() - r[cL].mean())
        er = r[err]
        ez = (er - er.mean()) / (er.std() + 1e-9)
        cong = chose[err] == pref
        zc.extend(ez[cong]); zi.extend(ez[~cong])
        per_cell.append((ez[cong], ez[~cong]))            # keep cell as resample unit
    from scipy.stats import mannwhitneyu
    mwu = mannwhitneyu(zc, zi, alternative="greater") if zc and zi else None
    pooled_p = mwu.pvalue if mwu else np.nan
    pooled_eff = (np.mean(zc) - np.mean(zi)) if (zc and zi) else np.nan
    # cell-clustered bootstrap CI (resample CELLS, the independent unit)
    lo = hi = np.nan
    if per_cell:
        rng = np.random.default_rng(0); boot = []
        idx = np.arange(len(per_cell))
        for _ in range(2000):
            bi = rng.choice(idx, len(idx), replace=True)
            cc = np.concatenate([per_cell[i][0] for i in bi])
            ii = np.concatenate([per_cell[i][1] for i in bi])
            if cc.size and ii.size:
                boot.append(cc.mean() - ii.mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  POOLED error trials (cell-aligned, n={len(zc)+len(zi)}, cells={len(per_cell)}): "
          f"choice-congruent z={np.mean(zc):+.3f} vs incongruent z={np.mean(zi):+.3f} "
          f"-> effect {pooled_eff:+.3f} SD [95% CI {lo:+.3f}, {hi:+.3f}], MWU p={pooled_p:.3g}")

    # Task 5 per-region purified decision census
    rows = []
    for reg in CORE_REGIONS:
        u = c[c["region"] == reg]
        ue = u[u["corr_diff"].notna()]
        fol = int((np.sign(ue["corr_diff"]) == np.sign(ue["err_diff"])).sum())
        gen = u[u["sig_choice_fdr"] == True]                       # noqa: E712
        gen_raw = u[u["sig_choice_raw"] == True]                   # noqa: E712
        pooledN = int(gen_raw["n_trials"].sum())
        rows.append(dict(region=reg, cells=len(u),
                         err_testable=len(ue), follow_choice=fol,
                         follow_stim=len(ue) - fol,
                         choice_sig_raw=len(gen_raw), choice_sig_fdr=len(gen),
                         pooledN_choice_raw=pooledN))
    rt = pd.DataFrame(rows)
    REGION_CSV.parent.mkdir(parents=True, exist_ok=True)
    rt.to_csv(REGION_CSV, index=False)
    print("\n========== Task 5: per-region genuine-decision census ==========")
    print(rt.to_string(index=False))

    make_figure(c, e, rt, rho)

    # Glanceable verdict
    print("\n========== GLANCEABLE VERDICT ==========")
    tot_sig = int(c["sig_choice_raw"].sum()); tot_fdr = int(c["sig_choice_fdr"].sum())
    print(f"  stimulus-independent choice coding (controlling stim+prior): "
          f"{tot_sig} cells raw p<.05 (excess ~{tot_sig-chance}), {tot_fdr} FDR")
    print(f"  on ERROR trials the population follows {'CHOICE' if pooled_eff>0 else 'STIMULUS'}: "
          f"pooled-trial MWU p={pooled_p:.3g} (effect ~{pooled_eff:.2f} SD); "
          f"cell-level {follow}/{nuse} follow choice (sign p={sign_p:.2g}, corr p={rho_p:.2g})")
    pooled_sig = (pooled_p < 0.05) and (pooled_eff > 0)
    if pooled_sig and c["sig_choice_fdr"].sum() >= 10:
        state = "REAL and poolable (per-cell + population)"
    elif pooled_sig:
        state = ("REAL at the POPULATION level but SMALL and per-cell-undetectable "
                 "(0 FDR) -> trial-starved / population-only")
    elif (rho > 0 and rho_p < 0.1):
        state = "weak/marginal, not significant -> trial-starved / inconclusive"
    else:
        state = "indistinguishable from stimulus"
    print(f"  -> stimulus-independent decision signal is: {state}")


def make_figure(c, e, rt, rho):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # A: decisive scatter — err_diff vs corr_diff (choice pref on correct vs errors)
    ax = axes[0]
    cmap = {r: col for r, col in zip(CORE_REGIONS,
            ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])}
    ax.scatter(e["corr_diff"], e["err_diff"], c=e["region"].map(cmap), alpha=0.7, s=24)
    lim = np.nanpercentile(np.abs(np.r_[e["corr_diff"], e["err_diff"]]), 98)
    ax.plot([-lim, lim], [-lim, lim], "g--", lw=1, label="follows CHOICE (y=x)")
    ax.plot([-lim, lim], [lim, -lim], "r--", lw=1, label="follows STIMULUS (y=-x)")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("choice-pref rate diff, CORRECT trials (R-L, Hz)")
    ax.set_ylabel("choice-organised rate diff, ERROR trials (R-L, Hz)")
    ax.set_title(f"Decisive test: errors follow choice or stimulus?  "
                 f"corr={rho:+.2f}", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper left")
    # B: per-region — error-trial follow-choice vs follow-stim + choice-sig cells
    ax = axes[1]
    x = np.arange(len(rt)); w = 0.27
    ax.bar(x - w, rt["follow_choice"], w, label="errors follow CHOICE", color="#2a8")
    ax.bar(x, rt["follow_stim"], w, label="errors follow STIMULUS", color="#c44")
    ax.bar(x + w, rt["choice_sig_raw"], w, label="choice-coding (p<.05)", color="#258")
    ax.set_xticks(x); ax.set_xticklabels(rt["region"])
    ax.set_ylabel("# cells")
    ax.set_xlabel("region")
    ax.set_title("Per-region genuine-decision census", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    fig.suptitle("Phase 2 step 3: genuine choice vs stimulus on decorrelated "
                 "(error + 0%-contrast) trials", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nSaved figure -> {FIG_PATH}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="use full-coverage features + write *_full outputs")
    args = ap.parse_args()
    if args.full:
        FEATURES_CSV = PROJECT_ROOT / "results" / "phase2_sel_features_full.csv"
        CELLS_CSV = PROJECT_ROOT / "results" / "phase2_choicestim_cells_full.csv"
        REGION_CSV = PROJECT_ROOT / "results" / "phase2_choicestim_region_full.csv"
        FIG_PATH = PROJECT_ROOT / "figures" / "phase2_choice_vs_stim_full.png"
    f, c = analyze()
    report(f, c)

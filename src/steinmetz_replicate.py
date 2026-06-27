"""Replicate the two empirical IBL findings on Steinmetz-2019 + place MRN's per-cell decision
effect against the 0.57 population-arm gate (PURE; reads results/steinmetz_features.csv).

Ports the IBL phase2 statistical cores faithfully (same permutation choice-AUC, BH-FDR, within-
fold residualisation, cell-aligned pooled error effect, pseudo-population L2 decode) and runs
them on the Steinmetz adapter output. Differences from IBL, by design:
  - movement control = wheel + face motion-energy + pupil (NO body DLC -> ~19% body-strip only
    partially reproducible: caveat);
  - NO block prior in Steinmetz (one fewer confound; pL dropped);
  - decorrelated trials = ERROR + EQUAL-contrast Go trials (Steinmetz equal-contrast == IBL 0%).

  python src/steinmetz_replicate.py            # all analyses + gate + comparison
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import (rankdata, false_discovery_control, binomtest,  # noqa: E402
                         spearmanr, mannwhitneyu)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ibl_one import PROJECT_ROOT  # noqa: E402

FEAT = PROJECT_ROOT / "results" / "steinmetz_features.csv"
SEL_CSV = PROJECT_ROOT / "results" / "steinmetz_selectivity_region.csv"
CVS_CSV = PROJECT_ROOT / "results" / "steinmetz_choicestim_region.csv"
DEC_CSV = PROJECT_ROOT / "results" / "steinmetz_decode.csv"
GATE_CSV = PROJECT_ROOT / "results" / "steinmetz_gate.csv"
CMP_CSV = PROJECT_ROOT / "results" / "steinmetz_vs_ibl.csv"
FIG = PROJECT_ROOT / "figures" / "steinmetz_vs_ibl.png"

REGIONS = ["MRN", "SC", "SNr"]
MOVE_COLS = ["wheel_speed", "wheel_disp", "paw_speed", "nose_speed"]   # wheel + face + pupil
N_PERM = 2000
FDR_Q = 0.05
MIN_SIDE = 8
# Apples-to-apples with IBL: the cascade ran on <=25% contrast (where choice~stimulus). At
# Steinmetz's full contrast set {0,.25,.5,1} the stimulus dominates, so the cascade's choice
# tests use CONTRAST_MAX. The clean stimulus-AND-prior-independent decision set is the
# EQUAL-contrast trials (signed==0: no stimulus side, no prior) -- the Steinmetz analog of
# IBL's 0%-contrast. (Low-contrast ERROR trials, IBL's decisive dissociation, are unusable
# here: median 0 per cell, since Steinmetz uses mostly high contrast.)
CONTRAST_MAX = 0.25
GATE_BAR_AUC = 0.57            # population-arm gate (from steinmetz_population_preflight)
GATE_BAR_SD = 0.24            # equivalent per-cell effect SD (calibration: AUC0.57 <-> 0.24 SD)
IBL_EFFECT_SD = 0.059        # IBL MRN pooled effect (audit fix: 0.085 was ALL-REGION; MRN-only
#                              is +0.059 SD, cell-clustered bootstrap CI [+0.001,+0.117], marginal)

# ---- ported pure stat cores (from phase2_selectivity / _choice_vs_stim / _population_decode) ----
def auc_perm(x, y, rng, n_perm=N_PERM):
    ok = np.isfinite(x); x, y = x[ok], y[ok]
    n = len(y); n1 = int(y.sum()); n0 = n - n1
    if n1 < MIN_SIDE or n0 < MIN_SIDE:
        return np.nan, np.nan, n0, n1
    r = rankdata(x)
    obs = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    pos = np.argsort(rng.random((n_perm, n)), axis=1)[:, :n1]
    null = (r[pos].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    p = (np.sum(np.abs(null - 0.5) >= abs(obs - 0.5)) + 1) / (n_perm + 1)
    return float(obs), float(p), n0, n1


def residual_rate(g, cols, ycol="rate_delib"):
    M, used = [np.ones(len(g))], []
    for c in cols:
        v = g[c].to_numpy(float)
        if np.isfinite(v).sum() >= MIN_SIDE and np.nanstd(v) > 0:
            v = np.where(np.isfinite(v), v, np.nanmean(v)); M.append(v); used.append(c)
    M = np.column_stack(M)
    y = g[ycol].to_numpy(float)
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    return y - M @ beta


def joint_choice_test(rate, chose, stim, rng, n_perm=1000):
    """Choice coefficient controlling stimulus side (NO prior in Steinmetz); permutation
    shuffles choice within stimulus-side strata."""
    n = len(rate)
    X = np.column_stack([np.ones(n), chose, stim])
    obs = np.linalg.lstsq(X, rate, rcond=None)[0][1]
    strata = [np.where(stim == s)[0] for s in (-1, 0, 1)]
    null = np.empty(n_perm)
    for k in range(n_perm):
        ch = chose.copy()
        for idx in strata:
            if idx.size > 1:
                ch[idx] = rng.permutation(ch[idx])
        null[k] = np.linalg.lstsq(np.column_stack([np.ones(n), ch, stim]), rate, rcond=None)[0][1]
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return float(obs), float(p)


# ---- decode helpers (ported from phase2_population_decode) ----
MIN_DEC = 4
N_TR, N_TE, R_OBS, N_PERM_D, R_SUB, RIDGE, LOGIT_C = 300, 120, 40, 80, 12, 1.0, 0.05


def _ridge_resid(rate, C, ifit, iap):
    Mf = np.c_[np.ones(len(ifit)), C[ifit]]
    beta = np.linalg.solve(Mf.T @ Mf + RIDGE * np.eye(Mf.shape[1]), Mf.T @ rate[ifit])
    return rate[iap] - np.c_[np.ones(len(iap)), C[iap]] @ beta


def _split_pools(data, cells, rng, perm):
    trp, tep = {}, {}
    for cell in cells:
        rate, C, chose = data[cell]; ch = chose.copy()
        if perm:
            rng.shuffle(ch)
        iL, iR = np.where(ch < 0)[0], np.where(ch > 0)[0]
        rng.shuffle(iL); rng.shuffle(iR)
        kL, kR = max(1, int(round(0.7 * len(iL)))), max(1, int(round(0.7 * len(iR))))
        tri = np.r_[iL[:kL], iR[:kR]]; tei = np.r_[iL[kL:], iR[kR:]]
        if iL[kL:].size == 0 or iR[kR:].size == 0:
            tei = np.r_[iL[-1:], iR[-1:]]
        rtr = _ridge_resid(rate, C, tri, tri); rte = _ridge_resid(rate, C, tri, tei)
        ctr, cte = ch[tri], ch[tei]
        trp[cell] = {-1: rtr[ctr < 0], 1: rtr[ctr > 0]}
        tep[cell] = {-1: rte[cte < 0], 1: rte[cte > 0]}
    return trp, tep


def _pseudo(pool, cells, rng, n):
    K = len(cells); X = np.empty((2 * n, K))
    for j, cell in enumerate(cells):
        X[:n, j] = rng.choice(pool[cell][-1], n); X[n:, j] = rng.choice(pool[cell][1], n)
    return X, np.r_[np.zeros(n, int), np.ones(n, int)]


def decode(data, cells, rng, perm=False, reps=R_OBS):
    out = []
    for _ in range(reps):
        trp, tep = _split_pools(data, cells, rng, perm)
        Xtr, ytr = _pseudo(trp, cells, rng, N_TR); Xte, yte = _pseudo(tep, cells, rng, N_TE)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=LOGIT_C, max_iter=400).fit(sc.transform(Xtr), ytr)
        out.append(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))
    return np.array(out)


def per_cell_auc_cv(data, cells, rng):
    out = []
    for cell in cells:
        rate, C, chose = data[cell]; accs = []
        for _ in range(10):
            iL, iR = np.where(chose < 0)[0], np.where(chose > 0)[0]
            rng.shuffle(iL); rng.shuffle(iR)
            kL, kR = max(1, int(round(0.7 * len(iL)))), max(1, int(round(0.7 * len(iR))))
            tei = np.r_[iL[kL:], iR[kR:]]; tri = np.r_[iL[:kL], iR[:kR]]
            if iL[kL:].size == 0 or iR[kR:].size == 0:
                tei = np.r_[iL[-1:], iR[-1:]]
            rte = _ridge_resid(rate, C, tri, tei); yte = (chose[tei] > 0).astype(int)
            if yte.min() != yte.max():
                accs.append(roc_auc_score(yte, rte))
        if accs:
            out.append(np.mean(accs))
    return np.array(out)


def prep_decode(f, mode):
    """mode='decorr' (equal+error, broad), 'equal' (equal-contrast only = clean stimulus-AND-
    prior-independent decision set; Steinmetz analog of IBL error-only-strict)."""
    f = f.copy()
    f["chose"] = -np.sign(f["choice"]); f["stim"] = np.sign(f["signed"])
    f["equal"] = f["signed"] == 0
    f["err"] = (f["signed"] != 0) & (f["chose"] != f["stim"])
    dec = f[f["equal"]] if mode == "equal" else f[f["equal"] | f["err"]]
    data, region = {}, {}
    for cell, g in dec.groupby("cell"):
        chose = g["chose"].to_numpy()
        if (chose < 0).sum() < MIN_DEC or (chose > 0).sum() < MIN_DEC:
            continue
        C = np.nan_to_num(np.c_[g["wheel_speed"].to_numpy(float), g["paw_speed"].to_numpy(float),
                                g["nose_speed"].to_numpy(float), g["signed"].to_numpy(float)], nan=0.0)
        data[cell] = (g["rate_delib"].to_numpy(float), C, chose); region[cell] = g["region"].iloc[0]
    return data, region


# ============================ analyses ============================
def run_selectivity(feats):
    """Movement+stimulus-controlled choice cascade. Run on all-contrast Go trials for power;
    the stimulus REGRESSION (stim_surv) isolates the stimulus-independent choice subset (the
    triple count). Note: at full contrast the RAW choice AUC is stimulus-inflated, so the
    decisive outputs are stim_surv and triple, not choice_raw."""
    rng = np.random.default_rng(0); rec = []
    for cell, g in feats.groupby("cell"):
        y = (g["choice"].to_numpy() > 0).astype(int)
        a_d, p_d, n0, n1 = auc_perm(g["rate_delib"].to_numpy(float), y, rng)
        _, p_e, *_ = auc_perm(g["rate_early"].to_numpy(float), y, rng)
        _, p_p, *_ = auc_perm(g["rate_peri"].to_numpy(float), y, rng)
        _, p_mv, *_ = auc_perm(residual_rate(g, MOVE_COLS), y, rng)        # wheel+face+pupil
        _, p_st, *_ = auc_perm(residual_rate(g, ["signed"]), y, rng)       # stimulus (no prior)
        rec.append(dict(cell=cell, region=g["region"].iloc[0], eid=g["eid"].iloc[0],
                        n_left=n0, n_right=n1, auc_delib=a_d, p_delib=p_d,
                        p_early=p_e, p_peri=p_p, p_move=p_mv, p_stim=p_st))
    c = pd.DataFrame(rec)
    m = c["p_delib"].notna()
    c.loc[m, "q_delib"] = false_discovery_control(c.loc[m, "p_delib"].to_numpy(), method="bh")
    c["sig_raw"] = c["p_delib"] < 0.05; c["sig_fdr"] = c["q_delib"] < FDR_Q
    c["leading"] = c["sig_raw"] & (c["p_early"] < 0.05)
    c["locked"] = c["sig_raw"] & ~(c["p_early"] < 0.05) & (c["p_peri"] < 0.05)
    c["move_surv"] = c["sig_raw"] & (c["p_move"] < 0.05)
    c["stim_surv"] = c["sig_raw"] & (c["p_stim"] < 0.05)
    c["triple"] = c["sig_raw"] & c["leading"] & c["move_surv"] & c["stim_surv"]
    rows = []
    for reg in REGIONS:
        u = c[c["region"] == reg]
        rows.append(dict(region=reg, raw_hiFR=len(u), choice_raw=int(u.sig_raw.sum()),
                         choice_fdr=int(u.sig_fdr.sum()), leading=int(u.leading.sum()),
                         locked=int(u.locked.sum()), move_surv=int(u.move_surv.sum()),
                         stim_surv=int(u.stim_surv.sum()), triple=int(u.triple.sum()),
                         sessions=u.eid.nunique()))
    rt = pd.DataFrame(rows); rt.to_csv(SEL_CSV, index=False)
    print("\n===== FINDING 1: movement+stimulus-controlled selectivity cascade =====")
    print(rt.to_string(index=False))
    return c, rt


def run_choicestim(feats):
    """Stimulus-independent choice coding on EQUAL-contrast trials (no stimulus side, no prior).
    Per cell: choice AUC on equal-contrast trials (permutation -> FDR); transfer test = does the
    cell's CORRECT-trial choice preference carry to pure-choice (equal) trials."""
    f = feats.copy()
    f["chose"] = -np.sign(f["choice"]); f["stim"] = np.sign(f["signed"])
    f["equal"] = f["signed"] == 0
    f["err"] = (f["signed"] != 0) & (f["chose"] != f["stim"])
    f["cor"] = (f["signed"] != 0) & (f["chose"] == f["stim"])
    rng = np.random.default_rng(0); rec = []
    for cell, g in f.groupby("cell"):
        r = g["rate_delib"].to_numpy(float); chose = g["chose"].to_numpy()
        eq = g["equal"].to_numpy()
        yq = (chose[eq] > 0).astype(int)
        p = auc_perm(r[eq], yq, rng)[1] if eq.sum() >= 2 * MIN_SIDE else np.nan
        cor = g["cor"].to_numpy()
        cR, cL = cor & (chose > 0), cor & (chose < 0)
        eR, eL = eq & (chose > 0), eq & (chose < 0)
        cd = ed = np.nan
        if cR.sum() >= 3 and cL.sum() >= 3 and eR.sum() >= 3 and eL.sum() >= 3:
            cd = r[cR].mean() - r[cL].mean(); ed = r[eR].mean() - r[eL].mean()
        rec.append(dict(cell=cell, region=g["region"].iloc[0], p_choice=p,
                        corr_diff=cd, equal_diff=ed))
    c = pd.DataFrame(rec)
    m = c["p_choice"].notna()
    c.loc[m, "q_choice"] = false_discovery_control(c.loc[m, "p_choice"].to_numpy(), method="bh")
    c["sig_raw"] = c["p_choice"] < 0.05; c["sig_fdr"] = c["q_choice"] < FDR_Q
    rows = []
    for reg in REGIONS:
        u = c[c["region"] == reg]; ut = u[u["corr_diff"].notna()]
        fol = int((np.sign(ut["corr_diff"]) == np.sign(ut["equal_diff"])).sum())
        eff, lo, hi, p_mwu = pooled_decision_effect(f[f.region == reg])
        rows.append(dict(region=reg, equal_testable=len(ut), transfer_choice=fol,
                         no_transfer=len(ut) - fol, choice_sig_raw=int(u.sig_raw.sum()),
                         choice_sig_fdr=int(u.sig_fdr.sum()), pooled_eff_sd=eff,
                         eff_lo=lo, eff_hi=hi, pooled_p=p_mwu))
    rt = pd.DataFrame(rows); rt.to_csv(CVS_CSV, index=False)
    print("\n===== FINDING 2: stimulus-independent choice coding on EQUAL-contrast trials =====")
    print(rt.to_string(index=False))
    return c, rt


def pooled_decision_effect(fr):
    """Cell-aligned pooled decision effect in SD on EQUAL-contrast trials (the clean stimulus-
    AND-prior-independent set), the Steinmetz analog of IBL's 0.085-SD error-trial statistic.
    pref is the cell's choice tuning sign from CORRECT trials; the effect asks whether that
    tuning transfers to pure-choice (equal-contrast) trials. + cell-clustered bootstrap CI."""
    zc, zi, per_cell = [], [], []
    for cell, g in fr.groupby("cell"):
        cor, eq, chose = g["cor"].to_numpy(), g["equal"].to_numpy(), g["chose"].to_numpy()
        cR, cL = cor & (chose > 0), cor & (chose < 0)
        if cR.sum() < 3 or cL.sum() < 3 or eq.sum() < 6:
            continue
        r = g["rate_delib"].to_numpy(float)
        pref = np.sign(r[cR].mean() - r[cL].mean())
        er = r[eq]; ez = (er - er.mean()) / (er.std() + 1e-9)
        cong = chose[eq] == pref
        if cong.sum() < 2 or (~cong).sum() < 2:
            continue
        zc.extend(ez[cong]); zi.extend(ez[~cong]); per_cell.append((ez[cong], ez[~cong]))
    if not (zc and zi):
        return np.nan, np.nan, np.nan, np.nan
    eff = np.mean(zc) - np.mean(zi)
    p = mannwhitneyu(zc, zi, alternative="greater").pvalue
    rng = np.random.default_rng(0); boot = []
    idx = np.arange(len(per_cell))
    for _ in range(2000):
        bi = rng.choice(idx, len(idx), replace=True)
        cc = np.concatenate([per_cell[i][0] for i in bi]); ii = np.concatenate([per_cell[i][1] for i in bi])
        if cc.size and ii.size:
            boot.append(cc.mean() - ii.mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(eff), float(lo), float(hi), float(p)


def run_decode(feats):
    data_b, reg_b = prep_decode(feats, mode="decorr")
    data_e, reg_e = prep_decode(feats, mode="equal")        # equal-only = clean strict
    rng = np.random.default_rng(0); rows = []
    for reg in REGIONS:
        cb = [c for c in data_b if reg_b[c] == reg]
        ce = [c for c in data_e if reg_e[c] == reg]
        if len(cb) < 5:
            print(f"  {reg}: too few cells ({len(cb)})"); continue
        obs = decode(data_b, cb, rng, reps=R_OBS); om = obs.mean()
        nm = np.array([decode(data_b, cb, rng, perm=True, reps=R_SUB).mean() for _ in range(N_PERM_D)])
        p = (np.sum(nm >= om) + 1) / (len(nm) + 1)
        pc = per_cell_auc_cv(data_e, ce, rng) if len(ce) >= 5 else np.array([np.nan])
        if len(ce) >= 5:
            oe = decode(data_e, ce, rng, reps=R_OBS); oem = oe.mean()
            ne = np.array([decode(data_e, ce, rng, perm=True, reps=R_SUB).mean() for _ in range(N_PERM_D)])
            pe = (np.sum(ne >= oem) + 1) / (len(ne) + 1)
        else:
            oem, pe = np.nan, np.nan
        rows.append(dict(region=reg, n_cells=len(cb), n_cells_equal=len(ce), decode_auc=om,
                         perm_p=p, equal_auc=oem, equal_p=pe, percell_mean_auc=pc.mean()))
        print(f"  {reg}: decorr(equal+err) {om:.3f} (p={p:.3g}) | EQUAL-only {oem:.3f} "
              f"(p={pe:.3g}) | per-cell {pc.mean():.3f} (n={len(cb)}, equal {len(ce)})")
    rt = pd.DataFrame(rows); rt.to_csv(DEC_CSV, index=False)
    print("\n===== FINDING 2b: pseudo-population decode (stimulus+prior-independent choice) =====")
    print("  (EQUAL-only = clean decision set: no stimulus side, no prior -- the apples-to-apples"
          "\n   analog of IBL's error-only-strict; decorr = equal+error broad set)")
    print(rt.to_string(index=False))
    return rt


def gate_measurement(feats, cvs_rt):
    """Place Steinmetz MRN per-cell decision effect against the 0.57 (=0.24 SD) population bar."""
    mrn = feats[feats.region == "MRN"].copy()
    mrn["chose"] = -np.sign(mrn["choice"]); mrn["stim"] = np.sign(mrn["signed"])
    mrn["equal"] = mrn["signed"] == 0
    dec = mrn[mrn["equal"]]                              # clean: equal-contrast (no stim, no prior)
    rng = np.random.default_rng(1)
    obs_dev, deb_dev = [], []
    for cell, g in dec.groupby("cell"):
        y = (g["chose"].to_numpy() > 0).astype(int)
        x = g["rate_delib"].to_numpy(float)
        n1, n0 = int(y.sum()), int((y == 0).sum())
        if n1 < MIN_DEC or n0 < MIN_DEC:
            continue
        r = rankdata(x); a = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        dev = abs(a - 0.5)
        # finite-sample null of |AUC-.5| by shuffling -> debias
        pos = np.argsort(rng.random((200, len(y))), axis=1)[:, :n1]
        nd = np.mean(np.abs((r[pos].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0) - 0.5))
        obs_dev.append(dev); deb_dev.append(max(dev - nd, 0.0))
    obs_auc = 0.5 + np.mean(obs_dev)
    deb_auc = 0.5 + np.mean(deb_dev)
    eff = cvs_rt[cvs_rt.region == "MRN"].iloc[0]                      # pooled effect SD route
    rows = [dict(metric="per-cell AUC (raw, equal-contrast)", steinmetz=obs_auc, bar=GATE_BAR_AUC,
                 ibl=0.53),
            dict(metric="per-cell AUC (finite-sample debiased)", steinmetz=deb_auc, bar=GATE_BAR_AUC,
                 ibl=0.53),
            dict(metric="pooled decision effect (SD)", steinmetz=eff.pooled_eff_sd, bar=GATE_BAR_SD,
                 ibl=IBL_EFFECT_SD)]
    gate = pd.DataFrame(rows); gate.to_csv(GATE_CSV, index=False)
    print("\n===== THE GATE: MRN per-cell decision effect vs 0.57 (=0.24 SD) =====")
    print(gate.to_string(index=False))
    # decision on the SD route (apples-to-apples with IBL's 0.085 SD and the calibration)
    eff_sd = eff.pooled_eff_sd
    live = (eff_sd >= GATE_BAR_SD) and (eff.eff_lo > 0)
    print(f"\n  MRN pooled decision effect = {eff_sd:+.3f} SD [95% CI {eff.eff_lo:+.3f}, "
          f"{eff.eff_hi:+.3f}]  vs IBL {IBL_EFFECT_SD} SD  vs bar {GATE_BAR_SD} SD")
    print(f"  debiased per-cell AUC = {deb_auc:.3f}  vs bar {GATE_BAR_AUC}")
    print(f"  -> Steinmetz MRN per-cell effect is "
          f"{'>= 0.57 bar: POPULATION ARM LIVE' if live else '< 0.57 bar: ARM STAYS GATED'}")
    return gate, deb_auc, eff_sd, bool(live)


def comparison(sel_rt, cvs_rt, dec_rt, deb_auc, eff_sd, live):
    # IBL benchmarks (from results CSVs)
    ibl_sel = {"MRN": dict(raw=777, fdr=47, lead=97, triple=29),
               "SC": dict(raw=427, fdr=27, lead=57, triple=17),
               "SNr": dict(raw=47, fdr=9, lead=17, triple=3)}
    ibl_dec = {"MRN": 0.79, "SC": 0.37, "SNr": 0.33}
    rows = []
    for reg in REGIONS:
        s = sel_rt[sel_rt.region == reg].iloc[0]
        cv = cvs_rt[cvs_rt.region == reg].iloc[0]
        d = dec_rt[dec_rt.region == reg]
        rows.append(dict(region=reg,
                         ibl_triple=ibl_sel[reg]["triple"], stein_triple=int(s.triple),
                         ibl_choice_fdr=ibl_sel[reg]["fdr"], stein_choice_fdr=int(s.choice_fdr),
                         ibl_decode_strict=ibl_dec[reg],
                         stein_decode_equal=float(d.equal_auc.iloc[0]) if len(d) else np.nan,
                         stein_decode_decorr=float(d.decode_auc.iloc[0]) if len(d) else np.nan,
                         stein_pooled_sd=cv.pooled_eff_sd, stein_pooled_p=cv.pooled_p))
    cmp = pd.DataFrame(rows); cmp.to_csv(CMP_CSV, index=False)
    print("\n===== IBL vs STEINMETZ replication table =====")
    print(cmp.to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(REGIONS)); w = 0.38
    a = ax[0]
    a.bar(x - w / 2, [cmp[cmp.region == r].ibl_triple.iloc[0] for r in REGIONS], w, label="IBL", color="#888")
    a.bar(x + w / 2, [cmp[cmp.region == r].stein_triple.iloc[0] for r in REGIONS], w, label="Steinmetz", color="#27a")
    a.set_xticks(x); a.set_xticklabels(REGIONS); a.set_ylabel("triple-filtered decision cells")
    a.set_title("Finding 1: selectivity cascade (triple)", fontsize=10, loc="left"); a.legend(fontsize=8)
    a = ax[1]
    iv = [cmp[cmp.region == r].ibl_decode_strict.iloc[0] for r in REGIONS]
    sv = [cmp[cmp.region == r].stein_decode_equal.iloc[0] for r in REGIONS]
    a.bar(x - w / 2, iv, w, label="IBL (error-only)", color="#888")
    a.bar(x + w / 2, sv, w, label="Steinmetz (equal-only)", color="#1a5")
    a.axhline(0.5, color="k", ls="--", lw=1); a.set_xticks(x); a.set_xticklabels(REGIONS)
    a.set_ylabel("strict decode AUC (stim+prior-indep)"); a.set_ylim(0.3, 0.95)
    a.set_title("Finding 2b: distributed choice decode", fontsize=10, loc="left"); a.legend(fontsize=8)
    a = ax[2]
    bars = [("IBL\nMRN", IBL_EFFECT_SD, "#888"), ("Steinmetz\nMRN", eff_sd, "#1a5" if live else "#c44")]
    a.bar([0, 1], [b[1] for b in bars], color=[b[2] for b in bars])
    cv = cvs_rt[cvs_rt.region == "MRN"].iloc[0]
    a.errorbar([1], [eff_sd], yerr=[[eff_sd - cv.eff_lo], [cv.eff_hi - eff_sd]], color="k", capsize=4)
    a.axhline(GATE_BAR_SD, color="r", ls="--", lw=1.5, label="0.57 gate (=0.24 SD)")
    a.set_xticks([0, 1]); a.set_xticklabels([b[0] for b in bars])
    a.set_ylabel("MRN pooled decision effect (SD)")
    a.set_title(f"THE GATE: per-cell effect vs 0.57\n-> arm {'LIVE' if live else 'GATED'}",
                fontsize=10, loc="left"); a.legend(fontsize=8)
    fig.suptitle("Steinmetz-2019 replication of the IBL decision-coding findings + the 0.57 "
                 "population-arm gate", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=150)
    print(f"\nSaved comparison figure -> {FIG}")
    return cmp


def main():
    feats = pd.read_csv(FEAT)
    print(f"Steinmetz features: {feats['cell'].nunique()} hi-FR cells, "
          f"{feats.groupby('region')['cell'].nunique().to_dict()}, {feats.eid.nunique()} sessions")
    sel_c, sel_rt = run_selectivity(feats)
    cvs_c, cvs_rt = run_choicestim(feats)
    dec_rt = run_decode(feats)
    gate, deb_auc, eff_sd, live = gate_measurement(feats, cvs_rt)
    comparison(sel_rt, cvs_rt, dec_rt, deb_auc, eff_sd, live)


if __name__ == "__main__":
    main()

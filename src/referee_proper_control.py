"""Proper-control calibration + real-data read-out — THE FORK. PURE; cached data only.

The expanded control over-corrects (eats movement-orthogonal signal by overfitting ~14 regressors
on ~30-trial sessions). This decides: under a control CALIBRATED to remove movement without eating
signal, where does the REAL decision signal land vs the 0.57 AUC / 0.24 SD bar?

INTEGRITY RULE: the control is chosen by SYNTHETIC calibration (Part 1), FROZEN, then applied to the
real data (Parts 2-3). The control is NEVER selected by its real-data answer.

Candidate controls (pluggable residualizations; the decoder is otherwise the published one):
  none, linear, expanded            -- baselines (expanded = the published control)
  ridge_expanded                    -- expanded feats, L2 by inner CV per fold (RidgeCV)
  crossfit_expanded                 -- expanded feats, cross-fitted residualization (double-ML:
                                       no trial's residual uses a movement model that saw it)
  pca_expanded                      -- expanded feats reduced to top movement PCs (85% var) first

  ./.venv/Scripts/python.exe src/referee_proper_control.py --stage check
  ./.venv/Scripts/python.exe src/referee_proper_control.py --stage calibrate
  ./.venv/Scripts/python.exe src/referee_proper_control.py --stage real
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV

# live progress when stdout is redirected to a file (Python block-buffers redirects by default)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ibl_one import PROJECT_ROOT
import audit_realtrial_decode as D
import referee_injection_control as IC

R = PROJECT_ROOT / "results"
OUT = R / "referee_response" / "proper_control"
OUT.mkdir(parents=True, exist_ok=True)

CONTROLS = ["none", "linear", "expanded", "ridge_expanded", "crossfit_expanded", "pca_expanded"]
C_FIXED = D.C_FIXED
RHO_REAL = 0.493


# ============================ pluggable residualization ============================
def resid_fold(X, Mraw, control, tr, te, rng):
    """Return (Xtr_resid, Xte_resid). Movement regressed out per the named control."""
    if control == "none":
        return X[tr], X[te]
    if control in ("linear", "expanded"):
        M = D.movement_design(Mraw, control)
        return D._resid(X, M, tr, te)                      # EXACT published OLS residualization
    M = D.movement_design(Mraw, "expanded")
    if control == "ridge_expanded":
        sc = StandardScaler().fit(M[tr])
        Atr, Ate = sc.transform(M[tr]), sc.transform(M[te])
        rg = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(Atr, X[tr])   # alpha by inner LOO-CV
        return X[tr] - rg.predict(Atr), X[te] - rg.predict(Ate)
    if control == "pca_expanded":
        sc = StandardScaler().fit(M[tr])
        Atr, Ate = sc.transform(M[tr]), sc.transform(M[te])
        k = min(M.shape[1], max(1, len(tr) - 2))
        pca = PCA(n_components=min(0.85, 0.999), svd_solver="full").fit(Atr[:, :k] if False else Atr)
        Ptr, Pte = pca.transform(Atr), pca.transform(Ate)
        a = np.c_[np.ones(len(tr)), Ptr]; beta = np.linalg.lstsq(a, X[tr], rcond=None)[0]
        ae = np.c_[np.ones(len(te)), Pte]
        return X[tr] - a @ beta, X[te] - ae @ beta
    if control == "crossfit_expanded":
        # test: residual from movement model fit on FULL train (out-of-sample for test)
        af = np.c_[np.ones(len(tr)), M[tr]]; bf = np.linalg.lstsq(af, X[tr], rcond=None)[0]
        Xte_r = X[te] - np.c_[np.ones(len(te)), M[te]] @ bf
        # train: inner K-fold cross-fit so each train trial is out-of-sample too
        Xtr_r = np.empty((len(tr), X.shape[1]), float)
        Mtr = M[tr]; k = min(5, len(tr))
        inner = KFold(n_splits=max(2, k), shuffle=True, random_state=int(rng.integers(1 << 31)))
        for itr, ite in inner.split(np.arange(len(tr))):
            a = np.c_[np.ones(len(itr)), Mtr[itr]]; b = np.linalg.lstsq(a, X[tr][itr], rcond=None)[0]
            Xtr_r[ite] = X[tr][ite] - np.c_[np.ones(len(ite)), Mtr[ite]] @ b
        return Xtr_r, Xte_r
    raise ValueError(control)


def cv_decode_ctrl(X, y, Mraw, control, rng, n_rep):
    """Published per-session CV decoder, with pluggable movement control. Matches D.cv_decode for
    control in {none,linear,expanded}."""
    nsp = min(5, int(np.bincount(y).min()))
    if nsp < 2:
        return np.nan
    aucs = []
    for _ in range(n_rep):
        skf = StratifiedKFold(nsp, shuffle=True, random_state=int(rng.integers(1 << 31)))
        yp = np.full(len(y), np.nan)
        for tr, te in skf.split(X, y):
            Xtr, Xte = resid_fold(X, Mraw, control, tr, te, rng)
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(C=C_FIXED, max_iter=500, solver="liblinear")
            clf.fit(sc.transform(Xtr), y[tr])
            yp[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
        if np.unique(y).size == 2:
            aucs.append(roc_auc_score(y, yp))
    return float(np.mean(aucs)) if aucs else np.nan


# ============================ session loading ============================
def load_sessions(dataset, region):
    if dataset == "IBL":
        feats = pd.read_csv(R / "phase2_sel_features_full.csv"); f, dec = D.ibl_prep(feats)
    else:
        feats = pd.read_csv(R / "steinmetz_features.csv"); f, dec = D.stein_prep(feats)
    sub = f[f.region == region]
    sessions = []
    for eid, g in sub.groupby("eid"):
        sm = D.session_matrix(g, dec)
        if sm is None:
            continue
        X, y, Mraw = sm
        if D.decodable(X, y):
            sessions.append((eid, X, y, Mraw))
    return sessions


# ============================ stage: check (reproduce anchors) ============================
def run_check():
    print("=== reproduce-first anchors (pluggable decoder must match published) ===")
    sess = load_sessions("IBL", "MRN")
    print(f"  IBL MRN {len(sess)} decodable sessions")
    for control in ("none", "linear", "expanded"):
        obs = np.array([cv_decode_ctrl(X, y, Mraw, control, np.random.default_rng(7), D.N_REP_OBS)
                        for (_, X, y, Mraw) in sess])
        print(f"    {control:9s}: combined AUC = {np.nanmean(obs):.4f}")
    print("  (published: none 0.6149, linear 0.5805, expanded 0.5279)")
    print("\n  injection d=0 chance check (expanded), rho=0/0.493:")
    for rho in (0.0, RHO_REAL):
        r = IC.eval_cell(sess, 0.0, round(rho, 3), 6, 999)
        print(f"    rho={rho:.3f} d=0: none {r['auc_none']:.3f}  expanded {r['auc_expanded']:.3f}")


# ============================ stage: calibrate ============================
def calib_eval(sessions, d, rho, controls, n_synth, base_seed, n_rep=5):
    """Inject ONCE per synth-rep (shared across controls), decode with each control + none."""
    acc = {c: [] for c in controls}
    for rep in range(n_synth):
        rng = np.random.default_rng(base_seed + 1009 * rep)
        inj = []
        for (_, Xbg, _, Mraw) in sessions:
            Xinj, yv, _ = IC.inject_session(Xbg, Mraw, d, rho, rng)
            inj.append((Xinj, yv, Mraw))
        for c in controls:
            vals = [cv_decode_ctrl(Xinj, yv, Mraw, c, np.random.default_rng(7), n_rep)
                    for (Xinj, yv, Mraw) in inj]
            acc[c].append(np.nanmean(vals))
    out = {}
    for c in controls:
        a = np.array(acc[c]); m = a.mean()
        se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        out[c] = (float(m), float(m - 1.96 * se), float(m + 1.96 * se))
    return out


def run_calibrate():
    print("=== PART 1: calibrate candidate controls on SYNTHETIC data (blind to real) ===")
    sessions = load_sessions("IBL", "MRN")
    print(f"  MRN {len(sessions)} sessions; realistic rho = {RHO_REAL}")
    N_SYNTH = 14
    grid = [(0.0, 0.0), (0.24, 0.0),          # test1(d=0,rho=0) ; test2 (rho=0 d=0.24, PRESERVE)
            (0.0, 1.0), (0.24, 1.0),          # test1(d=0,rho=1) ; test3 (rho=1 d=0.24, REMOVE)
            (0.0, RHO_REAL), (0.06, RHO_REAL), (0.12, RHO_REAL), (0.18, RHO_REAL),
            (0.24, RHO_REAL), (0.30, RHO_REAL)]   # realistic-rho recovery curve
    rows = []
    cache = {}
    for d, rho in grid:
        res = calib_eval(sessions, d, round(rho, 3), CONTROLS, N_SYNTH, 31337)
        cache[(d, rho)] = res
        for c in CONTROLS:
            m, lo, hi = res[c]
            rows.append(dict(d=d, rho=rho, control=c, auc=m, lo=lo, hi=hi))
        print(f"  d={d:.2f} rho={rho:.3f}: " +
              "  ".join(f"{c.split('_')[0][:5]}={res[c][0]:.3f}" for c in CONTROLS))
    df = pd.DataFrame(rows); df.to_csv(OUT / "calibration.csv", index=False)

    # ---- the three pass/fail tests ----
    print("\n--- calibration tests ---")
    unc0 = cache[(0.24, 0.0)]["none"][0]            # uncontrolled, orthogonal signal
    unc1 = cache[(0.24, 1.0)]["none"][0]            # uncontrolled, pure-movement signal
    tol_chance = 0.025                              # |AUC-0.5| Monte-Carlo tolerance
    verdict = {}
    print(f"  (uncontrolled refs: orthogonal d=0.24 AUC={unc0:.3f}; pure-movement d=0.24 AUC={unc1:.3f})")
    print(f"  {'control':18s} {'T1 d=0(r0/r1)':>16s} {'T2 preserve':>22s} {'T3 remove':>14s}  PASS?")
    for c in CONTROLS:
        d0_r0 = cache[(0.0, 0.0)][c][0]; d0_r1 = cache[(0.0, 1.0)][c][0]
        t2 = cache[(0.24, 0.0)][c][0]                # should ~= unc0
        t3 = cache[(0.24, 1.0)][c][0]                # should ~= 0.5
        retain = (t2 - 0.5) / (unc0 - 0.5) if unc0 > 0.5 else np.nan
        p1 = abs(d0_r0 - 0.5) < tol_chance and abs(d0_r1 - 0.5) < tol_chance
        p2 = retain >= 0.85                          # preserves orthogonal within MC noise
        p3 = abs(t3 - 0.5) < 0.035                    # removes pure movement to ~chance
        ok = p1 and p2 and p3
        verdict[c] = dict(p1=p1, p2=p2, p3=p3, ok=ok, retain=retain, t3=t3,
                          real_curve_d24=cache[(0.24, RHO_REAL)][c][0])
        print(f"  {c:18s} {d0_r0:.3f}/{d0_r1:.3f}{'':5s} "
              f"{t2:.3f} (retain {retain:.2f}) {'OK' if p2 else 'X'}   "
              f"{t3:.3f} {'OK' if p3 else 'X'}    {'PASS' if ok else 'fail'}")

    passers = [c for c in CONTROLS if verdict[c]["ok"]]
    print(f"\n  controls passing all three: {passers}")
    if passers:
        # most conservative = most movement-removing among passers = LOWEST realistic-rho d=0.24 recovery
        frozen = min(passers, key=lambda c: verdict[c]["real_curve_d24"])
        why = (f"lowest realistic-rho d=0.24 recovery among passers "
               f"({verdict[frozen]['real_curve_d24']:.3f}) -> most movement-removing while still "
               f"preserving the orthogonal signal (retain {verdict[frozen]['retain']:.2f})")
    else:
        frozen = min(CONTROLS, key=lambda c: (not verdict[c]["p2"], -verdict[c]["retain"]))
        why = "NO control passed all three; closest by preserve-test reported"
    print(f"  >>> FROZEN CONTROL = {frozen}  ({why})")
    print(f"\n  realistic-rho (={RHO_REAL}) recovery curve for FROZEN control '{frozen}':")
    for d in (0.0, 0.06, 0.12, 0.18, 0.24, 0.30):
        m, lo, hi = cache[(d, RHO_REAL)][frozen]
        print(f"    d={d:.2f}: AUC {m:.3f} [{lo:.3f},{hi:.3f}]" + ("   <- threshold" if d == 0.24 else ""))
    pd.DataFrame([dict(frozen=frozen, reason=why, **{f"{c}_pass": verdict[c]["ok"] for c in CONTROLS})]
                 ).to_csv(OUT / "frozen_control.csv", index=False)
    print(f"\nSaved -> {OUT/'calibration.csv'}, {OUT/'frozen_control.csv'}")
    return frozen


# ============================ stage: real-data read-out (Parts 2-3) ============================
def residualize_full(rate, Mraw, control, rng):
    """Per-cell movement residualization over ALL the cell's trials (not fold-based)."""
    rate = rate.astype(float)
    if control == "none":
        return rate
    if control in ("linear", "expanded"):
        M = D.movement_design(Mraw, control)
        A = np.c_[np.ones(len(rate)), M]
        return rate - A @ np.linalg.lstsq(A, rate, rcond=None)[0]
    M = D.movement_design(Mraw, "expanded")
    if control == "ridge_expanded":
        A = StandardScaler().fit_transform(M)
        rg = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(A, rate)
        return rate - rg.predict(A)
    if control == "pca_expanded":
        A = StandardScaler().fit_transform(M)
        P = PCA(n_components=min(0.85, 0.999), svd_solver="full").fit_transform(A)
        a = np.c_[np.ones(len(rate)), P]
        return rate - a @ np.linalg.lstsq(a, rate, rcond=None)[0]
    if control == "crossfit_expanded":
        out = np.empty(len(rate), float)
        kf = KFold(n_splits=min(5, len(rate)), shuffle=True, random_state=int(rng.integers(1 << 31)))
        for itr, ite in kf.split(rate):
            a = np.c_[np.ones(len(itr)), M[itr]]; b = np.linalg.lstsq(a, rate[itr], rcond=None)[0]
            out[ite] = rate[ite] - np.c_[np.ones(len(ite)), M[ite]] @ b
        return out
    raise ValueError(control)


def decode_real(sessions, control, rng, n_rep=D.N_REP_OBS, n_perm=200):
    """Combined real decode AUC + session-bootstrap CI + permutation p for a control."""
    obs = np.array([cv_decode_ctrl(X, y, Mraw, control, np.random.default_rng(7), n_rep)
                    for (_, X, y, Mraw) in sessions])
    comb = float(np.nanmean(obs))
    brng = np.random.default_rng(0)
    boot = [np.nanmean(obs[brng.integers(0, len(obs), len(obs))]) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    nulls = np.empty(n_perm)
    for k in range(n_perm):
        vals = []
        for (_, X, y, Mraw) in sessions:
            ys = y.copy(); rng.shuffle(ys)
            vals.append(cv_decode_ctrl(X, ys, Mraw, control, rng, 2))
        nulls[k] = np.nanmean(vals)
    p = (np.sum(nulls >= comb) + 1) / (n_perm + 1)
    return dict(auc=comb, lo=float(lo), hi=float(hi), perm_p=float(p), n_sessions=len(obs))


def pooled_effect_controlled(f, region, control, trial="err"):
    """AC.pooled_effect logic on movement-residualized (control) per-cell rates; cell-clustered CI."""
    f = f[f.region == region].copy()
    f["chose"] = -np.sign(f["choice"]); f["stim"] = np.sign(f["signed"])
    if trial == "err":
        f["cor"] = (f["absc"] > 0) & (f["chose"] == f["stim"])
        f["dec"] = (f["absc"] > 0) & (f["chose"] != f["stim"])
    else:
        f["cor"] = (f["signed"] != 0) & (f["chose"] == f["stim"])
        f["dec"] = f["signed"] == 0
    rng = np.random.default_rng(0); per = []
    for cell, g in f.groupby("cell"):
        cor, dec, chose = g["cor"].to_numpy(), g["dec"].to_numpy(), g["chose"].to_numpy()
        cR, cL = cor & (chose > 0), cor & (chose < 0)
        if cR.sum() < 3 or cL.sum() < 3 or dec.sum() < 6:
            continue
        rate = residualize_full(g["rate_delib"].to_numpy(float), g[D.MOV].to_numpy(), control, rng)
        pref = np.sign(rate[cR].mean() - rate[cL].mean())
        er = rate[dec]; ez = (er - er.mean()) / (er.std() + 1e-9); cong = chose[dec] == pref
        if cong.sum() < 2 or (~cong).sum() < 2:
            continue
        per.append((ez[cong], ez[~cong]))
    if not per:
        return dict(eff=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_cells=0)
    zc = np.concatenate([p[0] for p in per]); zi = np.concatenate([p[1] for p in per])
    eff = zc.mean() - zi.mean()
    boot = []
    idx = np.arange(len(per))
    for _ in range(5000):
        bi = rng.choice(idx, len(idx), replace=True)
        cc = np.concatenate([per[i][0] for i in bi]); ii = np.concatenate([per[i][1] for i in bi])
        boot.append(cc.mean() - ii.mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(eff=float(eff), ci_lo=float(lo), ci_hi=float(hi), n_cells=len(per))


def percell_auc_controlled(f, region, control):
    """Per-cell choice AUC after control; returns (df with before/after, summary)."""
    f = f[f.region == region].copy()
    rng = np.random.default_rng(0); rec = []
    for cell, g in f.groupby("cell"):
        y = (g["choice"].to_numpy() > 0).astype(int)
        if y.sum() < 5 or (y == 0).sum() < 5:
            continue
        raw = g["rate_delib"].to_numpy(float)
        res = residualize_full(raw, g[D.MOV].to_numpy(), control, rng)
        rec.append(dict(cell=cell, auc_before=roc_auc_score(y, raw),
                        auc_after=roc_auc_score(y, res), n=len(y)))
    return pd.DataFrame(rec)


def run_real():
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    frozen = pd.read_csv(OUT / "frozen_control.csv")["frozen"].iloc[0]
    print(f"=== PARTS 2-3: real-data read-out through FROZEN control = '{frozen}' ===")
    show = ["none", "linear", "expanded", frozen]
    seen = []; [seen.append(c) for c in show if c not in seen]; show = seen
    plot_rows = []
    blocks = [("IBL", "MRN", "err"), ("IBL", "SCm", "err"), ("Steinmetz", "MRN", "equal")]
    feat_cache = {}
    for dataset, region, trial in blocks:
        print(f"\n----- {dataset} {region} -----")
        sessions = load_sessions(dataset, region)
        print(f"  {len(sessions)} decodable sessions")
        if dataset not in feat_cache:
            path = R / ("phase2_sel_features_full.csv" if dataset == "IBL" else "steinmetz_features.csv")
            feat_cache[dataset] = pd.read_csv(path)
        feats = feat_cache[dataset]
        for control in show:
            dec = decode_real(sessions, control, np.random.default_rng(0),
                              n_perm=200 if control == frozen else 60)
            pe = pooled_effect_controlled(feats, region, control, trial)
            tag = " [FROZEN]" if control == frozen else ""
            print(f"  {control:18s}: decode AUC {dec['auc']:.3f} [{dec['lo']:.3f},{dec['hi']:.3f}] "
                  f"p={dec['perm_p']:.3f} | pooled {pe['eff']:+.3f} SD "
                  f"[{pe['ci_lo']:+.3f},{pe['ci_hi']:+.3f}] ({pe['n_cells']} cells){tag}")
            plot_rows.append(dict(dataset=dataset, region=region, control=control,
                                  auc=dec["auc"], lo=dec["lo"], hi=dec["hi"], perm_p=dec["perm_p"],
                                  pooled_sd=pe["eff"], pooled_lo=pe["ci_lo"], pooled_hi=pe["ci_hi"],
                                  frozen=(control == frozen)))
        # per-cell AUC distribution under frozen
        pc = percell_auc_controlled(feats, region, frozen)
        print(f"  per-cell choice-AUC under '{frozen}': median before {pc.auc_before.median():.3f} "
              f"-> after {pc.auc_after.median():.3f}; "
              f"frac after>0.57 = {(pc.auc_after>0.57).mean():.2f} "
              f"(>0.65 = {(pc.auc_after>0.65).mean():.2f})")
        pc.to_csv(OUT / f"percell_{dataset}_{region}.csv", index=False)

    # the 3 within-region-FDR SCm cells, before vs after frozen control
    print("\n----- the 3 within-region-FDR SCm cells: choice-AUC before -> after frozen control -----")
    scm_cells = ["478de1ce-d7e7-4221-9365-2abdc6e88fb6:410",
                 "53ecbf4f-e0d8-4fe6-a852-8b934a37a1c2:600",
                 "d5e5311c-8beb-4f8f-b798-3e9bfa6bcdd8:275"]
    pc = percell_auc_controlled(feat_cache["IBL"], "SCm", frozen)
    sub = pc[pc.cell.isin(scm_cells)]
    for _, r in sub.iterrows():
        print(f"    {r.cell}: AUC {r.auc_before:.3f} -> {r.auc_after:.3f}  (n={int(r.n)})")

    df = pd.DataFrame(plot_rows); df.to_csv(OUT / "real_by_control.csv", index=False)

    # plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [f"{r.dataset}\n{r.region}" for r in df[df.control == show[0]].itertuples()]
    groups = list(df.groupby(["dataset", "region"], sort=False))
    width = 0.8 / len(show)
    x = np.arange(len(groups))
    colmap = {"none": "#bbb", "linear": "#6aa", "expanded": "#c77", frozen: "#27a"}
    for j, control in enumerate(show):
        vals = [g[g.control == control].auc.iloc[0] for _, g in groups]
        los = [g[g.control == control].lo.iloc[0] for _, g in groups]
        his = [g[g.control == control].hi.iloc[0] for _, g in groups]
        err = [np.array(vals) - np.array(los), np.array(his) - np.array(vals)]
        lbl = control + (" [FROZEN]" if control == frozen else "")
        ax.bar(x + j * width, vals, width, yerr=err, capsize=3,
               color=colmap.get(control, "#999"), label=lbl)
    ax.axhline(0.57, color="crimson", ls="--", lw=1.5, label="0.57 bar")
    ax.axhline(0.50, color="gray", ls=":", lw=1.0)
    ax.set_xticks(x + width * (len(show) - 1) / 2)
    ax.set_xticklabels([f"{d}\n{r}" for (d, r), _ in groups])
    ax.set_ylabel("real movement-controlled decode AUC")
    ax.set_title(f"Real decision-signal decode by movement control (frozen = {frozen})",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "real_by_control.png", dpi=150)
    print(f"\nSaved -> {OUT/'real_by_control.csv'}, {OUT/'real_by_control.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["check", "calibrate", "real", "all"], default="check")
    a = ap.parse_args()
    if a.stage in ("check", "all"):
        run_check()
    if a.stage in ("calibrate", "all"):
        run_calibrate()
    if a.stage in ("real", "all"):
        run_real()

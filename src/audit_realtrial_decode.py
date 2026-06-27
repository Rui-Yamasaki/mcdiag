"""Decisive test of the main positive finding: REAL-TRIAL per-session CV decode of stimulus-
independent choice, replacing the overfitting pseudo-trial resampling (PURE; cached data only).

The 243-cell "distributed MRN code" is a PSEUDO-population (cells from ~120 disjoint sessions);
a true real-simultaneous-trial decode is only possible WITHIN a session. So we decode per session
(cells co-recorded, real decorrelated trials), held-out CV, movement residualised within fold,
a-priori-fixed L2 (no leakage), and aggregate across sessions (the independent unit) with a
properly-specified permutation null (one fixed label shuffle per draw; combined statistic = mean
session AUC).

  - IBL decorrelated = 0%-contrast + error;  Steinmetz = equal-contrast.
  - movement models: none | linear (wheel_speed, wheel_disp, body, arousal) | EXPANDED (degree-2
    polynomial + pairwise interactions of those 4 -> nonlinear, multidimensional). [True within-trial
    wheel velocity/accel time-series would need re-extraction; the binned summaries already include
    velocity-magnitude (wheel_speed); the polynomial adds the nonlinearity. Stated limitation.]

  python src/audit_realtrial_decode.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ibl_one import PROJECT_ROOT

R = PROJECT_ROOT / "results"
MOV = ["wheel_speed", "wheel_disp", "paw_speed", "nose_speed"]   # body=paw/face, arousal=nose/pupil
MIN_CELLS, MIN_SIDE = 8, 5
C_FIXED = 0.1               # a-priori L2 (no leakage); strong enough for p>n sessions
N_REP_OBS, N_REP_NULL, N_PERM = 6, 2, 400


def movement_design(Mraw, mode):
    if mode == "none":
        return None
    M = np.nan_to_num(Mraw.astype(float), nan=0.0)
    if mode == "linear":
        return M
    # expanded: squares + pairwise interactions (nonlinear, multidimensional)
    cols = [M]
    cols.append(M ** 2)
    p = M.shape[1]
    inter = [M[:, i] * M[:, j] for i in range(p) for j in range(i + 1, p)]
    if inter:
        cols.append(np.column_stack(inter))
    X = np.column_stack(cols)
    return X[:, np.std(X, axis=0) > 1e-9]


def _resid(X, M, tr, te):
    if M is None:
        return X[tr], X[te]
    A = np.c_[np.ones(len(tr)), M[tr]]
    beta = np.linalg.lstsq(A, X[tr], rcond=None)[0]
    At = np.c_[np.ones(len(te)), M[te]]
    return X[tr] - A @ beta, X[te] - At @ beta


def cv_decode(X, y, Mraw, mode, rng, n_rep):
    """Held-out stratified-k-fold CV AUC; movement residualised + standardised within fold;
    fixed a-priori L2. Returns mean held-out AUC over repeats (pooled predictions per repeat)."""
    M = movement_design(Mraw, mode)
    nsp = min(5, int(np.bincount(y).min()))
    if nsp < 2:
        return np.nan
    aucs = []
    for _ in range(n_rep):
        skf = StratifiedKFold(nsp, shuffle=True, random_state=int(rng.integers(1 << 31)))
        yp = np.full(len(y), np.nan)
        for tr, te in skf.split(X, y):
            Xtr, Xte = _resid(X, M, tr, te)
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(C=C_FIXED, max_iter=500, solver="liblinear")
            clf.fit(sc.transform(Xtr), y[tr])
            yp[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
        if np.unique(y).size == 2:
            aucs.append(roc_auc_score(y, yp))
    return float(np.mean(aucs)) if aucs else np.nan


def session_matrix(g, dec_mask_fn):
    """Pivot one session to (X = trials x cells real rates, y = chose, Mraw = movement per trial)
    on decorrelated trials. All cells in a session share the same ordered trials (simultaneous)."""
    cells = list(g["cell"].unique())
    t0 = g[g["cell"] == cells[0]].reset_index(drop=True)
    dec = dec_mask_fn(t0).to_numpy()
    nT = len(t0)
    rate = {}
    for c in cells:
        v = g[g["cell"] == c]["rate_delib"].to_numpy()
        if len(v) != nT:
            return None
        rate[c] = v
    X = np.column_stack([rate[c][dec] for c in cells])
    y = (t0["chose"].to_numpy()[dec] > 0).astype(int)
    Mraw = t0[MOV].to_numpy()[dec]
    return X, y, Mraw


def decodable(X, y):
    return (X.shape[1] >= MIN_CELLS and (y == 0).sum() >= MIN_SIDE and (y == 1).sum() >= MIN_SIDE)


def run_region(feats, region, dec_mask_fn, rng):
    sub = feats[feats.region == region]
    sessions = []
    for eid, g in sub.groupby("eid"):
        sm = session_matrix(g, dec_mask_fn)
        if sm is None:
            continue
        X, y, Mraw = sm
        if decodable(X, y):
            sessions.append((eid, X, y, Mraw))
    out = {}
    for mode in ("none", "linear", "expanded"):
        obs = np.array([cv_decode(X, y, Mraw, mode, np.random.default_rng(7), N_REP_OBS)
                        for (_, X, y, Mraw) in sessions])
        comb_obs = np.nanmean(obs)
        # proper combined null: one fixed shuffle per session per perm; stat = mean session AUC
        nulls = np.empty(N_PERM)
        for k in range(N_PERM):
            vals = []
            for (_, X, y, Mraw) in sessions:
                ys = y.copy(); rng.shuffle(ys)
                vals.append(cv_decode(X, ys, Mraw, mode, rng, N_REP_NULL))
            nulls[k] = np.nanmean(vals)
        p = (np.sum(nulls >= comb_obs) + 1) / (N_PERM + 1)
        out[mode] = dict(region=region, mode=mode, n_sessions=len(sessions),
                         mean_cells=float(np.mean([X.shape[1] for _, X, _, _ in sessions])),
                         mean_trials=float(np.mean([len(y) for _, _, y, _ in sessions])),
                         cv_auc=float(comb_obs), perm_p=float(p),
                         null_mean=float(nulls.mean()), null_p95=float(np.percentile(nulls, 95)),
                         frac_sess_auc_gt55=float(np.mean(obs > 0.55)))
    return out


def ibl_prep(feats):
    f = feats.copy()
    f["chose"] = -np.sign(f["choice"]); f["stim"] = np.sign(f["signed"])
    f["dec_is0"] = f["absc"] == 0
    f["dec_err"] = (f["absc"] > 0) & (f["chose"] != f["stim"])
    return f, (lambda t: t["dec_is0"] | t["dec_err"])


def stein_prep(feats):
    f = feats.copy()
    f["chose"] = -np.sign(f["choice"])
    f["dec_equal"] = f["signed"] == 0
    return f, (lambda t: t["dec_equal"])


def main():
    rows = []
    print("=== REAL-TRIAL per-session CV decode (movement: none / linear / EXPANDED-nonlinear) ===")
    print("    decodable session = >=8 simultaneous cells & >=5/side real decorrelated trials\n")
    for name, path, prep, regions in [
        ("IBL", R / "phase2_sel_features_full.csv", ibl_prep, ["MRN", "SCm", "SNr"]),
        ("Steinmetz", R / "steinmetz_features.csv", stein_prep, ["MRN", "SC", "SNr"]),
    ]:
        feats = pd.read_csv(path)
        f, dec_fn = prep(feats)
        print(f"--- {name} ---")
        for reg in regions:
            rng = np.random.default_rng(0)
            res = run_region(f, reg, dec_fn, rng)
            if not res:
                continue
            r0 = res["none"]
            print(f"  {reg}: {r0['n_sessions']} decodable sessions "
                  f"(mean {r0['mean_cells']:.0f} cells x {r0['mean_trials']:.0f} trials)")
            for mode in ("none", "linear", "expanded"):
                r = res[mode]; rows.append(dict(dataset=name, **r))
                flag = "sig" if r["perm_p"] < 0.05 else "n.s."
                print(f"    movement={mode:9s}: CV-AUC {r['cv_auc']:.3f}  perm-p {r['perm_p']:.3f} [{flag}]  "
                      f"(null {r['null_mean']:.3f}, p95 {r['null_p95']:.3f}; "
                      f"{r['frac_sess_auc_gt55']*100:.0f}% sess>0.55)")
    df = pd.DataFrame(rows)
    df.to_csv(R / "audit_realtrial_decode.csv", index=False)
    print(f"\nSaved -> {R / 'audit_realtrial_decode.csv'}")

    # verdict
    print("\n========== VERDICT ==========")
    for name in ("IBL", "Steinmetz"):
        m = df[(df.dataset == name) & (df.region.isin(["MRN"])) & (df["mode"] == "expanded")]
        if len(m):
            r = m.iloc[0]
            surv = (r.cv_auc > 0.55) and (r.perm_p < 0.05)
            print(f"  {name} MRN, real-trial CV + EXPANDED movement: AUC {r.cv_auc:.3f}, p {r.perm_p:.3f} "
                  f"-> {'SURVIVES' if surv else 'ERODES'}")


if __name__ == "__main__":
    main()

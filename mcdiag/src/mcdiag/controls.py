"""Candidate movement controls and the per-block cross-validated decoder.

A movement "control" is a residualization: it regresses a movement design matrix out of the
per-trial neural activity before a choice decoder is trained and scored. The decoder is held
fixed across controls (cross-validated L2 logistic regression), so any difference in decode
accuracy is attributable to the control, not the classifier.

Controls (all share the held-fixed decoder):
  none      no residualization (the uncontrolled reference).
  linear    ordinary least squares on the raw movement features.
  expanded  OLS on a degree-2 expansion (squares plus pairwise products) of the movement
            features. Flexible, and at low trial counts it overfits and removes genuine signal.
  ridge     L2-penalized regression on the expanded features, penalty chosen by inner
            cross-validation per fold.
  pca       OLS on the top principal components of the expanded features (85 percent variance).
            Removes high-variance movement but can leave low-variance movement untouched.
  crossfit  expanded features residualized with cross-fitting (double machine learning), so no
            trial's residual uses a movement model that was fit on that trial.

The numerics here are lifted unchanged from the calibration engine used in the paper.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

#: a-priori L2 penalty for the fixed decoder (no leakage, strong enough for the p greater than n regime)
C_FIXED = 0.1

#: the candidate controls, ordered from no control to most flexible
CONTROL_NAMES = ("none", "linear", "ridge", "pca", "expanded", "crossfit")


def movement_design(movement, mode):
    """Build the movement design matrix for a control.

    movement is [n_trials, n_features]. Returns None for 'none', the raw features for 'linear',
    and the degree-2 expansion (features, squares, pairwise products) for the expanded controls.
    """
    if mode == "none":
        return None
    M = np.nan_to_num(np.asarray(movement, dtype=float), nan=0.0)
    if mode == "linear":
        return M
    cols = [M, M ** 2]
    p = M.shape[1]
    inter = [M[:, i] * M[:, j] for i in range(p) for j in range(i + 1, p)]
    if inter:
        cols.append(np.column_stack(inter))
    X = np.column_stack(cols)
    return X[:, np.std(X, axis=0) > 1e-9]


def _ols_resid(X, M, tr, te):
    """Residualize X (trials by cells) on [1, M], fitting on the train fold, applying to both."""
    A = np.c_[np.ones(len(tr)), M[tr]]
    beta = np.linalg.lstsq(A, X[tr], rcond=None)[0]
    At = np.c_[np.ones(len(te)), M[te]]
    return X[tr] - A @ beta, X[te] - At @ beta


def residualize_fold(X, movement, control, tr, te, rng):
    """Return (train residual, test residual) for one decoder fold under the named control.

    X is [n_trials, n_cells]. The test residual is always computed from a movement model fit on
    the train fold only, so test scoring never sees a movement model trained on its own trials.
    """
    if control == "none":
        return X[tr], X[te]
    if control == "linear":
        return _ols_resid(X, movement_design(movement, "linear"), tr, te)
    if control == "expanded":
        return _ols_resid(X, movement_design(movement, "expanded"), tr, te)
    M = movement_design(movement, "expanded")
    if control == "ridge":
        sc = StandardScaler().fit(M[tr])
        Atr, Ate = sc.transform(M[tr]), sc.transform(M[te])
        rg = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(Atr, X[tr])
        return X[tr] - rg.predict(Atr), X[te] - rg.predict(Ate)
    if control == "pca":
        sc = StandardScaler().fit(M[tr])
        Atr, Ate = sc.transform(M[tr]), sc.transform(M[te])
        pca = PCA(n_components=0.85, svd_solver="full").fit(Atr)
        Ptr, Pte = pca.transform(Atr), pca.transform(Ate)
        a = np.c_[np.ones(len(tr)), Ptr]
        beta = np.linalg.lstsq(a, X[tr], rcond=None)[0]
        ae = np.c_[np.ones(len(te)), Pte]
        return X[tr] - a @ beta, X[te] - ae @ beta
    if control == "crossfit":
        # test fold: residual from a movement model fit on the full train fold (out of sample)
        af = np.c_[np.ones(len(tr)), M[tr]]
        bf = np.linalg.lstsq(af, X[tr], rcond=None)[0]
        Xte_r = X[te] - np.c_[np.ones(len(te)), M[te]] @ bf
        # train fold: inner K-fold cross-fit so each train trial is also out of sample
        Xtr_r = np.empty((len(tr), X.shape[1]), float)
        Mtr = M[tr]
        inner = KFold(n_splits=max(2, min(5, len(tr))), shuffle=True,
                      random_state=int(rng.integers(1 << 31)))
        for itr, ite in inner.split(np.arange(len(tr))):
            a = np.c_[np.ones(len(itr)), Mtr[itr]]
            b = np.linalg.lstsq(a, X[tr][itr], rcond=None)[0]
            Xtr_r[ite] = X[tr][ite] - np.c_[np.ones(len(ite)), Mtr[ite]] @ b
        return Xtr_r, Xte_r
    raise ValueError(f"unknown control: {control}")


def cv_decode(X, y, movement, control, rng, n_repeats=6):
    """Cross-validated choice decode AUC for one block, under one movement control.

    X is [n_trials, n_cells], y is [n_trials] binary. Returns the mean held-out AUC over
    n_repeats stratified k-fold repeats (movement residualized and standardized within fold).
    Returns NaN if the block has too few trials of either class.
    """
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2)
    nsp = min(5, int(counts.min()))
    if nsp < 2:
        return np.nan
    aucs = []
    for _ in range(n_repeats):
        skf = StratifiedKFold(nsp, shuffle=True, random_state=int(rng.integers(1 << 31)))
        yp = np.full(len(y), np.nan)
        for tr, te in skf.split(X, y):
            Xtr, Xte = residualize_fold(X, movement, control, tr, te, rng)
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(C=C_FIXED, max_iter=500, solver="liblinear")
            clf.fit(sc.transform(Xtr), y[tr])
            yp[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
        if np.unique(y).size == 2:
            aucs.append(roc_auc_score(y, yp))
    return float(np.mean(aucs)) if aucs else np.nan

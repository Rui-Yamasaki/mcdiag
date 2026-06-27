"""PART 1 — Over-correction injection control (referee response). PURE; cached data only.

Question: does the published movement-control + decode pipeline remove ONLY confound, or does it
also remove a GENUINE decision signal that is naturally correlated with the developing movement?

Design (reuses the UNCHANGED pipeline in audit_realtrial_decode.py):
  1. Background = real MRN hi-FR cells on decorrelated (0%+error) trials, per co-recorded session.
     The injected choice labels are read off the injected carrier, so the real spike matrix X_bg
     carries NO information about the injected target (verified by the d=0 rows -> chance).
  2. Injected signal: a unit-variance per-trial carrier
         s = rho * m_hat  +  sqrt(1-rho^2) * q
     m_hat = leading movement PC (a real movement axis in span(M_linear), unit var);
     q     = a fresh decision carrier made orthogonal to span(M_linear), unit var.
     Injected choice label y = (s > median(s))  (balanced).  corr(s, movement-projection)=rho.
     Each cell i gets  rate += g_i * s  with g_i = d * sd_i / Delta_s  so the injected PER-CELL
     choice separation (mean(y=1)-mean(y=0))/sd_background = d exactly, for every rho.
     => movement regression removes the rho*m_hat part; because y genuinely depends on m_hat, that
        removal also eats real choice information -> this is precisely the over-correction the
        referee fears, with a KNOWN ground-truth signal.
  3. Run the published per-session CV decoder with movement = none | linear | EXPANDED(nonlinear),
     predicting the true injected y. Also a pooled cell-clustered residual SD effect.
  4. Sweep d in {0,.06,.12,.18,.24,.30} x rho in {0,.3,.6,rho_real}; n_synth repeats -> mean + CI.

Decision number: at d=0.24 (threshold) with the REALISTIC rho, does the movement-controlled
(EXPANDED) decode stay >= 0.57 (pipeline preserves a true movement-correlated signal -> the
paper's null is safe) or get pulled < 0.57 (over-correction -> the "below threshold" claim is
confounded by the control)?

  ./.venv/Scripts/python.exe src/referee_injection_control.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:                                     # live progress when stdout is redirected to a file
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from ibl_one import PROJECT_ROOT
import audit_realtrial_decode as D   # the UNCHANGED published pipeline

R = PROJECT_ROOT / "results"
OUT = R / "referee_response"
OUT.mkdir(parents=True, exist_ok=True)

REGION = "MRN"
D_GRID = [0.0, 0.06, 0.12, 0.18, 0.24, 0.30]
RHO_BASE = [0.0, 0.3, 0.6]
N_SYNTH = 16          # synthetic repeats per (d, rho) for a stable mean + CI
SWEEP_REPS = 6        # internal CV repeats inside the published cv_decode (its default N_REP_OBS)
SEED = 20240613


# ---------- realistic choice<->movement correlation in the real data ----------
def movement_subspace(Mraw):
    """Return standardized linear movement design [n x p] (nan->0), used to define span(M)."""
    M = np.nan_to_num(Mraw.astype(float), nan=0.0)
    M = M - M.mean(0)
    sd = M.std(0); sd[sd < 1e-9] = 1.0
    return M / sd


def proj_onto(M, x):
    """Least-squares projection of x onto [1, M]."""
    A = np.c_[np.ones(len(x)), M]
    beta = np.linalg.lstsq(A, x, rcond=None)[0]
    return A @ beta


def real_choice_movement_corr(sessions):
    """Per-session multiple correlation R = corr(real choice, projection of choice on movement);
    pooled trial-weighted. Also movement->choice cross-val AUC for context."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    Rs, ws, mv_aucs = [], [], []
    for (_, _, y, Mraw) in sessions:
        c = (y.astype(float) - y.mean())
        if c.std() < 1e-9:
            continue
        M = movement_subspace(Mraw)
        fit = proj_onto(M, c)
        r = np.corrcoef(c, fit)[0, 1]
        Rs.append(abs(r)); ws.append(len(y))
        # movement -> choice CV-AUC (how decodable is choice from movement alone)
        nsp = min(5, int(np.bincount(y).min()))
        if nsp >= 2:
            yp = np.full(len(y), np.nan)
            skf = StratifiedKFold(nsp, shuffle=True, random_state=0)
            for tr, te in skf.split(M, y):
                clf = LogisticRegression(C=1.0, max_iter=500).fit(M[tr], y[tr])
                yp[te] = clf.predict_proba(M[te])[:, 1]
            if np.unique(y).size == 2:
                mv_aucs.append(roc_auc_score(y, yp))
    Rs, ws = np.array(Rs), np.array(ws, float)
    pooled_R = float(np.sum(Rs * ws) / ws.sum())
    return pooled_R, float(np.median(Rs)), float(np.mean(mv_aucs))


# ---------- injection ----------
def make_carrier(M, rho, rng):
    """Unit-variance carrier s = rho*m_hat + sqrt(1-rho^2)*q, with m_hat in span(M) (leading
    movement PC) and q orthogonal to span(M). Returns s and realized corr(s, proj_M(s))."""
    n = M.shape[0]
    # m_hat: leading movement PC (real movement axis), unit variance
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    m_hat = U[:, 0] * np.sign(U[0, 0] if U[0, 0] != 0 else 1.0)
    m_hat = (m_hat - m_hat.mean()); m_hat /= (m_hat.std() + 1e-12)
    # q: fresh Gaussian made orthogonal to span(M), unit variance
    g = rng.standard_normal(n)
    q = g - proj_onto(M, g)
    q = q - q.mean(); q /= (q.std() + 1e-12)
    s = rho * m_hat + np.sqrt(max(1 - rho ** 2, 0.0)) * q
    s = s - s.mean(); s /= (s.std() + 1e-12)
    pm = proj_onto(M, s)
    realized = float(np.corrcoef(s, pm)[0, 1]) if pm.std() > 1e-12 else 0.0
    return s, realized


def inject_session(Xbg, Mraw, d, rho, rng):
    """Add a per-cell choice signal of separation d (SD units) with movement-corr rho.
    Returns (X_inj, y_inj, realized_rho)."""
    M = movement_subspace(Mraw)
    s, realized = make_carrier(M, rho, rng)
    y = (s > np.median(s)).astype(int)
    dS = s[y == 1].mean() - s[y == 0].mean()          # separation of the carrier by its own label
    sd = Xbg.std(0); sd[sd < 1e-9] = 1.0
    if d == 0.0 or abs(dS) < 1e-9:
        Xinj = Xbg.copy()
    else:
        g = d * sd / dS                               # per-cell gain -> per-cell separation = d
        Xinj = Xbg + np.outer(s, g)
    return Xinj, y, realized


# ---------- pooled cell-clustered residual SD effect ----------
def pooled_sd(sessions_inj, mode, n_boot=2000, seed=0):
    """Per cell: residualise rate on movement(mode), z-score, mean(z|y=1)-mean(z|y=0) aligned to
    the KNOWN injected direction (all cells share +g*s -> prefer y=1). Pool across cells with a
    cell-clustered bootstrap CI."""
    per = []
    for (Xinj, y, Mraw) in sessions_inj:
        Mdes = D.movement_design(Mraw, mode)
        for i in range(Xinj.shape[1]):
            x = Xinj[:, i].astype(float)
            if Mdes is not None:
                A = np.c_[np.ones(len(x)), Mdes]
                x = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
            if x.std() < 1e-9:
                continue
            z = (x - x.mean()) / x.std()
            if (y == 1).sum() < 2 or (y == 0).sum() < 2:
                continue
            per.append(z[y == 1].mean() - z[y == 0].mean())
    if not per:
        return np.nan, np.nan, np.nan
    per = np.array(per)
    eff = float(per.mean())
    rng = np.random.default_rng(seed)
    boot = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return eff, float(lo), float(hi)


# ---------- one (d, rho) cell ----------
def eval_cell(sessions_bg, d, rho, n_synth, base_seed):
    """Return dict of mean AUC (per mode) + CI across n_synth synthetic repeats, realized rho,
    and pooled SD residual (none + expanded) from the last repeat (for reporting)."""
    aucs = {m: [] for m in ("none", "linear", "expanded")}
    realized = []
    last_inj = None
    for rep in range(n_synth):
        rng = np.random.default_rng(base_seed + 1009 * rep)
        inj = []
        for (eid, Xbg, _, Mraw) in sessions_bg:
            Xinj, y, rl = inject_session(Xbg, Mraw, d, rho, rng)
            inj.append((Xinj, y, Mraw))
            realized.append(rl)
        last_inj = inj
        for mode in ("none", "linear", "expanded"):
            vals = [D.cv_decode(Xinj, y, Mraw, mode, np.random.default_rng(7), SWEEP_REPS)
                    for (Xinj, y, Mraw) in inj]
            aucs[mode].append(np.nanmean(vals))
    out = dict(d=d, rho=rho, realized_rho=float(np.mean(realized)), n_synth=n_synth)
    for mode in ("none", "linear", "expanded"):
        a = np.array(aucs[mode]); m = float(a.mean())
        se = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
        out[f"auc_{mode}"] = m
        out[f"auc_{mode}_lo"] = m - 1.96 * se
        out[f"auc_{mode}_hi"] = m + 1.96 * se
    # pooled SD residual on the last synthetic dataset (uncontrolled + expanded-controlled)
    out["pooled_sd_none"] = pooled_sd(last_inj, "none")[0]
    out["pooled_sd_expanded"], out["pooled_sd_exp_lo"], out["pooled_sd_exp_hi"] = pooled_sd(last_inj, "expanded")
    # attenuation = (controlled-0.5)/(uncontrolled-0.5)
    unc = out["auc_none"] - 0.5
    out["attenuation_expanded"] = float((out["auc_expanded"] - 0.5) / unc) if unc > 1e-6 else np.nan
    out["attenuation_linear"] = float((out["auc_linear"] - 0.5) / unc) if unc > 1e-6 else np.nan
    return out


def main():
    print("=== PART 1: over-correction injection control (region", REGION + ") ===")
    feats = pd.read_csv(R / "phase2_sel_features_full.csv")
    f, dec_fn = D.ibl_prep(feats)
    sub = f[f.region == REGION]
    sessions = []
    for eid, g in sub.groupby("eid"):
        sm = D.session_matrix(g, dec_fn)
        if sm is None:
            continue
        X, y, Mraw = sm
        if D.decodable(X, y):
            sessions.append((eid, X, y, Mraw))
    print(f"  {len(sessions)} decodable MRN sessions "
          f"(mean {np.mean([X.shape[1] for _,X,_,_ in sessions]):.1f} cells x "
          f"{np.mean([len(y) for _,_,y,_ in sessions]):.1f} decorrelated trials)")

    rho_real, rho_med, mv_auc = real_choice_movement_corr(sessions)
    print(f"\n  REAL choice<->movement coupling (decorrelated trials):")
    print(f"    pooled multiple-R(choice, movement projection) = {rho_real:.3f}  (median {rho_med:.3f})")
    print(f"    movement->choice CV-AUC (decodability of choice from movement alone) = {mv_auc:.3f}")
    print(f"  -> REALISTIC rho = {rho_real:.3f}\n")

    rho_grid = RHO_BASE + [round(rho_real, 3)]
    rows = []
    for rho in rho_grid:
        for d in D_GRID:
            r = eval_cell(sessions, d, rho, N_SYNTH, SEED)
            rows.append(r)
            print(f"  rho={rho:5.3f} (realized {r['realized_rho']:.3f}) d={d:.2f} | "
                  f"unctrl {r['auc_none']:.3f}  lin {r['auc_linear']:.3f}  "
                  f"EXP {r['auc_expanded']:.3f} [{r['auc_expanded_lo']:.3f},{r['auc_expanded_hi']:.3f}]  "
                  f"atten(exp) {r['attenuation_expanded']:.2f}  pooledSD(exp) {r['pooled_sd_expanded']:+.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "injection_control.csv", index=False)
    print(f"\nSaved table -> {OUT / 'injection_control.csv'}")

    # ---- decision number ----
    print("\n========== DECISION NUMBER ==========")
    dec = df[(np.isclose(df.d, 0.24)) & (np.isclose(df.rho, round(rho_real, 3)))].iloc[0]
    verdict = ("PRESERVES (>=0.57) -> over-correction objection ANSWERED; the paper's null is safe"
               if dec.auc_expanded >= 0.57 else
               "PULLED < 0.57 -> the pipeline OVER-CORRECTS; 'below threshold' is confounded by the control")
    print(f"  d=0.24 SD, realistic rho={round(rho_real,3)}: movement-controlled (EXPANDED) decode "
          f"AUC = {dec.auc_expanded:.3f}  [{dec.auc_expanded_lo:.3f}, {dec.auc_expanded_hi:.3f}]")
    print(f"  uncontrolled AUC = {dec.auc_none:.3f}; attenuation = {dec.attenuation_expanded:.2f}")
    print(f"  -> {verdict}")

    # d=0 sanity for every rho
    print("\n  d=0 sanity (must be ~chance for every rho):")
    for rho in rho_grid:
        z = df[(np.isclose(df.d, 0.0)) & (np.isclose(df.rho, rho))].iloc[0]
        print(f"    rho={rho:5.3f}: unctrl {z.auc_none:.3f}  lin {z.auc_linear:.3f}  exp {z.auc_expanded:.3f}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(rho_grid)))
    for rho, col in zip(rho_grid, colors):
        sd = df[np.isclose(df.rho, rho)].sort_values("d")
        lbl = f"rho={rho:.3f}" + ("  (REALISTIC)" if np.isclose(rho, round(rho_real, 3)) else "")
        ax.plot(sd.d, sd.auc_expanded, "-o", color=col, label=lbl)
        ax.fill_between(sd.d, sd.auc_expanded_lo, sd.auc_expanded_hi, color=col, alpha=0.12)
    ax.axhline(0.57, color="crimson", ls="--", lw=1.5, label="0.57 recoverability bar")
    ax.axhline(0.50, color="gray", ls=":", lw=1.2, label="0.50 chance")
    ax.axvline(0.24, color="black", ls=":", lw=1.0, alpha=0.6)
    ax.set_xlabel("injected per-cell choice separation  d  (SD units)")
    ax.set_ylabel("movement-controlled (EXPANDED) decode AUC")
    ax.set_title(f"Injection control — {REGION}: does the published control preserve a true,\n"
                 "movement-correlated decision signal?", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "injection_control.png", dpi=150)
    print(f"\nSaved plot -> {OUT / 'injection_control.png'}")


if __name__ == "__main__":
    main()

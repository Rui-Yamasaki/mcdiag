"""Recompute the applied Results under the VALID movement control(s). PURE; cached data only.

Proper-control established: the over-correcting control is `expanded`; the two PASSERS are
`pca_expanded` (frozen primary) and `linear` (second). IMPORTANT NUANCE established by reading the
code: the published per-cell CASCADE (phase2_selectivity.residual_rate) already used LINEAR OLS on
wheel(+DLC) -- a VALID control -- so the over-correction affected the small-n DECODE
(audit_realtrial_decode mode='expanded') only. This recompute regenerates cascade + decode + pooled +
robustness + subpopulation under the valid controls, reports BOTH passers, with `expanded` as contrast.

  ./.venv/Scripts/python.exe -u src/referee_corrected_results.py --stage cascade|decode|robust|subpop|contrast|all
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from ibl_one import PROJECT_ROOT
import audit_realtrial_decode as D
import audit_corrections as AC
import referee_proper_control as PC
import referee_fdr_scope as FD
import steinmetz_replicate as SR

R = PROJECT_ROOT / "results"
OUT = R / "referee_response" / "corrected_results"
OUT.mkdir(parents=True, exist_ok=True)

CONTROLS = ["none", "linear", "expanded", "pca_expanded"]
IBL_REGIONS = ["MRN", "SCm", "SNr", "GRN", "IRN"]
ST_REGIONS = ["MRN", "SC", "SNr"]
PMOVE_CACHE = OUT / "percell_pmove.csv"


# ============================ shared: recompute per-cell p_move under each control ============================
def recompute_pmove(feats, controls):
    rng = np.random.default_rng(0)
    rows = []
    for cell, g in feats.groupby("cell"):
        y = (g["choice"].to_numpy() > 0).astype(int)
        raw = g["rate_delib"].to_numpy(float)
        Mraw = g[D.MOV].to_numpy()
        rec = dict(cell=cell, region=g["region"].iloc[0],
                   fr_window=float(g["fr_window"].iloc[0]), n=len(g))
        for ctl in controls:
            res = raw if ctl == "none" else PC.residualize_full(raw, Mraw, ctl, rng)
            obs, p, *_ = SR.auc_perm(res, y, rng)
            rec[f"auc_move_{ctl}"] = obs
            rec[f"p_move_{ctl}"] = p
        rows.append(rec)
    return pd.DataFrame(rows)


def build_pmove():
    if PMOVE_CACHE.exists():
        return pd.read_csv(PMOVE_CACHE)
    frames = []
    for ds, path in [("IBL", R / "phase2_sel_features_full.csv"),
                     ("Steinmetz", R / "steinmetz_features.csv")]:
        print(f"  recomputing per-cell p_move ({ds}) under {CONTROLS} ...")
        f = pd.read_csv(path)
        df = recompute_pmove(f, CONTROLS); df["dataset"] = ds
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(PMOVE_CACHE, index=False)
    return out


# ============================ Task 1 — corrected cascade ============================
def load_cached_pvals():
    """choice/leading/stim p's are control-INDEPENDENT -> use the published cached values."""
    ic = pd.read_csv(AC.IBL_CELLS)
    ibl = ic[["cell", "region", "p_delib", "p_early", "p_stim"]].copy()
    ibl["p_move_published"] = np.where(ic["has_dlc"], ic["p_move_full"], ic["p_move_wheel"])
    ibl["dataset"] = "IBL"
    st = pd.read_csv(FD.OUT / "steinmetz_cascade_pvals.csv")
    st = st[["cell", "region", "p_delib", "p_early", "p_stim", "p_move"]].rename(
        columns={"p_move": "p_move_published"}); st["dataset"] = "Steinmetz"
    return pd.concat([ibl, st], ignore_index=True)


def cascade_for(c, regions, dataset, control_pcol):
    """Run within-region + dataset-wide cascade for one control's p_move column."""
    cc = c[c.dataset == dataset].copy().reset_index(drop=True)
    cc["p_move"] = cc[control_pcol]
    rows = []
    for scope, sig in [("dataset-wide BH", FD.sig_datasetwide(cc)),
                       ("within-region BH", FD.sig_withinregion(cc))]:
        rows += FD.counts(cc, sig, regions, scope, dataset)
    return pd.DataFrame(rows)


def run_cascade():
    print("=== Task 1: CORRECTED CASCADE (movement-independent under each control) ===")
    print("  NOTE: published cascade used LINEAR wheel(+DLC) -- a VALID control. Anchor = 'published'.")
    pm = build_pmove()
    cached = load_cached_pvals()
    c = cached.merge(pm[["cell"] + [f"p_move_{k}" for k in CONTROLS]], on="cell", how="left")
    control_cols = {"published(linear)": "p_move_published",
                    "linear(recomputed)": "p_move_linear",
                    "pca_expanded(frozen)": "p_move_pca_expanded",
                    "expanded(broken)": "p_move_expanded"}
    allrows = []
    for label, pcol in control_cols.items():
        for dataset, regions in [("IBL", IBL_REGIONS), ("Steinmetz", ST_REGIONS)]:
            t = cascade_for(c, regions, dataset, pcol); t["control"] = label
            allrows.append(t)
    df = pd.concat(allrows, ignore_index=True)
    df.to_csv(OUT / "corrected_cascade.csv", index=False)

    # report: movement-independent + triple, per region, both scopes, all controls
    for dataset in ("IBL", "Steinmetz"):
        print(f"\n----- {dataset}: movement-independent -> triple-coded (raw counts already FDR'd) -----")
        for scope in ("dataset-wide BH", "within-region BH"):
            print(f"  [{scope}]  control: move-indep ALL / triple ALL  (per-region triple where >0)")
            for label in control_cols:
                sub = df[(df.dataset == dataset) & (df.scheme == scope) & (df.control == label)]
                allr = sub[sub.region == "ALL"].iloc[0]
                tri = sub[(sub.region != "ALL") & (sub.triple_fdr > 0)]
                triwhere = ", ".join(f"{r.region}={r.triple_fdr}" for _, r in tri.iterrows()) or "none"
                print(f"    {label:22s}: move-indep {int(allr.move_indep_fdr):3d}  "
                      f"triple {int(allr.triple_fdr):2d}  [{triwhere}]")
    print(f"\nSaved -> {OUT/'corrected_cascade.csv'}")
    return df


# ============================ Task 1b — raw move-survive contrast (the control-disagreement result) ============================
def run_movecontrast():
    print("=== Task 1b/5: raw movement-independent survivors by control (the disagreement) ===")
    pm = build_pmove()
    ic = pd.read_csv(AC.IBL_CELLS)
    pub = pd.Series(np.where(ic["has_dlc"], ic["p_move_full"], ic["p_move_wheel"]), index=ic["cell"])
    rows = []
    for dataset, regions, cellsrc in [("IBL", IBL_REGIONS, ic.set_index("cell")),
                                      ("Steinmetz", ST_REGIONS,
                                       pd.read_csv(FD.OUT / "steinmetz_cascade_pvals.csv").set_index("cell"))]:
        sub = pm[pm.dataset == dataset].set_index("cell")
        sig = (cellsrc["p_delib"].reindex(sub.index) < 0.05)
        for reg in regions + ["ALL"]:
            m = sig.index.isin(sub.index) & ((sub["region"] == reg) if reg != "ALL" else True) & sig.values
            r = dict(dataset=dataset, region=reg, n_choice_sig=int(m.sum()))
            for ctl in ("linear", "expanded", "pca_expanded"):
                r[f"movesurv_{ctl}"] = int((m & (sub[f"p_move_{ctl}"] < 0.05)).values.sum())
            if dataset == "IBL":
                r["movesurv_published"] = int((m & (pub.reindex(sub.index) < 0.05)).values.sum())
            rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(OUT / "overcorrection_contrast.csv", index=False)
    print(df.to_string(index=False))
    al = df[(df.dataset == "IBL") & (df.region == "ALL")].iloc[0]
    print(f"\n  IBL choice-sig cells whose choice survives movement control (RAW):")
    print(f"    linear(valid)={al.movesurv_linear}  pca_expanded(under-removes)={al.movesurv_pca_expanded}  "
          f"expanded(over-removes)={al.movesurv_expanded}")
    print(f"  -> expanded ERASES {al.movesurv_linear - al.movesurv_expanded} of {al.movesurv_linear} "
          f"({(al.movesurv_linear-al.movesurv_expanded)/al.movesurv_linear:.0%}) linear-valid move-indep cells;")
    print(f"     pca under-removes (+{al.movesurv_pca_expanded - al.movesurv_linear}) -> its extra survivors are "
          f"residual low-variance movement, NOT new signal.")
    print(f"\nSaved -> {OUT/'overcorrection_contrast.csv'}")
    return df


# ============================ Task 2 — per-region decode + pooled SD ============================
def run_decode():
    print("=== Task 2: corrected per-region decode + pooled SD (vs 0.57 / 0.24 bars) ===")
    blocks = [("IBL", r, "err") for r in IBL_REGIONS] + [("Steinmetz", r, "equal") for r in ST_REGIONS]
    feat_cache = {}
    rows = []
    for dataset, region, trial in blocks:
        sessions = PC.load_sessions(dataset, region)
        if dataset not in feat_cache:
            path = R / ("phase2_sel_features_full.csv" if dataset == "IBL" else "steinmetz_features.csv")
            feat_cache[dataset] = pd.read_csv(path)
        feats = feat_cache[dataset]
        print(f"\n  {dataset} {region}: {len(sessions)} decodable sessions")
        for ctl in CONTROLS:
            if len(sessions) >= 2:
                dec = PC.decode_real(sessions, ctl, np.random.default_rng(0),
                                     n_perm=120 if ctl in ("linear", "pca_expanded") else 60)
            else:
                dec = dict(auc=np.nan, lo=np.nan, hi=np.nan, perm_p=np.nan, n_sessions=len(sessions))
            pe = PC.pooled_effect_controlled(feats, region, ctl, trial)
            def flag(lo, hi, bar):
                if np.isnan(lo): return "n/a"
                return "CLEARS" if lo > bar else ("BELOW" if hi < bar else "straddles")
            rows.append(dict(dataset=dataset, region=region, control=ctl, n_sessions=dec["n_sessions"],
                             decode_auc=dec["auc"], decode_lo=dec["lo"], decode_hi=dec["hi"],
                             decode_p=dec["perm_p"], decode_bar=flag(dec["lo"], dec["hi"], 0.57),
                             pooled_sd=pe["eff"], pooled_lo=pe["ci_lo"], pooled_hi=pe["ci_hi"],
                             pooled_n=pe["n_cells"], pooled_bar=flag(pe["ci_lo"], pe["ci_hi"], 0.24)))
            print(f"    {ctl:13s}: decode {dec['auc']:.3f} [{dec['lo']:.3f},{dec['hi']:.3f}] "
                  f"({flag(dec['lo'],dec['hi'],0.57)} 0.57) | pooled {pe['eff']:+.3f} "
                  f"[{pe['ci_lo']:+.3f},{pe['ci_hi']:+.3f}] ({flag(pe['ci_lo'],pe['ci_hi'],0.24)} 0.24) "
                  f"n={pe['n_cells']}")
    pd.DataFrame(rows).to_csv(OUT / "corrected_decode_by_region.csv", index=False)
    print(f"\nSaved -> {OUT/'corrected_decode_by_region.csv'}")


# ============================ Task 4 — movement-independent subpopulation ============================
def run_subpop():
    print("=== Task 4: movement-independent choice subpopulation (valid controls) ===")
    pm = build_pmove()
    rows = []; exemplars = []
    for dataset, regions, path in [("IBL", IBL_REGIONS, R / "phase2_sel_features_full.csv"),
                                   ("Steinmetz", ST_REGIONS, R / "steinmetz_features.csv")]:
        cellsrc = (pd.read_csv(AC.IBL_CELLS).set_index("cell") if dataset == "IBL"
                   else pd.read_csv(FD.OUT / "steinmetz_cascade_pvals.csv").set_index("cell"))
        sub = pm[pm.dataset == dataset].set_index("cell")
        sub = sub.join(cellsrc[["p_delib"]], how="left")
        sub["choice_sig"] = sub["p_delib"] < 0.05
        for reg in regions:
            u = sub[sub.region == reg]
            n = len(u)
            for ctl in ("linear", "pca_expanded"):
                # within-region FDR on p_move among the region cells, require choice-sig
                from scipy.stats import false_discovery_control
                q = np.full(n, np.nan); pv = u[f"p_move_{ctl}"].to_numpy()
                ok = np.isfinite(pv)
                if ok.any():
                    q[ok] = false_discovery_control(pv[ok], method="bh")
                mi = (u["choice_sig"].to_numpy() & (q < 0.05))
                tail = (u[f"auc_move_{ctl}"] > 0.65).mean()
                rows.append(dict(dataset=dataset, region=reg, control=ctl, n_cells=n,
                                 move_indep_FDR=int(mi.sum()), frac_move_indep=mi.sum() / max(n, 1),
                                 median_auc_after=float(u[f"auc_move_{ctl}"].median()),
                                 frac_auc_after_gt065=float(tail)))
            # exemplars: choice-sig cells with high post-control AUC under BOTH valid controls
            ex = u[(u["choice_sig"]) & (u["auc_move_linear"] >= 0.70) & (u["auc_move_pca_expanded"] >= 0.70)]
            for cell, r in ex.iterrows():
                exemplars.append(dict(dataset=dataset, region=reg, cell=cell, n=int(r["n"]),
                                      fr_window=round(float(r["fr_window"]), 1),
                                      auc_linear=round(float(r["auc_move_linear"]), 3),
                                      auc_pca=round(float(r["auc_move_pca_expanded"]), 3),
                                      p_move_linear=round(float(r["p_move_linear"]), 4),
                                      p_move_pca=round(float(r["p_move_pca_expanded"]), 4)))
    df = pd.DataFrame(rows); df.to_csv(OUT / "selective_subpopulation.csv", index=False)
    ex = pd.DataFrame(exemplars).sort_values(["dataset", "auc_linear"], ascending=[True, False])
    ex.to_csv(OUT / "exemplar_cells.csv", index=False)
    print("  per-region movement-independent (within-region FDR) under valid controls:")
    for dataset in ("IBL", "Steinmetz"):
        d = df[df.dataset == dataset]
        for reg in (IBL_REGIONS if dataset == "IBL" else ST_REGIONS):
            rl = d[(d.region == reg) & (d.control == "linear")].iloc[0]
            rp = d[(d.region == reg) & (d.control == "pca_expanded")].iloc[0]
            print(f"    {dataset} {reg:4s} (n={int(rl.n_cells)}): move-indep linear={int(rl.move_indep_FDR)} "
                  f"pca={int(rp.move_indep_FDR)} | tail(AUC>0.65) linear={rl.frac_auc_after_gt065:.2f}")
    print(f"\n  exemplar movement-independent choice cells (AUC>=0.70 post-control under BOTH valid controls): "
          f"{len(ex)}")
    print(ex.to_string(index=False))
    print(f"\nSaved -> {OUT/'selective_subpopulation.csv'}, {OUT/'exemplar_cells.csv'}")


# ============================ Task 3 — corrected robustness sweep (MRN) ============================
def sessions_from(feats, region, rate_col, fr_floor):
    """Build decodable MRN sessions from a feature df, using rate_col as the rate + a FR floor."""
    f = feats.copy()
    f = f[f["fr_window"] >= fr_floor]
    f["chose"] = -np.sign(f["choice"]); f["stim"] = np.sign(f["signed"])
    f["dec_is0"] = f["absc"] == 0
    f["dec_err"] = (f["absc"] > 0) & (f["chose"] != f["stim"])
    dec_fn = lambda t: t["dec_is0"] | t["dec_err"]
    sub = f[f.region == region]
    out = []
    for eid, g in sub.groupby("eid"):
        cells = list(g["cell"].unique())
        t0 = g[g["cell"] == cells[0]].reset_index(drop=True)
        dec = dec_fn(t0).to_numpy(); nT = len(t0)
        rate = {}
        ok = True
        for c in cells:
            v = g[g["cell"] == c][rate_col].to_numpy()
            if len(v) != nT:
                ok = False; break
            rate[c] = v
        if not ok:
            continue
        X = np.column_stack([rate[c][dec] for c in cells])
        y = (t0["chose"].to_numpy()[dec] > 0).astype(int)
        Mraw = t0[D.MOV].to_numpy()[dec]
        if X.shape[1] >= D.MIN_CELLS and (y == 0).sum() >= D.MIN_SIDE and (y == 1).sum() >= D.MIN_SIDE:
            out.append((eid, X, y, Mraw))
    return out


def pooled_rc(feats, region, control, rate_col):
    """pooled_effect_controlled but on an arbitrary rate column (for window/bin variants)."""
    f = feats[feats.region == region].copy()
    f["chose"] = -np.sign(f["choice"]); f["stim"] = np.sign(f["signed"])
    f["cor"] = (f["absc"] > 0) & (f["chose"] == f["stim"])
    f["dec"] = (f["absc"] > 0) & (f["chose"] != f["stim"])
    rng = np.random.default_rng(0); per = []
    for cell, g in f.groupby("cell"):
        cor, dec, chose = g["cor"].to_numpy(), g["dec"].to_numpy(), g["chose"].to_numpy()
        cR, cL = cor & (chose > 0), cor & (chose < 0)
        if cR.sum() < 3 or cL.sum() < 3 or dec.sum() < 6:
            continue
        rate = PC.residualize_full(g[rate_col].to_numpy(float), g[D.MOV].to_numpy(), control, rng)
        pref = np.sign(rate[cR].mean() - rate[cL].mean())
        er = rate[dec]; ez = (er - er.mean()) / (er.std() + 1e-9); cong = chose[dec] == pref
        if cong.sum() < 2 or (~cong).sum() < 2:
            continue
        per.append((ez[cong], ez[~cong]))
    if not per:
        return np.nan, np.nan, np.nan
    zc = np.concatenate([p[0] for p in per]); zi = np.concatenate([p[1] for p in per])
    eff = zc.mean() - zi.mean()
    boot = [np.concatenate([per[i][0] for i in rng.choice(len(per), len(per))]).mean()
            - np.concatenate([per[i][1] for i in rng.choice(len(per), len(per))]).mean() for _ in range(2000)]
    return float(eff), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run_robust():
    print("=== Task 3: corrected robustness sweep (MRN), valid controls vs the bar ===")
    print("  (the OLD robustness table used the broken EXPANDED control -> everything at chance)")
    main = pd.read_csv(R / "phase2_sel_features_full.csv")
    settings = []
    for fl in (10, 15, 25):
        settings.append((f"floor{fl}Hz", main, "rate_delib", fl))
    for win in ("early", "delib", "peri"):
        settings.append((f"window-{win}", main, f"rate_{win}", 10))
    for b in (10, 20, 50):
        settings.append((f"bin{b}ms", pd.read_csv(R / f"robustness/binsize_feat_b{b}.csv"), "rate_delib", 10))
    rows = []
    for name, feats, rcol, fl in settings:
        sess = sessions_from(feats, "MRN", rcol, fl)
        for ctl in ("linear", "pca_expanded"):
            if len(sess) >= 2:
                obs = np.array([PC.cv_decode_ctrl(X, y, Mraw, ctl, np.random.default_rng(7), 6)
                                for (_, X, y, Mraw) in sess])
                comb = float(np.nanmean(obs))
                brng = np.random.default_rng(0)
                boot = [np.nanmean(obs[brng.integers(0, len(obs), len(obs))]) for _ in range(2000)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
            else:
                comb = lo = hi = np.nan
            pe, plo, phi = pooled_rc(feats, "MRN", ctl, rcol)
            barflag = "n/a" if np.isnan(lo) else ("CLEARS" if lo > 0.57 else "BELOW" if hi < 0.57 else "straddles")
            rows.append(dict(setting=name, control=ctl, n_sessions=len(sess), decode_auc=comb,
                             decode_lo=lo, decode_hi=hi, decode_bar=barflag,
                             pooled_sd=pe, pooled_lo=plo, pooled_hi=phi))
            print(f"  {name:14s} {ctl:13s}: decode {comb:.3f} [{lo:.3f},{hi:.3f}] ({barflag} 0.57) | "
                  f"pooled {pe:+.3f} [{plo:+.3f},{phi:+.3f}]")
    pd.DataFrame(rows).to_csv(OUT / "corrected_robustness.csv", index=False)
    print(f"\nSaved -> {OUT/'corrected_robustness.csv'}")


# ============================ Task 5 — figure ============================
def run_figure():
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    oc = pd.read_csv(OUT / "overcorrection_contrast.csv")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ibl = oc[(oc.dataset == "IBL") & (oc.region != "ALL")]
    regions = list(ibl.region)
    x = np.arange(len(regions)); w = 0.27
    ax.bar(x - w, ibl.movesurv_expanded, w, color="#c77", label="expanded (broken, over-removes)")
    ax.bar(x, ibl.movesurv_linear, w, color="#6aa", label="linear (valid, = published)")
    ax.bar(x + w, ibl.movesurv_pca_expanded, w, color="#27a", label="pca_expanded (frozen, under-removes)")
    ax.set_xticks(x); ax.set_xticklabels(regions)
    ax.set_ylabel("raw movement-independent choice cells")
    ax.set_xlabel("IBL region")
    ax.set_title("Movement-independent survivors by control — over-correction (expanded) halves the\n"
                 "valid-control count; pca under-removes (residual low-variance movement)", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "cascade_by_control.png", dpi=150)
    print(f"Saved -> {OUT/'cascade_by_control.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="cascade")
    a = ap.parse_args()
    if a.stage in ("cascade", "all"):
        run_cascade()
    if a.stage in ("movecontrast", "all"):
        run_movecontrast()
    if a.stage in ("subpop", "all"):
        run_subpop()
    if a.stage in ("decode", "all"):
        run_decode()
    if a.stage in ("robust", "all"):
        run_robust()
    if a.stage in ("figure", "all"):
        run_figure()

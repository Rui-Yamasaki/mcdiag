"""Phase 2 / step 2 - PROPER movement-controlled choice-selectivity (TRIMMED spikes).

Replaces the Phase-2 census heuristic (single-window AUC, chance-SD subtraction)
with a defensible cascade, and confronts the movement confound (choice direction
is confounded with wheel-turn direction) for the first time:

  RAW           high-FR (>=10 Hz in-window) good units in {GRN,MRN,IRN,SNr,SCm}
  PERM-SIG      choice ROC-AUC in the deliberative window [stimOn, firstMovement],
                shuffle null (>=1000 perms) -> p, Benjamini-Hochberg FDR across cells
  MOVE-SURVIVE  residual choice AUC after regressing out movement (wheel speed/disp,
                DLC paw speed if present) AND nuisance (signed contrast, block pL);
                still FDR-significant
  LEADING       choice AUC in an EARLY window [stimOn+0.1, firstMovement-0.2]
                (>=200 ms before movement) FDR-significant -> decision-like, leads
                movement.  LOCKED = peri-movement [fm-0.1, fm+0.1] sig but not early.
  TRIPLE        PERM-SIG & MOVE-SURVIVE & LEADING = genuinely decision-related pop.

Note the deliberative window is pre-movement BY CONSTRUCTION (firstMovement is the
first wheel motion), so the wheel regressor is weak inside it and the lead/lock
timing is the primary movement control; the regression additionally removes any
sub-threshold / DLC movement and the stimulus/prior nuisances.

No model fitting. No git. Stages:
    python src/phase2_selectivity.py --stage features --cap 12   # heavy I/O (background)
    python src/phase2_selectivity.py --stage analyze             # perm + FDR + cascade
    python src/phase2_selectivity.py --stage report              # figure + verdict
    python src/phase2_selectivity.py --stage test-one            # probe one insertion
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, false_discovery_control  # noqa: E402

from brainbox.io.one import SessionLoader, SpikeSortingLoader  # noqa: E402
from iblatlas.atlas import BrainRegions  # noqa: E402

from ibl_one import BWM_PROJECT, PROJECT_ROOT, get_one  # noqa: E402
from phase2_census import (CORE_REGIONS, GOOD_LABEL, MAX_ABS_CONTRAST, RT_HI,  # noqa: E402
                           RT_LO, build_sample)

warnings.filterwarnings("ignore", message="Multiple revisions")

FR_HI = 10.0                 # in-window high-FR threshold (recoverable lane)
MIN_PER_SIDE = 8
N_PERM = 2000
FDR_Q = 0.05
EARLY = (0.10, 0.20)         # early window = [stimOn+0.10, firstMovement-0.20]
PERI = (0.10, 0.10)          # peri window  = [fm-0.10, fm+0.10]

FEATURES_CSV = PROJECT_ROOT / "results" / "phase2_sel_features.csv"
CELLS_CSV = PROJECT_ROOT / "results" / "phase2_sel_cells.csv"
REGION_CSV = PROJECT_ROOT / "results" / "phase2_sel_region.csv"
FIG_PATH = PROJECT_ROOT / "figures" / "phase2_selectivity_controlled.png"


# --- rich deliberative-trial table -------------------------------------------
def delib_trials(trials):
    on = trials["stimOn_times"].to_numpy(float)
    fm = trials["firstMovement_times"].to_numpy(float)
    rsp = trials["response_times"].to_numpy(float)
    ch = trials["choice"].to_numpy(float)
    cl = np.nan_to_num(trials["contrastLeft"].to_numpy(float))
    cr = np.nan_to_num(trials["contrastRight"].to_numpy(float))
    pL = trials["probabilityLeft"].to_numpy(float)
    absc = np.fmax(cl, cr) * 100.0
    rt = fm - on
    m = ((ch != 0) & np.isfinite(on) & np.isfinite(fm) & np.isfinite(rsp)
         & (rt >= RT_LO) & (rt <= RT_HI) & (absc <= MAX_ABS_CONTRAST))
    return pd.DataFrame(dict(on=on[m], fm=fm[m], choice=ch[m], absc=absc[m],
                             signed=(cr - cl)[m], pL=pL[m]))


# --- per-trial movement features (cell-independent) --------------------------
def wheel_features(one, eid, tr):
    """Mean wheel speed and net displacement in [on, fm] per trial."""
    try:
        w = one.load_object(eid, "wheel", collection="alf")
        ts, pos = w["timestamps"], w["position"]
        vel = np.gradient(pos, ts)
        speed = np.empty(len(tr)); disp = np.empty(len(tr))
        for i, (a, b) in enumerate(zip(tr["on"], tr["fm"])):
            sl = slice(np.searchsorted(ts, a), np.searchsorted(ts, b))
            speed[i] = np.mean(np.abs(vel[sl])) if sl.stop > sl.start else 0.0
            disp[i] = np.interp(b, ts, pos) - np.interp(a, ts, pos)
        return speed, disp
    except Exception:
        return np.full(len(tr), np.nan), np.full(len(tr), np.nan)


def _marker_speed_in_windows(x, y, lik, ct, tr):
    x = x.copy(); y = y.copy()
    x[lik < 0.9] = np.nan; y[lik < 0.9] = np.nan
    sp = np.sqrt(np.gradient(x, ct) ** 2 + np.gradient(y, ct) ** 2)
    out = np.full(len(tr), np.nan)
    for i, (a, b) in enumerate(zip(tr["on"], tr["fm"])):
        sl = slice(np.searchsorted(ct, a), np.searchsorted(ct, b))
        if sl.stop > sl.start:
            out[i] = np.nanmean(sp[sl])
    return out


def dlc_speeds(one, eid, tr):
    """Mean right-paw and nose-tip speed in [on, fm] per trial from left-camera DLC
    (best effort; DLC exists for only some sessions). Returns (paw_speed, nose_speed)."""
    try:
        cam = one.load_object(eid, "leftCamera", collection="alf",
                              attribute=["dlc", "times"])
        dlc, ct = cam["dlc"], cam["times"]
        paw = _marker_speed_in_windows(dlc["paw_r_x"].to_numpy(), dlc["paw_r_y"].to_numpy(),
                                       dlc["paw_r_likelihood"].to_numpy(), ct, tr)
        nose = (_marker_speed_in_windows(dlc["nose_tip_x"].to_numpy(),
                                         dlc["nose_tip_y"].to_numpy(),
                                         dlc["nose_tip_likelihood"].to_numpy(), ct, tr)
                if "nose_tip_x" in dlc.columns else np.full(len(tr), np.nan))
        return paw, nose
    except Exception:
        return np.full(len(tr), np.nan), np.full(len(tr), np.nan)


def _rate(st_sorted, a, b):
    return (np.searchsorted(st_sorted, b) - np.searchsorted(st_sorted, a)) / (b - a)


def process_features(one, br, pid, cleanup=True, use_dlc=False):
    ssl = SpikeSortingLoader(pid=pid, one=one)
    eid, pname = ssl.eid, ssl.pname
    coll = f"alf/{pname}/pykilosort"
    sl = SessionLoader(one=one, eid=eid); sl.load_trials()
    tr = delib_trials(sl.trials)
    if len(tr) < 2 * MIN_PER_SIDE or (tr["choice"] < 0).sum() < MIN_PER_SIDE \
            or (tr["choice"] > 0).sum() < MIN_PER_SIDE:
        return None, dict(pid=pid, eid=str(eid), n=len(tr), skip="few_trials")

    clusters = one.load_object(eid, "clusters", collection=coll)
    channels = one.load_object(eid, "channels", collection=coll)
    spikes = one.load_object(eid, "spikes", collection=coll,
                             attribute=["times", "clusters"])
    st, sc = spikes["times"], spikes["clusters"].astype(int)
    ch_beryl = np.asarray(br.id2acronym(channels["brainLocationIds_ccf_2017"],
                                        mapping="Beryl"))
    clu_region = ch_beryl[clusters["channels"].astype(int)]
    label = clusters["metrics"]["label"].to_numpy(float)
    Trec = float(st[-1])

    wsp, wdisp = wheel_features(one, eid, tr)
    # DLC body-movement (paw + nose) is opt-in: available for only some sessions,
    # 12-39 MB each. Wheel is the direct choice-confound; DLC adds body-movement.
    if use_dlc:
        paw, nose = dlc_speeds(one, eid, tr)
    else:
        paw = nose = np.full(len(tr), np.nan)
    has_dlc = bool(np.isfinite(paw).any() or np.isfinite(nose).any())

    rows = []
    target = set(CORE_REGIONS)
    for u in range(len(clusters["channels"])):
        if clu_region[u] not in target or label[u] < GOOD_LABEL:
            continue
        ut = st[sc == u]
        if ut.size == 0:
            continue
        fr_overall = ut.size / Trec
        # per-trial rates in three windows
        rd = np.array([_rate(ut, a, b) for a, b in zip(tr["on"], tr["fm"])])
        if rd.mean() < FR_HI:                       # high-FR (in-window) only
            continue
        re = np.array([_rate(ut, a + EARLY[0], b - EARLY[1])
                       if (b - EARLY[1]) > (a + EARLY[0]) else np.nan
                       for a, b in zip(tr["on"], tr["fm"])])
        rp = np.array([_rate(ut, b - PERI[0], b + PERI[1]) for b in tr["fm"]])
        for i in range(len(tr)):
            rows.append(dict(cell=f"{pid}:{u}", region=clu_region[u], eid=str(eid),
                             fr_overall=fr_overall, fr_window=float(rd.mean()),
                             choice=int(tr["choice"].iloc[i]),
                             absc=float(tr["absc"].iloc[i]),
                             signed=float(tr["signed"].iloc[i]),
                             pL=float(tr["pL"].iloc[i]),
                             rate_delib=float(rd[i]), rate_early=float(re[i]),
                             rate_peri=float(rp[i]), wheel_speed=float(wsp[i]),
                             wheel_disp=float(wdisp[i]), paw_speed=float(paw[i]),
                             nose_speed=float(nose[i])))
    if cleanup:
        spath = Path(one.eid2path(eid))
        for pat in (f"alf/{pname}/pykilosort/**/spikes.times.npy",
                    f"alf/{pname}/pykilosort/**/spikes.clusters.npy",
                    "alf/**/_ibl_leftCamera.dlc.pqt",
                    "alf/**/_ibl_leftCamera.times.npy"):
            for f in spath.glob(pat):
                try:
                    f.unlink()
                except OSError:
                    pass
    df = pd.DataFrame(rows)
    meta = dict(pid=pid, eid=str(eid), n_trials=len(tr),
                n_hiFR_cells=df["cell"].nunique() if len(df) else 0,
                has_dlc=bool(has_dlc),
                wheel_ok=bool(np.isfinite(wsp).any()))
    return df, meta


# --- permutation choice AUC ---------------------------------------------------
def auc_perm(x, y, rng, n_perm=N_PERM):
    """Two-sided permutation p for choice ROC-AUC of feature x vs labels y(0/1)."""
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    n = len(y); n1 = int(y.sum()); n0 = n - n1
    if n1 < MIN_PER_SIDE or n0 < MIN_PER_SIDE:
        return np.nan, np.nan, n1, n0
    r = rankdata(x)
    obs = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    pos = np.argsort(rng.random((n_perm, n)), axis=1)[:, :n1]   # random pos-class sets
    null = (r[pos].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    p = (np.sum(np.abs(null - 0.5) >= abs(obs - 0.5)) + 1) / (n_perm + 1)
    return float(obs), float(p), n1, n0


def residual_rate(g, cols):
    """Regress the named per-trial covariates out of rate_delib (linear)."""
    M, used = [np.ones(len(g))], []
    for c in cols:
        v = g[c].to_numpy(float)
        if np.isfinite(v).sum() >= MIN_PER_SIDE and np.nanstd(v) > 0:
            v = np.where(np.isfinite(v), v, np.nanmean(v))
            M.append(v); used.append(c)
    M = np.column_stack(M)
    y = g["rate_delib"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    return y - M @ beta


def run_analyze():
    """Permutation choice-AUC in each window + SEPARATE movement and stimulus/prior
    controls. FDR is applied to the primary deliberative-window test; the movement
    (wheel) survival, stimulus/prior survival, and lead/lock split are conditional
    classifications among already choice-responsive cells (raw permutation p)."""
    feats = pd.read_csv(FEATURES_CSV)
    rng = np.random.default_rng(0)
    rec = []
    dlc_cols = [c for c in ("paw_speed", "nose_speed") if c in feats.columns]
    for cell, g in feats.groupby("cell"):
        y = (g["choice"].to_numpy() > 0).astype(int)
        a_d, p_d, n0, n1 = auc_perm(g["rate_delib"].to_numpy(float), y, rng)
        a_e, p_e, *_ = auc_perm(g["rate_early"].to_numpy(float), y, rng)
        a_p, p_p, *_ = auc_perm(g["rate_peri"].to_numpy(float), y, rng)
        # Task 3: MOVEMENT control. wheel-only AND wheel+DLC (where DLC present).
        has_dlc = any(np.isfinite(g[dc].to_numpy(float)).sum() >= MIN_PER_SIDE
                      for dc in dlc_cols)
        a_mw, p_mw, *_ = auc_perm(
            residual_rate(g, ["wheel_speed", "wheel_disp"]), y, rng)
        if has_dlc:
            a_mf, p_mf, *_ = auc_perm(
                residual_rate(g, ["wheel_speed", "wheel_disp"] + dlc_cols), y, rng)
        else:
            a_mf, p_mf = a_mw, p_mw
        # Task 4: STIMULUS/PRIOR control = signed contrast + block pL (SEPARATE)
        a_st, p_st, *_ = auc_perm(residual_rate(g, ["signed", "pL"]), y, rng)
        rec.append(dict(cell=cell, region=g["region"].iloc[0],
                        eid=g["eid"].iloc[0], n_left=n0, n_right=n1,
                        fr_window=g["fr_window"].iloc[0], has_dlc=has_dlc,
                        auc_delib=a_d, p_delib=p_d, auc_early=a_e, p_early=p_e,
                        auc_peri=a_p, p_peri=p_p,
                        p_move_wheel=p_mw, p_move_full=p_mf, p_stim=p_st))
    c = pd.DataFrame(rec)
    m = c["p_delib"].notna()
    c.loc[m, "q_delib"] = false_discovery_control(c.loc[m, "p_delib"].to_numpy(),
                                                  method="bh")
    # primary corrected significance (Task 1)
    c["sig_fdr"] = c["q_delib"] < FDR_Q
    c["sig_raw"] = c["p_delib"] < 0.05
    # best-available movement control: wheel+DLC where DLC present, else wheel-only
    c["p_move"] = np.where(c["has_dlc"], c["p_move_full"], c["p_move_wheel"])
    # conditional classifications among choice-responsive (raw-sig) cells
    c["move_survive"] = c["sig_raw"] & (c["p_move"] < 0.05)            # Task 3 (full)
    c["move_survive_wheel"] = c["sig_raw"] & (c["p_move_wheel"] < 0.05)
    c["stim_survive"] = c["sig_raw"] & (c["p_stim"] < 0.05)           # Task 4
    c["leading"] = c["sig_raw"] & (c["p_early"] < 0.05)              # Task 2: pre-movement
    c["locked"] = c["sig_raw"] & (~(c["p_early"] < 0.05)) & (c["p_peri"] < 0.05)
    # genuinely decision-related: choice-sig, leads movement, survives movement+stim
    c["triple"] = c["sig_raw"] & c["leading"] & c["move_survive"] & c["stim_survive"]
    CELLS_CSV.parent.mkdir(parents=True, exist_ok=True)
    c.to_csv(CELLS_CSV, index=False)

    nn = int(m.sum()); chance = round(0.05 * nn)
    print(f"Analyzed {len(c)} high-FR cells (perms={N_PERM}).")
    print(f"  deliberative choice: raw p<0.05 = {int(c['sig_raw'].sum())} "
          f"(chance ~{chance}) -> excess ~{int(c['sig_raw'].sum())-chance}; "
          f"FDR q<{FDR_Q} = {int(c['sig_fdr'].sum())}")
    print(f"  WINDOW (raw p<0.05): early/pre-move {int((c['p_early']<0.05).sum())} "
          f"vs peri-move {int((c['p_peri']<0.05).sum())}  (chance ~{chance}) "
          f"-> signal is {'MOVEMENT-LOCKED' if (c['p_peri']<0.05).sum() > 1.5*(c['p_early']<0.05).sum() else 'mixed'}")
    print(f"  among raw-sig choice cells (n={int(c['sig_raw'].sum())}): "
          f"survive wheel-regression {int(c['move_survive'].sum())} | "
          f"survive contrast/prior {int(c['stim_survive'].sum())} | "
          f"leading {int(c['leading'].sum())} | locked {int(c['locked'].sum())}")
    print(f"  TRIPLE (choice & leading & movement-surviving & contrast-surviving): "
          f"{int(c['triple'].sum())}")
    # DLC coverage + wheel-only vs wheel+DLC comparison (Task 2)
    ndlc = int(c["has_dlc"].sum())
    print(f"  DLC body-movement available: {ndlc}/{len(c)} cells "
          f"({ndlc/len(c):.0%}).")
    sub = c[c["sig_raw"] & c["has_dlc"]]
    if len(sub):
        print(f"  DLC-subset choice-sig cells (n={len(sub)}): survive WHEEL-only "
              f"{int((sub['p_move_wheel']<0.05).sum())} vs survive WHEEL+BODY(DLC) "
              f"{int((sub['p_move_full']<0.05).sum())} -> body-movement removes "
              f"{int((sub['p_move_wheel']<0.05).sum())-int((sub['p_move_full']<0.05).sum())} more")
    return c


def run_report():
    c = pd.read_csv(CELLS_CSV)
    rows = []
    for reg in CORE_REGIONS:
        u = c[c["region"] == reg]
        lead = u[u["leading"]]; tri = u[u["triple"]]
        rows.append(dict(region=reg, raw_hiFR=len(u),
                         choice_raw=int(u["sig_raw"].sum()),
                         choice_fdr=int(u["sig_fdr"].sum()),
                         leading=int(u["leading"].sum()),
                         locked=int(u["locked"].sum()),
                         move_surv_wheel=int(u["move_survive_wheel"].sum()),
                         move_surv=int(u["move_survive"].sum()),
                         stim_surv=int(u["stim_survive"].sum()),
                         triple=int(u["triple"].sum()),
                         dlc_cells=int(u["has_dlc"].sum()),
                         sessions=u["eid"].nunique(),
                         pooledN_leading=int((lead["n_left"] + lead["n_right"]).sum()),
                         pooledN_triple=int((tri["n_left"] + tri["n_right"]).sum())))
    rt = pd.DataFrame(rows)
    REGION_CSV.parent.mkdir(parents=True, exist_ok=True)
    rt.to_csv(REGION_CSV, index=False)
    print("\n========== per-region filter cascade ==========")
    print(rt.to_string(index=False))
    make_figure(c, rt)

    tot = rt[["raw_hiFR", "choice_raw", "choice_fdr", "leading", "locked",
              "move_surv", "stim_surv", "triple"]].sum()
    print("\n========== GLANCEABLE VERDICT ==========")
    print(f"  raw high-FR {tot['raw_hiFR']} -> choice (raw p<.05) {tot['choice_raw']} "
          f"(FDR {tot['choice_fdr']}) -> leading {tot['leading']} & "
          f"movement-surviving {tot['move_surv']} & contrast-surviving "
          f"{tot['stim_surv']} -> TRIPLE {tot['triple']}")
    print(f"  movement attribution: {tot['locked']} movement-LOCKED vs "
          f"{tot['leading']} decision-LEADING choice cells")
    ndlc = int(c["has_dlc"].sum())
    subw = int((c["sig_raw"] & c["has_dlc"] & (c["p_move_wheel"] < 0.05)).sum())
    subf = int((c["sig_raw"] & c["has_dlc"] & (c["p_move_full"] < 0.05)).sum())
    print(f"  DLC coverage: {ndlc}/{len(c)} cells ({ndlc/len(c):.0%}); on that subset "
          f"choice-sig survive wheel-only {subw} vs wheel+body {subf} "
          f"(adding body removes {subw-subf})")
    best = rt.sort_values("triple", ascending=False).iloc[0]
    print(f"  best region: {best['region']} - {int(best['triple'])} triple-filtered "
          f"decision cells (pooled N~{int(best['pooledN_triple'])}); "
          f"{int(best['leading'])} leading (pooled N~{int(best['pooledN_leading'])})")
    tri = int(tot["triple"]); lead = int(tot["leading"])
    state = ("INTACT" if tri >= 25 else "THINNED-BUT-VIABLE" if lead >= 12
             else "COLLAPSED-INTO-MOVEMENT")
    print(f"  -> modelable decision-related lane: {state}")


def make_figure(c, rt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    stages = ["raw_hiFR", "choice_raw", "leading", "move_surv", "triple"]
    labels = ["raw\nhigh-FR", "choice\n(p<.05)", "leading\n(pre-move)",
              "movement\nsurviving", "TRIPLE\n(decision)"]
    x = np.arange(len(stages)); w = 0.16
    for i, reg in enumerate(rt["region"]):
        vals = rt.loc[rt["region"] == reg, stages].to_numpy().ravel()
        ax.bar(x + (i - 2) * w, vals, w, label=reg)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("# high-FR cells")
    ax.set_title("Selectivity filter cascade per region", fontsize=10, loc="left")
    ax.legend(fontsize=8, title="region")
    # B: lead vs lock — early vs peri choice selectivity for choice-responsive cells
    ax = axes[1]
    ps = c[c["sig_raw"]]
    col = ps["leading"].map({True: "#27a", False: "#c33"})
    ax.scatter(np.abs(ps["auc_early"] - 0.5), np.abs(ps["auc_peri"] - 0.5),
               c=col, alpha=0.7, s=22)
    lim = max(0.25, np.nanmax(np.abs(ps[["auc_early", "auc_peri"]].to_numpy() - 0.5)) + 0.02)
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("|AUC-0.5|  early / pre-movement (leads)")
    ax.set_ylabel("|AUC-0.5|  peri-movement (locked)")
    ax.set_title(f"Lead (blue) vs movement-locked (red): {int(ps['leading'].sum())} "
                 f"leading, {int(ps['locked'].sum())} locked", fontsize=10, loc="left")
    fig.suptitle("Phase 2 step 2: movement-controlled choice selectivity "
                 "in BWM decision-core regions", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nSaved figure -> {FIG_PATH}")


def build_sample_full(one):
    """ALL official BWM insertions covering the core target regions (no cap)."""
    pids, seen = [], set()
    for reg in CORE_REGIONS:
        ids, _ = one.search_insertions(atlas_acronym=reg, project=BWM_PROJECT,
                                       details=True)
        for pid in [str(i) for i in ids]:
            if pid not in seen:
                seen.add(pid); pids.append(pid)
    return pids


def run_features(one, br, pids, cleanup, use_dlc):
    done = set()
    if FEATURES_CSV.exists():
        done = set(pd.read_csv(FEATURES_CSV, usecols=["cell"])["cell"]
                   .str.split(":").str[0].unique())
    todo = [p for p in pids if p not in done]
    print(f"Feature extraction: {len(pids)} insertions, {len(todo)} to do.")
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    for i, pid in enumerate(todo, 1):
        try:
            df, meta = process_features(one, br, pid, cleanup=cleanup, use_dlc=use_dlc)
            n = 0 if df is None else df["cell"].nunique()
            if df is not None and len(df):
                df.to_csv(FEATURES_CSV, mode="a", header=not FEATURES_CSV.exists(),
                          index=False)
            print(f"  [{i}/{len(todo)}] {pid[:8]} -> {n} hi-FR cells "
                  f"(dlc={meta.get('has_dlc')}, trials={meta.get('n_trials')})")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(todo)}] {pid[:8]} FAILED: {repr(exc)[:90]}")
    print(f"\nSaved features -> {FEATURES_CSV}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["features", "analyze", "report", "test-one"],
                    default="analyze")
    ap.add_argument("--cap", type=int, default=12)
    ap.add_argument("--no-cleanup", action="store_true")
    ap.add_argument("--dlc", action="store_true", help="also load DLC body speeds (slow)")
    ap.add_argument("--full", action="store_true",
                    help="ALL target-region insertions + full-coverage output paths")
    args = ap.parse_args()

    if args.full:   # publication full-coverage: separate output files
        global FEATURES_CSV, CELLS_CSV, REGION_CSV, FIG_PATH
        FEATURES_CSV = PROJECT_ROOT / "results" / "phase2_sel_features_full.csv"
        CELLS_CSV = PROJECT_ROOT / "results" / "phase2_sel_cells_full.csv"
        REGION_CSV = PROJECT_ROOT / "results" / "phase2_sel_region_full.csv"
        FIG_PATH = PROJECT_ROOT / "figures" / "phase2_selectivity_controlled_full.png"

    if args.stage == "analyze":
        run_analyze(); return
    if args.stage == "report":
        run_report(); return

    one = get_one(); print(f"Connected: {one.alyx.base_url}\n")
    br = BrainRegions()
    if args.stage == "test-one":
        import time
        ids, _ = one.search_insertions(atlas_acronym="SNr", project=BWM_PROJECT,
                                       details=True)
        t = time.time()
        df, meta = process_features(one, br, str(ids[0]), cleanup=not args.no_cleanup,
                                    use_dlc=args.dlc)
        print(f"meta: {meta}  [{time.time()-t:.1f}s]")
        if df is not None and len(df):
            # quick wheel-stillness check inside the deliberative window
            per_trial = df.drop_duplicates("eid")  # not per trial; recompute below
            g = df.groupby("cell").first()
            print(f"cells: {df['cell'].nunique()} | rows: {len(df)}")
            print("wheel_speed in [stimOn,fm] (per-trial): "
                  f"median {df['wheel_speed'].median():.3f}, "
                  f"frac~0 (<0.05) {np.mean(df['wheel_speed']<0.05):.0%}")
            print(f"paw_speed finite: {np.isfinite(df['paw_speed']).mean():.0%}")
            print(df[["region", "fr_window", "rate_delib", "rate_early", "rate_peri",
                      "wheel_speed", "wheel_disp", "paw_speed"]].describe().to_string())
    elif args.stage == "features":
        pids = build_sample_full(one) if args.full else build_sample(one, args.cap)
        print(f"{'FULL coverage' if args.full else 'sampled'}: {len(pids)} insertions")
        run_features(one, br, pids, cleanup=not args.no_cleanup, use_dlc=args.dlc)


if __name__ == "__main__":
    main()

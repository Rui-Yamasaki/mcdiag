"""Steinmetz-2019 -> IBL-pipeline adapter (binned NMA release; PURE, no download).

Maps the cached 10 ms-binned Steinmetz fields onto the per-(cell,trial) feature schema the
phase2 analyses consume, so the IBL selectivity / choice-vs-stimulus / population-decode logic
can be re-run on Steinmetz. 10 ms binning matches our IBL pipeline -> this is the right route.

Conventions (verified on the data):
  - choice: chose = -response (on correct unequal-contrast trials corr(response,sign(cr-cl))
    = -1.0 exactly). We store `choice = response`, so the IBL downstream `chose=-sign(choice)`
    is correct. NoGo (response==0) trials are EXCLUDED.
  - signed = contrast_right - contrast_left ; stim_side = sign(signed) ; absc = max(cl,cr).
  - movement onset: reaction_time[:,0] is wheel-onset in MS rel. stimulus (col 1 = valid flag);
    fm_bin = 50 + round(RT_ms/10) (stim onset = bin 50). Falls back to response_time bin when
    RT invalid. Deliberation window = [stim, fm]; pre = [stim+0.05, fm-0.05]; peri = [fm-0.05,
    fm+0.05] (bins re-indexed to movement onset).
  - movement covariates: wheel (|velocity|) + face motion-energy + pupil area, in-window means.
    Mapped onto IBL columns wheel_speed/wheel_disp/paw_speed(=face)/nose_speed(=pupil) so the
    IBL movement controls pick them up. NO body DLC (face ME is the partial proxy); NO block
    prior in Steinmetz (pL set constant 0.5).

  python src/steinmetz_adapter.py --stage smoke     # validate on one MRN session (gate)
  python src/steinmetz_adapter.py --stage build     # build full features CSV (MRN/SC/SNr)
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from ibl_one import PROJECT_ROOT  # noqa: E402

DATA = PROJECT_ROOT / "data" / "steinmetz"
FEATURES_CSV = PROJECT_ROOT / "results" / "steinmetz_features.csv"
SMOKE_FIG = PROJECT_ROOT / "figures" / "steinmetz_adapter_smoke.png"

BIN = 0.01                      # s/bin
STIM_BIN = 50                   # stim onset = 0.5 s -> bin 50
FR_HI = 10.0                    # in-window high-FR floor (matches IBL phase2)
MIN_PER_SIDE = 8
REGION_MAP = {"MRN": "MRN", "SNr": "SNr", "SCm": "SC", "SCig": "SC"}   # SC = SCm+SCig


def load_all():
    out = []
    for j in range(3):
        out += list(np.load(DATA / f"steinmetz_part{j}.npz", allow_pickle=True)["dat"])
    return out


def trial_table(s):
    """Per-trial behaviour + movement-onset bin for Go trials (response != 0)."""
    resp = s["response"].astype(float)
    cl, cr = s["contrast_left"].astype(float), s["contrast_right"].astype(float)
    fb = s["feedback_type"].astype(float)
    rt = np.asarray(s["reaction_time"], float)              # (n,2): ms rel stim, valid flag
    rt_ms, rt_ok = rt[:, 0], rt[:, 1] > 0
    rsp_t = np.asarray(s["response_time"], float).ravel()   # s, window frame
    n = len(resp)
    rsp_bin = np.round(rsp_t / BIN).astype(int)             # response completion bin
    fm_bin = np.where(rt_ok & np.isfinite(rt_ms) & (rt_ms > 0),
                      STIM_BIN + np.round(rt_ms / 10.0), rsp_bin).astype(int)
    # movement onset can't precede stim+30ms nor exceed response completion (physical cap)
    fm_bin = np.clip(fm_bin, STIM_BIN + 3, np.clip(rsp_bin, STIM_BIN + 4, 249))
    chose = -resp                                           # +1 = chose right
    signed = cr - cl
    go = resp != 0
    return pd.DataFrame(dict(trial=np.arange(n), go=go, choice=resp, chose=chose,
                             signed=signed, absc=np.fmax(cl, cr), feedback=fb,
                             fm_bin=fm_bin))


def _wmean(arr, a, b):
    """mean over bins [a,b) along last axis, NaN-safe."""
    seg = arr[..., a:b]
    return np.nanmean(seg) if seg.size else np.nan


def session_features(s, sess_id):
    ba = np.array([str(x) for x in s["brain_area"]])
    keep_reg = np.array([REGION_MAP.get(r) for r in ba], dtype=object)
    sel = np.array([r is not None for r in keep_reg])
    if not sel.any():
        return None
    spks = s["spks"]                                        # (neurons, trials, 250)
    tt = trial_table(s)
    g = tt[tt.go].copy()
    if len(g) < 2 * MIN_PER_SIDE or (g.chose < 0).sum() < MIN_PER_SIDE \
            or (g.chose > 0).sum() < MIN_PER_SIDE:
        return None
    wheel = np.abs(s["wheel"][0])                           # (trials,250) |velocity|
    wvel = s["wheel"][0]
    face = s["face"][0] if np.size(s.get("face", [])) else np.full_like(wheel, np.nan)
    pup = s["pupil"][0] if np.size(s.get("pupil", [])) else np.full_like(wheel, np.nan)

    # per-trial movement covariates in the deliberative window [stim, fm]
    rows = []
    nidx = np.where(sel)[0]
    # precompute per-trial window covariates
    cov = {}
    for _, t in g.iterrows():
        a, b = STIM_BIN, int(t.fm_bin)
        cov[int(t.trial)] = (float(np.mean(wheel[int(t.trial), a:b])) if b > a else 0.0,
                             float(np.sum(wvel[int(t.trial), a:b]) * BIN) if b > a else 0.0,
                             _wmean(face[int(t.trial)], a, b), _wmean(pup[int(t.trial)], a, b))
    for u in nidx:
        reg = keep_reg[u]
        sp = spks[u]                                        # (trials,250)
        # in-window rates per Go trial
        rec = []
        rd = []
        for _, t in g.iterrows():
            tr = int(t.trial); a, b = STIM_BIN, int(t.fm_bin)
            dur = (b - a) * BIN
            rdelib = sp[tr, a:b].sum() / dur if dur > 0 else 0.0
            # pre/peri re-indexed to movement onset
            pa0, pa1 = STIM_BIN + 5, b - 5
            rpre = (sp[tr, pa0:pa1].sum() / ((pa1 - pa0) * BIN)) if pa1 > pa0 else np.nan
            rperi = sp[tr, max(b - 5, 0):b + 5].sum() / (min(b + 5, 250) - max(b - 5, 0)) / BIN
            rd.append(rdelib); rec.append((tr, rdelib, rpre, rperi))
        rd = np.array(rd)
        if rd.mean() < FR_HI:                               # high-FR lane only
            continue
        fr_overall = sp.sum() / (sp.shape[0] * sp.shape[1] * BIN)
        for (tr, rdelib, rpre, rperi) in rec:
            t = g[g.trial == tr].iloc[0]
            w, wd, fc, pp = cov[tr]
            rows.append(dict(cell=f"{sess_id}:{u}", region=reg, eid=sess_id,
                             fr_overall=fr_overall, fr_window=float(rd.mean()),
                             choice=int(t.choice), absc=float(t.absc),
                             signed=float(t.signed), pL=0.5,
                             feedback=int(t.feedback),
                             rate_delib=float(rdelib), rate_early=float(rpre),
                             rate_peri=float(rperi),
                             wheel_speed=w, wheel_disp=wd, paw_speed=fc, nose_speed=pp))
    return pd.DataFrame(rows) if rows else None


def build(verbose=True):
    sess = load_all()
    parts = []
    for s in sess:
        sid = f"{s['mouse_name']}_{s['date_exp']}"
        df = session_features(s, sid)
        if df is not None and len(df):
            parts.append(df)
            if verbose:
                byr = df.groupby("region")["cell"].nunique().to_dict()
                print(f"  {sid}: {df['cell'].nunique()} hi-FR cells {byr}")
    feats = pd.concat(parts, ignore_index=True)
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(FEATURES_CSV, index=False)
    print(f"\nSaved features -> {FEATURES_CSV}")
    print(f"  total hi-FR cells: {feats['cell'].nunique()} | rows: {len(feats)}")
    print(f"  per region: {feats.groupby('region')['cell'].nunique().to_dict()}")
    print(f"  sessions: {feats['eid'].nunique()}")
    return feats


def _auc(x, y):
    ok = np.isfinite(x); x, y = x[ok], y[ok]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 < 3 or n0 < 3:
        return np.nan
    r = rankdata(x)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def smoke():
    """Validate the adapter on ONE MRN session: field mapping + PSTH + choice-AUC sanity."""
    sess = load_all()
    target = None
    for s in sess:
        ba = np.array([str(x) for x in s["brain_area"]])
        if (ba == "MRN").sum() >= 30:
            target = s; break
    if target is None:
        print("SMOKE FAIL: no MRN session found"); return False
    sid = f"{target['mouse_name']}_{target['date_exp']}"
    print(f"== SMOKE: {sid} ==")
    tt = trial_table(target)
    g = tt[tt.go]
    # RT-derived movement onset sanity
    rsp_bin = np.round(np.asarray(target["response_time"], float).ravel() / BIN)
    okfm = (g.fm_bin.to_numpy() > STIM_BIN) & (g.fm_bin.to_numpy() <= rsp_bin[g.trial])
    print(f"  Go trials: {len(g)} (chose L {int((g.chose<0).sum())} / R {int((g.chose>0).sum())})")
    print(f"  movement onset in (stim, response_time]: {okfm.mean():.0%} of trials "
          f"(median fm_bin {int(g.fm_bin.median())} = {(g.fm_bin.median()-STIM_BIN)*BIN*1000:.0f} ms post-stim)")
    print(f"  decorrelated: equal-contrast {int((g.signed==0).sum())}, "
          f"error {int(((g.signed!=0)&(g.chose!=np.sign(g.signed))).sum())}")
    df = session_features(target, sid)
    if df is None or not len(df):
        print("SMOKE FAIL: no features built"); return False
    mrn = df[df.region == "MRN"]
    print(f"  MRN hi-FR cells: {mrn['cell'].nunique()} | feature columns: {list(df.columns)}")
    print(f"  movement covariates finite: wheel {np.isfinite(df.wheel_speed).mean():.0%}, "
          f"face {np.isfinite(df.paw_speed).mean():.0%}, pupil {np.isfinite(df.nose_speed).mean():.0%}")
    # per-cell choice AUC (raw, all Go trials) sanity
    aucs = []
    for cell, gg in mrn.groupby("cell"):
        y = (gg.choice.to_numpy() > 0).astype(int)         # arbitrary label; |AUC-.5| is the signal
        aucs.append(_auc(gg.rate_delib.to_numpy(float), y))
    aucs = np.array([a for a in aucs if np.isfinite(a)])
    sel = np.abs(aucs - 0.5)
    print(f"  MRN per-cell choice AUC: mean|AUC-.5| {sel.mean():.3f}, "
          f"max {sel.max():.3f}, cells |AUC-.5|>0.15: {int((sel>0.15).sum())}/{len(aucs)}")

    # PSTH sanity: most choice-selective MRN cell, split by chose L/R, stim-aligned
    best = mrn.groupby("cell").apply(
        lambda gg: abs(_auc(gg.rate_delib.to_numpy(float), (gg.choice.to_numpy() > 0).astype(int)) - 0.5),
        include_groups=False).sort_values(ascending=False)
    bcell = best.index[0]
    u = int(bcell.split(":")[1])
    spks = target["spks"][u]
    gtr = g.trial.to_numpy()
    chose = g.chose.to_numpy()
    L = gtr[chose < 0]; R = gtr[chose > 0]
    tvec = (np.arange(250) - STIM_BIN) * BIN
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(tvec, spks[L].mean(0) / BIN, label=f"chose L (n={len(L)})", color="#27a")
    ax[0].plot(tvec, spks[R].mean(0) / BIN, label=f"chose R (n={len(R)})", color="#c44")
    ax[0].axvline(0, color="k", lw=0.8, ls="--"); ax[0].set_xlim(-0.3, 1.0)
    ax[0].set_xlabel("time from stim (s)"); ax[0].set_ylabel("firing rate (Hz)")
    ax[0].set_title(f"Most choice-selective MRN cell {bcell}\n(|AUC-.5|={best.iloc[0]:.2f})", fontsize=9)
    ax[0].legend(fontsize=8)
    ax[1].hist(sel, bins=20, color="#888"); ax[1].axvline(0.15, color="r", ls="--", lw=1)
    ax[1].set_xlabel("|choice AUC - 0.5| (MRN cells)"); ax[1].set_ylabel("# cells")
    ax[1].set_title("MRN per-cell choice selectivity (raw, all Go trials)", fontsize=9)
    fig.suptitle(f"Steinmetz adapter smoke test — {sid}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    SMOKE_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SMOKE_FIG, dpi=150)
    print(f"  saved PSTH sanity -> {SMOKE_FIG}")
    ok = (okfm.mean() > 0.8) and (mrn["cell"].nunique() >= 10) and (sel.max() > 0.15)
    print(f"  SMOKE {'PASS' if ok else 'FAIL'}: movement-onset sane, fields map, "
          f"MRN choice selectivity present")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "build"], default="smoke")
    args = ap.parse_args()
    if args.stage == "smoke":
        smoke()
    else:
        build()

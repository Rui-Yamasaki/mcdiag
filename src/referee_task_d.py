"""TASK D — exact numbers for manuscript text fixes (report-only; cached data, no new modelling)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm, false_discovery_control

from ibl_one import PROJECT_ROOT
R = PROJECT_ROOT / "results"


def task_d1_decode_p():
    print("=== D1. Movement-controlled decode p per analysis setting (Table S1 sweep) ===")
    fr = pd.read_csv(R / "robustness_summary.csv")
    print("  Firing-rate floor / window (results/robustness_summary.csv):")
    for _, r in fr.iterrows():
        print(f"    {r.knob}={str(r.value):<5}: decode AUC {r.MRN_decode_AUC:.4f}  p {r.MRN_decode_p:.4f}")
    bz = pd.read_csv(R / "robustness/binsize_summary.csv")
    print("  Bin size (results/robustness/binsize_summary.csv):")
    for _, r in bz.iterrows():
        lbl = r.bin if pd.isna(r.bin_ms) else f"{int(r.bin_ms)}ms"
        print(f"    {lbl:<6}: decode AUC {r.MRN_decode_AUC:.4f}  p {r.MRN_decode_p:.4f}")
    allp = list(fr.MRN_decode_p) + list(bz.MRN_decode_p)
    print(f"  -> across-settings decode-p range = [{min(allp):.3f}, {max(allp):.3f}]")
    # the "per-session p ~ 0.06-0.10" phrase:
    rt = pd.read_csv(R / "audit_realtrial_decode.csv")
    ibl = rt[(rt.dataset == "IBL") & (rt.region == "MRN") & (rt["mode"] == "expanded")].iloc[0]
    st = rt[(rt.dataset == "Steinmetz") & (rt.region == "MRN") & (rt["mode"] == "expanded")].iloc[0]
    print(f"  CLARIFY 'per-session p ~ 0.06-0.10': the real-trial per-session-CV decode is the only "
          f"per-session computation; its MRN movement-controlled (expanded) perm-p =")
    print(f"     IBL MRN {ibl.cv_auc:.3f} p={ibl.perm_p:.3f} ; Steinmetz MRN {st.cv_auc:.3f} p={st.perm_p:.3f}")
    print(f"  -> 'p~0.06-0.10' = the two datasets' MRN expanded-movement decode p (0.057 / 0.095), "
          f"NOT a separate per-session range. Across-settings range is {min(allp):.3f}-{max(allp):.3f}.")


def task_d2_peri_pre():
    print("\n=== D2. Fig 3a peri- vs pre-movement choice-selective fraction test ===")
    c = pd.read_csv(R / "phase2_sel_cells_full.csv")
    n = len(c)
    npre = int((c.p_early < 0.05).sum()); nperi = int((c.p_peri < 0.05).sum())
    ppre, pperi = npre / n, nperi / n
    print(f"  n(hi-FR cells)={n}; pre(early) {npre} ({ppre:.1%}); peri {nperi} ({pperi:.1%})")
    # two-proportion z-test (as requested; treats the two windows as independent)
    phat = (npre + nperi) / (2 * n)
    se = np.sqrt(phat * (1 - phat) * (2 / n))
    z = (pperi - ppre) / se
    p_z = 2 * (1 - norm.cdf(abs(z)))
    print(f"  two-proportion z-test: z={z:.2f}, p={p_z:.2e}")
    # McNemar paired (SAME cells in both windows -> the correct test)
    e = (c.p_early < 0.05).to_numpy(); pe = (c.p_peri < 0.05).to_numpy()
    b = int((pe & ~e).sum()); cc = int((e & ~pe).sum())
    chi2 = (abs(b - cc) - 1) ** 2 / (b + cc)
    from scipy.stats import chi2 as chi2d
    p_mc = 1 - chi2d.cdf(chi2, 1)
    print(f"  McNemar paired (same cells; peri-only={b}, pre-only={cc}): chi2(cc)={chi2:.1f}, p={p_mc:.2e}")
    print("  -> both reject; peri >> pre. McNemar is the appropriate test (paired); z-test as requested.")


def task_d3_bodypose():
    print("\n=== D3. Body-pose increment denominator (wheel-only -> wheel+pose) ===")
    rf = pd.read_csv(R / "phase2_sel_region_full.csv")
    wheel = int(rf.move_surv_wheel.sum()); full = int(rf.move_surv.sum())
    print(f"  wheel-only movement-independent survivors = {wheel}  (per region: "
          f"{dict(zip(rf.region, rf.move_surv_wheel))})")
    print(f"  wheel+pose movement-independent survivors = {full}  (per region: "
          f"{dict(zip(rf.region, rf.move_surv))})")
    removed = wheel - full
    print(f"  removed by adding body pose = {removed} = {removed/wheel:.1%} OF THE {wheel} wheel-survivors "
          f"(NOT of the 301 choice-selective).  Expected 161->131, 18.6%.")


def task_d4_auc_map():
    print("\n=== D4. AUC mapping (SD <-> AUC) ===")
    rt = pd.read_csv(R / "audit_realtrial_decode.csv")
    auc = rt[(rt.dataset == "IBL") & (rt.region == "MRN") & (rt["mode"] == "expanded")].cv_auc.iloc[0]
    print(f"  measured MRN movement-controlled POPULATION decode AUC = {auc:.4f} (~0.53)")
    for sd in (0.059, 0.06):
        print(f"  Gaussian SD->AUC map: effect {sd:.3f} SD -> AUC = Phi({sd}/sqrt2) = {norm.cdf(sd/np.sqrt(2)):.4f}")
    print("  -> the pooled +0.06 SD maps to AUC ~0.52 (Gaussian), while the population decode is ~0.53;")
    print("     '~0.06 SD, about AUC 0.53' should read AUC ~0.52 (SD-map) vs 0.528 (population decode) "
          "-- two different statistics, both small.")


def task_d5_shortfall():
    print("\n=== D5. Most-generous shortfall to the 0.24 SD bar ===")
    fr = pd.read_csv(R / "robustness_summary.csv")
    floors = fr[fr.knob == "fr_floor"]
    vals = {int(r.value): r.MRN_pooled_SD for _, r in floors.iterrows()}
    print(f"  MRN pooled decision effect across FR floors: {dict((k, round(v,4)) for k,v in vals.items())}")
    most = max(vals.values())
    print(f"  most generous (largest) = {most:.3f} SD (at 25 Hz floor); range ~0.06-0.13 SD")
    print(f"  shortfall to 0.24 bar = 0.24/{most:.3f} = {0.24/most:.2f}x  (i.e. ~1.8x, NOT 'two or more')")


if __name__ == "__main__":
    task_d1_decode_p()
    task_d2_peri_pre()
    task_d3_bodypose()
    task_d4_auc_map()
    task_d5_shortfall()

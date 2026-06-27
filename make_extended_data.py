#!/usr/bin/env python
"""Build the Extended Data items for the Nature Computational Science submission.

This script TABULATES already-committed results; it runs no new analysis and changes no number.
Every value is read from a cached CSV under results/ and the source file + columns are named in
each output's header. It produces:

  Extended Data Table 1  (ed_table1_recovery_grid.{csv,md})
      Step-vs-ramp recovery rate by mean firing rate (rows) x pooled trial count N (columns), at the
      canonical ramp diffusion sigma = 0.4, in two blocks: negative-binomial (realistic, Fig 2a) and
      clean Poisson (best case, Extended Data Fig 1 / former Fig S1). 0.80-recovery crossings marked.
      Sources: results/phase1_recovery_nb_grid_R40.csv  (NB, column recovery, fano >= 1.5)
               results/phase1_recovery_sweep.csv         (Poisson, column correct)

  Extended Data Fig 1/2  (extended_data_fig1.{png,pdf}, extended_data_fig2.{png,pdf})
      The two former Supplementary Figures, re-exported by their own generation code at <=180 mm
      (300 dpi) and mirrored here: Fig S1 -> ED Fig 1 (clean-Poisson recovery map),
      Fig S2 -> ED Fig 2 (diffusion-sigma sensitivity).

The control-family ground-truth benchmark (former ED Table) is NOT built here: it duplicates
Supplementary Table 2 and stays in the Supplementary Information.

Run:  python make_extended_data.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIGDIR = ROOT / "figures"
ED = ROOT / "extended_data"
ED.mkdir(parents=True, exist_ok=True)

MM_PER_IN = 25.4
WIDTH_CAP_MM = 180.0


# ============================================================ Extended Data Table 1 (recovery grid)
NS = [20, 40, 80, 160, 320]
FRS = [2.0, 5.0, 10.0, 20.0, 40.0]
THRESH = 0.80


def _first_clear(by_n):
    clears = [n for n in NS if n in by_n.index and by_n.loc[n] >= THRESH - 1e-9]
    return int(min(clears)) if clears else None


def build_ed_table1():
    nb = pd.read_csv(RES / "phase1_recovery_nb_grid_R40.csv")
    nb_over = nb[nb["fano"] >= 1.5]
    nb_grid = nb_over.groupby(["fr", "N"])["recovery"].mean().unstack("N")

    pois = pd.read_csv(RES / "phase1_recovery_sweep.csv")
    pois_grid = pois.groupby(["fr", "N"])["correct"].mean().unstack("N")

    rows = []
    for emission, grid in [("negative_binomial", nb_grid), ("poisson", pois_grid)]:
        for fr in FRS:
            by_n = grid.loc[fr]
            row = {"emission": emission, "fr": fr}
            row.update({f"N{n}": round(float(by_n.loc[n]), 3) for n in NS})
            row["first_N_reaching_0.80"] = _first_clear(by_n)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(ED / "ed_table1_recovery_grid.csv", index=False)

    # --- human-readable .md (one table block per emission; >=0.80 cells marked **bold**) ---
    def block(emission, grid, label):
        hdr = "| mean FR (Hz) | " + " | ".join(f"N={n}" for n in NS) + " | first N reaching 0.80 |"
        sep = "|" + "---|" * (len(NS) + 2)
        lines = [hdr, sep]
        for fr in FRS:
            by_n = grid.loc[fr]
            cells = []
            for n in NS:
                v = float(by_n.loc[n])
                cells.append(f"**{v:.3f}**" if v >= THRESH - 1e-9 else f"{v:.3f}")
            fc = _first_clear(by_n)
            lines.append(f"| {fr:g} | " + " | ".join(cells) + f" | {fc if fc else '--'} |")
        return f"### {label}\n\n" + "\n".join(lines)

    md = f"""# Extended Data Table 1. Step-vs-ramp recovery grid (firing rate x pooled trials).

Recovery rate for the step-vs-ramp model-selection test, by mean firing rate (rows) and pooled trial
count N (columns), at the canonical ramp diffusion sigma = 0.4. Recovery is the fraction of synthetic
units whose true generator (step or ramp) is correctly recovered. Two emission models are shown: the
realistic negative-binomial (overdispersed, Fano >= 1.5; the regime of real spikes, Fig 2a) and the
clean-Poisson best case (the lenient upper bound, Extended Data Fig 1). Cells at or above the 0.80
recovery threshold are **bold**; the last column gives the smallest N that reaches 0.80 in that row.

{block("negative_binomial", nb_grid, "Negative-binomial emission (realistic; Fig 2a)")}

{block("poisson", pois_grid, "Clean-Poisson emission (best case; Extended Data Fig 1)")}

**Reading the grid.** At the IBL single-unit operating point (about 20 Hz, about 40 trials) neither
emission recovers the generator: realistic NB recovery is {nb_grid.loc[20.0, 40]:.3f} and needs pooling to
N = {_first_clear(nb_grid.loc[20.0])} to clear 0.80, while the lenient clean-Poisson case is {pois_grid.loc[20.0, 40]:.3f} and clears 0.80 at
N = {_first_clear(pois_grid.loc[20.0])} (about half the NB trial count). The grids confirm that overdispersion, not just trial
count, sets the recovery floor.

Sources:
- Negative-binomial block: results/phase1_recovery_nb_grid_R40.csv (column recovery; rows with
  fano >= 1.5 averaged over fano in {{1.5, 2, 3}} and over both generators; R = 40 repeats). sigma = 0.4 is
  fixed for the whole file (no sigma column).
- Poisson block: results/phase1_recovery_sweep.csv (column correct; averaged over both generators).
  sigma = 0.4 is fixed for the whole file (no sigma column).
Cross-reference: Fig 2a (NB) and Extended Data Fig 1 / former Supplementary Fig 1 (Poisson).
"""
    (ED / "ed_table1_recovery_grid.md").write_text(md, encoding="utf-8", newline="\n")
    return df


# ============================================================ Extended Data figures (mirror)
def mirror_ed_figures():
    """Mirror the re-exported former Supplementary Figures into extended_data/ and assert each is
    300 dpi and <= 180 mm wide. (figS1/figS2 are re-exported at <=180 mm by their own generation
    code: make_fig2.render_figS and make_figS2.main.)"""
    pairs = [("figS1", "extended_data_fig1"), ("figS2", "extended_data_fig2")]
    out = []
    for src, dst in pairs:
        for ext in ("png", "pdf"):
            shutil.copyfile(FIGDIR / f"{src}.{ext}", ED / f"{dst}.{ext}")
        im = Image.open(ED / f"{dst}.png")
        dpi = im.info.get("dpi", (300, 300))[0]
        w_mm = im.size[0] / dpi * MM_PER_IN
        assert dpi >= 299.0, f"{dst}.png is {dpi:.0f} dpi (< 300)"
        assert w_mm <= WIDTH_CAP_MM + 1e-6, f"{dst}.png is {w_mm:.1f} mm wide (> {WIDTH_CAP_MM})"
        out.append((dst, im.size[0], im.size[1], dpi, w_mm))
    return out


def main():
    t1 = build_ed_table1()
    figs = mirror_ed_figures()
    print(f"wrote {ED/'ed_table1_recovery_grid.csv'}  ({len(t1)} rows)")
    print(f"wrote {ED/'ed_table1_recovery_grid.md'}")
    for dst, w, h, dpi, w_mm in figs:
        print(f"mirrored {dst}.png/.pdf  ({w}x{h}px, {dpi:.0f} dpi, {w_mm:.1f} mm wide)  [<=180 mm OK]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""One-command regeneration of every display item, from cached results/ (no simulations re-run).

    python make_figures.py

Regenerates Figs 1-6, Fig S1 (via make_fig2.py::render_figS), Fig S2, and the Supplementary
Information document, each from cached CSV/JSON under results/ using the shared figstyle.py. Figures
are written to figures/ as 300 dpi PNG + vector PDF (per figstyle rcParams); the Supplementary
Information to docs/supplementary_information.md.

Fig 2 and Fig 5 read the R=40 grids (phase1_recovery_nb_grid_R40.csv,
phase1_recovery_poisson_misspec_R40.csv), matching the manuscript. That R=40 source is wired into
make_fig2.py / make_fig5.py directly (not passed as an argument), so invoking them the normal way
reproduces the final R=40 panels. NOTHING here re-runs the recovery simulations — every script only
reads cached results/.

Each script is launched as an isolated subprocess (identical to running it standalone), timed, and
its expected outputs are checked for existence AND fresh modification time (to catch silent failures).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "figures"
DOCSDIR = ROOT / "docs"

# (label, script, output_dir, [expected output files])
PIPELINE = [
    ("Fig 1",             "make_fig1.py",        FIGDIR,  ["fig1.png", "fig1.pdf"]),
    ("Fig 2 + Fig S1",    "make_fig2.py",        FIGDIR,  ["fig2.png", "fig2.pdf", "figS1.png", "figS1.pdf"]),
    ("Fig 3",             "make_fig3.py",        FIGDIR,  ["fig3.png", "fig3.pdf"]),
    ("Fig 4",             "make_fig4.py",        FIGDIR,  ["fig4.png", "fig4.pdf"]),
    ("Fig 5",             "make_fig5.py",        FIGDIR,  ["fig5.png", "fig5.pdf"]),
    ("Fig 6",             "make_fig6.py",        FIGDIR,  ["fig6.png", "fig6.pdf"]),
    ("Fig S2",            "make_figS2.py",       FIGDIR,  ["figS2.png", "figS2.pdf"]),
    ("Supplementary Info", "make_supplementary.py", DOCSDIR, ["supplementary_information.md"]),
]


def run_one(label, script, outdir, outputs):
    before = {f: ((outdir / f).stat().st_mtime if (outdir / f).exists() else None) for f in outputs}
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(ROOT / script)], cwd=str(ROOT),
                          capture_output=True, text=True)
    dt = time.time() - t0
    missing = [f for f in outputs if not (outdir / f).exists()]
    stale = [f for f in outputs
             if before[f] is not None and (outdir / f).exists()
             and (outdir / f).stat().st_mtime <= before[f]]
    ok = (proc.returncode == 0) and not missing and not stale
    print(f"[{'OK' if ok else 'FAIL':4s}] {label:18s} {dt:6.1f}s  ->  {', '.join(outputs)}", flush=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-4:]) if proc.stderr else "(no stderr)"
        print(f"        exit {proc.returncode}; stderr tail:\n        {tail}", flush=True)
    if missing:
        print(f"        MISSING (cache error?): {missing}", flush=True)
    if stale:
        print(f"        NOT REWRITTEN this run: {stale}", flush=True)
    return dict(label=label, ok=ok, dt=dt, missing=missing, stale=stale, rc=proc.returncode)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    DOCSDIR.mkdir(parents=True, exist_ok=True)
    print(f"Regenerating all display items from cached results/  (python={sys.executable})\n", flush=True)
    t0 = time.time()
    results = [run_one(*item) for item in PIPELINE]
    total = time.time() - t0
    n_ok = sum(r["ok"] for r in results)
    print(f"\n{n_ok}/{len(results)} steps OK in {total:.1f}s total.", flush=True)
    if n_ok == len(results):
        print("Regenerated: Figs 1-6 (6), Fig S1 + Fig S2 (2), Supplementary Information (1).", flush=True)
    else:
        print("FAILED steps:", [r["label"] for r in results if not r["ok"]], flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

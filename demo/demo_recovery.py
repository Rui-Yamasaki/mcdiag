"""Self-contained DEMO — no IBL download required (~2-3 minutes on a laptop).

Runs the validated step-vs-ramp recovery engine on a tiny SYNTHETIC grid, using the bundled
curated reaction-time window pool (demo/window_pool.npy: 2074 RTs in [0.5, 2.0] s, float-only,
no identifiers). No real spike data is touched — the engine simulates step/ramp spike trains from
known generators (negative-binomial, overdispersed emission) and recovers which model wins by
held-out cross-validated likelihood.

This is a small, fast, ILLUSTRATIVE synthetic run — NOT the full R=40 grid behind the manuscript
figures. It demonstrates the headline qualitative pattern: realistic recovery is near chance for a
single session (N=40) and rises once trials are pooled (N=160).

    python demo/demo_recovery.py

Expected: recovery ~0.60 at N=40 and higher (~0.70) at N=160 — the PATTERN (recovery rises with
trial count), not exact values. Because this uses a small demo replicate count (R=10) vs the
manuscript's R=40, these numbers are NOT expected to equal the exact figure recovery values.
Engine = src/phase1_recovery_hardened. Runtime ~2-3 minutes (40 cross-validated fits, run serially).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import phase1_recovery_hardened as H  # noqa: E402  validated engine (pure numpy/scipy, offline)

WINDOWS = np.load(Path(__file__).resolve().parent / "window_pool.npy")
FR, K, FANO, R = 20.0, 50, 1.5, 10          # 20 Hz, production grid K=50, overdispersed, 10 reps/cell
GENS = ("step", "ramp")
NS = (40, 160)


def main():
    print(f"DEMO: step-vs-ramp recovery (NB emission, Fano={FANO}, FR={FR:g} Hz, K={K}, R={R}/cell)")
    print(f"  window pool: {len(WINDOWS)} curated RTs (bundled, offline); "
          f"engine = phase1_recovery_hardened\n")
    t0 = time.time()
    for N in NS:
        hits = 0
        for gi, g in enumerate(GENS):
            for r in range(R):
                seed = 9000 + (NS.index(N) * len(GENS) + gi) * R + r
                hits += H.recover_canonical(g, N, FR, seed, WINDOWS, K, FANO, "nb")["correct"]
        rate = hits / (len(GENS) * R)
        print(f"  N={N:4d}:  recovery = {rate:.2f}   ({hits}/{len(GENS) * R} correct)")
    print(f"\ndone in {time.time() - t0:.0f}s — recovery rises with N "
          "(pooling across sessions enables identifiability).")


if __name__ == "__main__":
    main()

# Movement controls bias decision-signal estimates across the brain
DOI: 10.5281/zenodo.20686415
https://doi.org/10.21203/rs.3.rs-10037105/v1

Reanalysis code, figures, and the `mcdiag` movement-control diagnostic for a calibrated,
ground-truth test of how movement controls and recording yield gate what can be inferred about a
single-trial decision signal in mouse midbrain and hindbrain. Primary dataset: the IBL
Brain-Wide Map; cross-checked against Steinmetz et al. 2019.

Two contributions:

1. A recovery/identifiability framework that quantifies when a single-trial decision signal is
   statistically recoverable from spike counts at realistic trial counts (the 0.57 AUC
   recoverability bar).
2. A calibrated confound-control reanalysis showing that common movement controls distort the
   decision-signal estimate in both directions: flexible nonlinear controls over-correct (they
   absorb movement-orthogonal signal), dimensionality-reduced controls under-remove (they leave
   low-variance movement in), and only a calibration-passing linear control sits between the two.
   Under that valid control, an apparent per-cell choice code sits at the recoverability
   boundary, not above it.

## mcdiag: movement-control diagnostic

`mcdiag` injects test signals built from each session's own movement covariance and checks
whether a candidate movement control recovers them, flagging over-correction and under-removal
before you trust a result. It is what selects the linear control used in the paper.

    pip install -e mcdiag
    pytest mcdiag        # 13 tests; the recovery/synthetic tests run real simulations (~19 min)

> **Status:** code + figures backing a submitted manuscript. Reviewer-facing; fully reproducible
> from cached results (see *Reproduce the figures*). No raw neural data is bundled.

---

## Repository structure

```
.
├── make_figures.py              # one-command regeneration of Figs 1-6, the Extended Data figure sources, and the SI
├── make_fig1.py … make_fig6.py  # main manuscript figures (each reads cached results/, writes figures/)
├── make_figS2.py                # Extended Data Fig. 2 source (diffusion / intermediacy detail); Extended Data Fig. 1 is make_fig2.py::render_figS
├── make_supplementary.py        # Supplementary Information (5 tables, no figures) from corrected results -> docs/supplementary_information.md
├── make_extended_data.py        # Extended Data items: recovery-grid table + Extended Data Fig. 1 / 2 -> extended_data/
├── figstyle.py                  # shared manuscript style (Okabe-Ito palette, 300 dpi PNG + vector PDF, layout)
├── requirements.txt             # pinned dependencies (== versions); Python 3.13.7, Windows 11
├── LICENSE                      # BSD-3-Clause
│
├── src/                         # analysis pipeline (produces the cached results/)
│   ├── ibl_one.py               # shared public Open-Alyx ONE connection helper (read-only)
│   ├── list_sessions.py         # connect + list Brain-Wide-Map sessions/insertions
│   ├── psth_smoketest.py        # data-access smoke test (stream small products, build a PSTH)
│   ├── phase1_behavior.py       # deliberation-window behavioural analysis
│   ├── phase1b_engagement.py    # is the slow-RT mode engaged deliberation?
│   ├── phase1_recovery.py       # step-vs-ramp recovery engine + forward-LL validation (hmmlearn cross-check)
│   ├── phase1_recovery_hardened.py  # realistic regime: negative-binomial (overdispersed) emission + generators
│   ├── phase1_recovery_nb_grid.py   # full FR x N recovery grid under the realistic NB regime (feeds Fig 2a/5a)
│   ├── phase2_census.py         # population census of candidate regions (trimmed spikes)
│   ├── phase2_selectivity.py    # movement-controlled choice selectivity (permutation + FDR cascade)
│   ├── phase2_choice_vs_stim.py # choice-vs-stimulus on decorrelated (error / equal-contrast) trials
│   ├── phase2_population_decode.py  # pseudo-population choice decoder
│   ├── phase2_mrn_recovery_preflight.py  # MRN population step-vs-ramp recovery pre-flight
│   ├── audit_corrections.py     # dataset-wide BH-FDR cascade, pooled effects, gate, decode (audit pass)
│   ├── audit_ramp_validate.py   # forward-LL brute-force enumeration + sigma x K robustness
│   ├── audit_realtrial_decode.py    # real-trial cross-validated decode audit
│   ├── steinmetz_adapter.py     # Steinmetz-2019 -> IBL-format adapter
│   ├── steinmetz_coverage.py    # cross-dataset coverage assessment
│   ├── steinmetz_replicate.py   # replicate the two IBL findings + the 0.57 population-arm gate
│   ├── steinmetz_population_preflight.py  # derives the ~0.57 AUC / ~0.24 SD identifiability gate (R=50)
│   ├── robustness_sweep.py , robustness_binsize.py   # one-at-a-time robustness sweeps
│   ├── rerun_recovery_highR.py , rerun_poisson_misspec_R40.py  # R=40 re-runs (new CSVs; originals kept)
│   └── run_*.py , stepramp.py , phase*_validate.py , persist_/export_*.py   # launchers + helpers
│
├── results/                     # cached analysis outputs the figures read (small CSV/JSON; see below)
├── figures/                     # regenerable PNG (300 dpi) + PDF (vector)  [git-ignored: rebuild on demand]
├── docs/                        # Supplementary Information (supplementary_information.md = single authoritative SI) + project notes
├── demo/                        # self-contained demo + bundled tiny test input (window_pool.npy)
├── tests/                       # smoke test (forward-likelihood engine correctness)
├── docs/                        # detailed method/audit notes (robustness, technical audit, replication, …)
└── data/                        # local IBL/ONE + Steinmetz caches  [git-ignored; not part of the repo]
```

**Cached `results/` files the figure pipeline reads** (everything else under `results/` is analysis
provenance; large per-trial feature matrices are git-ignored - see *Data access*):
`phase1_recovery_validation.json`, `ramp_validation_forward.json`,
`phase1_recovery_nb_grid_R40.csv`, `phase1_recovery_poisson_misspec_R40.csv`,
`phase1_recovery_nb_grid.csv` (Fig S1 annotation), `phase1_recovery_sweep.csv`,
`phase1_recovery_stepiness.csv`, `ramp_validation_robustness.csv`, `audit_cascade_fdr.csv`,
`audit_pooled_effects.csv`, `audit_realtrial_decode.csv`, `steinmetz_gate.csv`,
`phase2_sel_cells_full.csv`, `phase2_sel_region_full.csv`, `phase2_session_delib.csv`,
`phase2_census_units.csv`, `robustness_summary.csv`, `results/robustness/binsize_summary.csv`.

---

## Install

- **OS tested:** Windows 11 (Python 3.13.7). Linux/macOS are expected to work - the figure
  pipeline, demo, and smoke test use only the cross-platform scientific stack; the IBL data stack
  (`ibllib`, `ONE-api`) is officially supported on Linux/macOS/Windows.
- **Python:** 3.13.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     |     Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

**Expected install time:** ~10-15 min on a typical broadband connection for the full pinned
environment (the IBL stack pulls large wheels: `ibllib`, `numba`, `scikit-image`, Qt). If you only
need to regenerate figures / run the demo, the core scientific subset
(`numpy pandas scipy matplotlib joblib scikit-learn hmmlearn`) installs in ~2-3 min.

`requirements.txt` is a full `pip freeze` with exact `==` pins. There is no `uv`/`conda` lock -
the project uses a standard `venv` + `pip` workflow.

---

## Data access

No raw neural data is committed (the `data/` cache is git-ignored). To **re-download / re-extract**
from source:

- **IBL Brain-Wide Map** - accessed via the `ONE-api` against the *public* Open Alyx instance.
  Credentials are required but are the **public, read-only** ones documented by IBL - **not a
  private secret** (`src/ibl_one.py` connects with `silent=True`; the password is IBL's public
  string `international` for the open Brain-Wide Map release).
  See the ONE documentation: <https://int-brain-lab.github.io/ONE/> and the IBL data portal
  <https://openalyx.internationalbrainlab.org>. The pipeline streams only small spike-sorting /
  trials products - never raw AP/LFP voltage.
- **Steinmetz et al. 2019** - converted to IBL-compatible format by `src/steinmetz_adapter.py`
  (the source `.npz` archives live under the git-ignored `data/steinmetz/`).

You do **not** need any data download to reproduce the figures, run the demo, or run the smoke test
- those use the cached `results/` and the bundled `demo/window_pool.npy`.

---

## Reproduce the display items

```bash
python make_figures.py         # Figs 1-6 (+ the Extended Data figure sources) and, via make_supplementary.py, the SI
python make_supplementary.py   # the tables-only Supplementary Information -> docs/supplementary_information.md
python make_extended_data.py   # Extended Data Fig. 1 / Extended Data Fig. 2 and the recovery-grid table -> extended_data/
```

Regenerates **Figs 1-6, Extended Data Fig. 1, Extended Data Fig. 2, the recovery-grid Extended Data
table, and the Supplementary Information** from cached `results/` only (no simulations re-run), as
300 dpi PNG + vector PDF in `figures/` and `extended_data/`, with the SI in
`docs/supplementary_information.md`. `make_figures.py` already invokes `make_supplementary.py`;
`make_extended_data.py` then builds the Extended Data items (it re-exports Extended Data Fig. 1 / 2
at <=180 mm and mirrors them into `extended_data/`).

**Expected run time:** ~18 s for `make_figures.py` (each script runs as an isolated subprocess;
per-item status, timing, and existence/freshness checks are printed). Fig 2 and Fig 5 read the
**R=40** recovery grids that match the manuscript.

---

## Demo

A self-contained demo and a smoke test, both offline (no IBL download):

```bash
python demo/demo_recovery.py     # ~2-3 min
python tests/test_smoke.py       # ~5 s   (also runs under: pytest tests/test_smoke.py)
```

- **`demo/demo_recovery.py`** runs the validated step-vs-ramp recovery engine on a tiny synthetic
  grid (FR = 20 Hz, N ∈ {40, 160}, negative-binomial emission, R = 10), using the bundled curated
  reaction-time pool `demo/window_pool.npy` (2074 RTs, float-only, no identifiers). This is a
  small, fast, **illustrative** synthetic run - **not** the full R = 40 grid behind the manuscript
  figures - so its ~0.60 / 0.70 recovery values are **not expected to equal the exact figure
  values**; it simply demonstrates the qualitative pattern: recovery is near chance for a single
  session (N = 40) and rises once trials are pooled (N = 160).
- **`tests/test_smoke.py`** independently re-derives the implemented Poisson-HMM forward
  log-likelihood and checks it against `hmmlearn.PoissonHMM.score()` from scratch, then asserts the
  three recorded forward-likelihood validation numbers (the Table S2 / Fig 1c fixtures:
  ≈2×10⁻¹⁴ brute-force, ≈4×10⁻¹² ramp-vs-hmmlearn, ≈2×10⁻¹³ step-vs-hmmlearn) reproduce to within
  machine precision - so the engine-correctness claim is independently checkable.

---

## Analysis parameters (replicate counts)

For the Methods statement, the recovery analyses behind the figures use these replicate counts (R):
realistic NB grid (Fig 2a, 5a) **R = 40**; Poisson-misspecification comparison (Fig 2b) **R = 40**;
population-arm gate / pre-flight (the ~0.57 AUC ≈ 0.24 SD threshold) **R = 50**; clean-Poisson
best-case map (Fig S1) **R = 60**; diffusion sweep (Fig 2c / Fig S2) **R = 40** (one cell at 80);
intermediacy sweep (Fig 2d / Fig S2) **R = 40**.

---

## Documentation

Detailed method and audit notes live in `docs/`: `robustness.md` (one-at-a-time sweeps),
`technical_audit.md` (adversarial audit + corrections), `audit_corrections.md`,
`ramp_validation.md` (forward-likelihood validation), `steinmetz_replication_feasibility.md`,
`repo_inventory.md`.

## License

BSD-3-Clause - see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the software archive (this repository) and the accompanying
manuscript.

**Software archive (this repository):**

> Yamasaki, R. (2026). *Movement controls bias decision-signal estimates across the brain:
> a calibrated, ground-truth test* (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20686415

**Manuscript:**

> Yamasaki, R. (2026). *Movement controls bias decision-signal estimates across the brain
> a calibrated, ground-truth test.* Preprint (Research Square, in review); https://doi.org/10.21203/rs.3.rs-10037105/v1



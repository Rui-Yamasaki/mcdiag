"""Human-readable report formatting for the calibration and recoverability results."""
from __future__ import annotations


def format_calibration(res):
    lines = []
    lines.append("MOVEMENT CONTROL CALIBRATION")
    lines.append("=" * 60)
    lines.append(f"blocks: {res.n_blocks}  cells/block: {_brief(res.n_cells_per_block)}  "
                 f"trials/block: {_brief(res.n_trials_per_block)}")
    lines.append(f"measured choice-to-movement correlation (realistic rho): {res.measured_rho:.3f}")
    lines.append(f"threshold signal injected: {res.threshold_d} SD per cell  "
                 f"(uncontrolled preserve-test AUC {res.uncontrolled_preserve_auc:.3f})")
    lines.append("")
    lines.append(f"{'control':10s} {'no-signal':>10s} {'preserve':>9s} {'retain':>7s} "
                 f"{'remove':>7s} {'valid':>6s}")
    for name, r in res.controls.items():
        retain = "n/a" if r.preserve_retain != r.preserve_retain else f"{r.preserve_retain:.2f}"
        remove = "n/a" if r.remove_auc != r.remove_auc else f"{r.remove_auc:.3f}"
        lines.append(f"{name:10s} {r.no_signal_auc:>10.3f} {r.preserve_auc:>9.3f} {retain:>7s} "
                     f"{remove:>7s} {('yes' if r.valid else 'no'):>6s}")
    lines.append("")
    if res.recommended:
        rec = res.controls[res.recommended]
        lines.append(f"RECOMMENDED CONTROL: {res.recommended}")
        lines.append(f"  valid controls ranked by movement removed (most first): "
                     f"{', '.join(res.valid_ranked)}")
        lines.append(f"  at the realistic rho ({res.measured_rho:.3f}) a threshold signal recovers "
                     f"to AUC {rec.realistic_auc:.3f} through this control")
    else:
        lines.append("RECOMMENDED CONTROL: none")
        lines.append(f"  {res.note}")
    lines.append("")
    lines.append("why each rejected or sub-optimal control is not recommended for this data:")
    for name, r in res.controls.items():
        if (not r.valid) or getattr(r, "under_removes", False):
            lines.append(f"  - {r.reason}")
    return "\n".join(lines)


def format_recoverability(res):
    lines = []
    lines.append("RECOVERABILITY CHECK")
    lines.append("=" * 60)
    lines.append(f"firing rate: {res.firing_rate_hz:g} Hz   trials: {res.n_trials}   "
                 f"simultaneous cells: {res.n_simultaneous_cells}")
    lines.append("")
    verdict = "DISTINGUISHABLE" if res.per_cell_distinguishable else "NOT distinguishable"
    lines.append(f"per-cell single-trial decision signal: {verdict}")
    lines.append(f"  achievable per-cell AUC {res.achievable_per_cell_auc:.3f} "
                 f"(5th pct {res.per_cell_auc_lo:.3f}) vs bar {res.per_cell_auc_bar:.3f}  "
                 f"(gap {res.per_cell_auc_gap:+.3f})")
    pop = "RECOVERABLE" if res.population_recoverable else "NOT recoverable"
    lines.append(f"population step-versus-ramp dynamics: {pop}")
    lines.append(f"  {res.n_simultaneous_cells} simultaneous cells vs "
                 f"{res.population_requirement}-cell bar  (gap {res.population_cell_gap:+d} cells)")
    lines.append("")
    lines.append(f"note: {res.note}")
    return "\n".join(lines)


def _brief(seq):
    if not seq:
        return "0"
    lo, hi = min(seq), max(seq)
    return f"{lo}" if lo == hi else f"{lo}-{hi}"

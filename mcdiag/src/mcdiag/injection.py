"""Choice-null backgrounds and the covariance-preserving signal injection.

This module is the reason the diagnostic is trustworthy. It builds synthetic test datasets whose
ONLY choice information is a signal we inject ourselves, of known size and known entanglement with
movement, and crucially the movement part of that signal is drawn from the USER's own movement
covariance rather than from a single high-variance direction.

Why that matters. A control can pass a naive calibration that injects movement along the leading
(high-variance) movement direction, yet still fail on real data because real movement confound
also lives in low-variance directions. A PCA-based control that keeps only high-variance movement
is the textbook case: it removes the high-variance test signal (so it looks valid) but leaves the
low-variance confound (so it under-removes in practice). By drawing the injected movement signal
from the full movement covariance (a random projection of the movement features, resampled each
repeat), this calibration exercises low-variance directions too, so under-removal is caught.

Construction. For a target movement entanglement rho in [0, 1] the per-trial carrier is

    s = rho * m_hat + sqrt(1 - rho^2) * q

where m_hat is a unit-variance random movement-subspace direction (covariance preserving) and q
is a unit-variance direction orthogonal to the movement subspace (the pure decision part). The
injected binary choice label is the sign of s about its median, so the label genuinely depends on
the movement-correlated part when rho is large. Each cell receives gain * s with the gain set so
the injected per-cell choice separation equals d standard deviations of that cell's background
activity, for every rho.
"""
from __future__ import annotations

import numpy as np


def movement_subspace(movement):
    """Standardize the raw movement features (NaN to zero, center, unit scale). Returns [n, p]."""
    M = np.nan_to_num(np.asarray(movement, dtype=float), nan=0.0)
    M = M - M.mean(0)
    sd = M.std(0)
    sd[sd < 1e-9] = 1.0
    return M / sd


def project_onto(M, x):
    """Least-squares projection of x onto [1, M]."""
    A = np.c_[np.ones(len(x)), M]
    beta = np.linalg.lstsq(A, x, rcond=None)[0]
    return A @ beta


def measured_choice_movement_corr(choice, movement):
    """Multiple correlation R between choice and its projection on the movement subspace.

    This is the data's actual choice-to-movement coupling, the realistic entanglement at which a
    genuine decision signal would be confounded. Returned as a value in [0, 1].
    """
    c = np.asarray(choice, dtype=float)
    c = c - c.mean()
    if c.std() < 1e-9:
        return 0.0
    M = movement_subspace(movement)
    fit = project_onto(M, c)
    r = np.corrcoef(c, fit)[0, 1]
    return float(abs(r)) if np.isfinite(r) else 0.0


def make_carrier(movement, rho, rng):
    """Unit-variance carrier s = rho * m_hat + sqrt(1 - rho^2) * q.

    m_hat is a covariance-preserving random movement-subspace direction (a random projection of
    the movement features), NOT the leading principal component. q is orthogonal to the movement
    subspace. Returns (s, realized_rho) where realized_rho is the achieved correlation between s
    and its own movement projection.
    """
    M = movement_subspace(movement)
    n, p = M.shape
    # covariance-preserving movement direction: random combination of movement features
    w = rng.standard_normal(p)
    m_hat = M @ w
    if m_hat.std() < 1e-12:                       # degenerate movement, fall back to noise
        m_hat = rng.standard_normal(n)
    m_hat = m_hat - m_hat.mean()
    m_hat = m_hat / (m_hat.std() + 1e-12)
    # decision direction orthogonal to the movement subspace
    g = rng.standard_normal(n)
    q = g - project_onto(M, g)
    q = q - q.mean()
    q = q / (q.std() + 1e-12)
    s = rho * m_hat + np.sqrt(max(1.0 - rho ** 2, 0.0)) * q
    s = s - s.mean()
    s = s / (s.std() + 1e-12)
    pm = project_onto(M, s)
    realized = float(np.corrcoef(s, pm)[0, 1]) if pm.std() > 1e-12 else 0.0
    return s, realized


def inject(activity, movement, d, rho, rng):
    """Inject a known choice signal into background activity.

    activity is [n_trials, n_cells] background (its real spike statistics are preserved), movement
    is [n_trials, n_features]. d is the injected per-cell choice separation in background-SD units.
    rho is the target movement entanglement. Returns (injected_activity, injected_choice_labels,
    realized_rho).
    """
    Xbg = np.asarray(activity, dtype=float)
    s, realized = make_carrier(movement, rho, rng)
    y = (s > np.median(s)).astype(int)
    delta_s = s[y == 1].mean() - s[y == 0].mean()         # separation of the carrier by its label
    sd = Xbg.std(0)
    sd[sd < 1e-9] = 1.0
    if d == 0.0 or abs(delta_s) < 1e-9:
        Xinj = Xbg.copy()
    else:
        gain = d * sd / delta_s                           # per-cell gain so separation equals d SD
        Xinj = Xbg + np.outer(s, gain)
    return Xinj, y, realized


def choice_null_background(activity, choice, rng):
    """Shuffle the choice labels so the background carries no genuine choice information, while
    preserving the neural activity statistics and the trial structure. Returns shuffled labels.

    The injection above does not require this (it reads its own labels off the carrier), but it is
    exposed because the no-signal test uses a background whose labels are unrelated to activity.
    """
    y = np.asarray(choice).copy()
    rng.shuffle(y)
    return y

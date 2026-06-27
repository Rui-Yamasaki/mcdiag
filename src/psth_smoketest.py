"""IBL public-data access smoke test -> PSTH (Tasks 4-6).

End-to-end, reproducible:
  4. Load spike times, spike clusters, and cluster brain-region assignments for
     one pinned probe insertion. Print #units, #spikes, duration, regions.
  5. Load the session trials table and report which columns exist.
  6. Build a stimulus-onset-aligned PSTH for a few units and save the figure to
     figures/psth_smoketest.png.

This streams ONLY small spike-sorting / trials products from the PUBLIC Open
Alyx instance. It never downloads raw AP/LFP voltage.

Run (uses the pinned default insertion):
    python src/psth_smoketest.py
Override the insertion:
    python src/psth_smoketest.py --pid <PROBE_INSERTION_UUID>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from brainbox.io.one import SessionLoader, SpikeSortingLoader  # noqa: E402

from ibl_one import DEFAULT_PID, PROJECT_ROOT, get_one  # noqa: E402

# --- PSTH parameters (fixed -> reproducible) --------------------------------
WINDOW = (-0.5, 1.0)       # seconds relative to stimulus onset
BIN_SIZE = 0.020           # 20 ms bins (within the requested 10-25 ms range)
N_UNITS = 5                # number of units to plot
FR_BAND = (1.0, 30.0)      # Hz: "reasonable firing rate" band for unit pick
FIG_PATH = PROJECT_ROOT / "figures" / "psth_smoketest.png"


def _get(obj, *names):
    """Fetch a field from a Bunch/dict or a DataFrame, trying several names."""
    for name in names:
        if hasattr(obj, "columns"):  # pandas DataFrame
            if name in obj.columns:
                return np.asarray(obj[name])
        elif name in obj:            # Bunch / dict
            return np.asarray(obj[name])
    raise KeyError(f"none of {names} found in {type(obj).__name__}")


def load_spikes(one, pid):
    """Return (ssl, eid, pname, spikes, clusters) for a probe insertion."""
    ssl = SpikeSortingLoader(pid=pid, one=one)
    # By default this loads pykilosort spikes/clusters/channels (small derived
    # products) -- NOT raw voltage.
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters = ssl.merge_clusters(spikes, clusters, channels)
    return ssl, ssl.eid, ssl.pname, spikes, clusters


def summarize(spikes, clusters):
    """Print Task-4 summary and return (duration, firing_rate_per_cluster)."""
    times = _get(spikes, "times")
    sclusters = _get(spikes, "clusters")
    acronym = _get(clusters, "acronym", "acronyms")

    n_clusters = len(acronym)
    n_spikes = times.size
    duration = float(times.max() - times.min())
    counts = np.bincount(sclusters, minlength=n_clusters)
    firing_rate = counts / duration
    regions = sorted(set(map(str, acronym)))

    print("\n========== Task 4: spike-sorting summary ==========")
    print(f"  # units (clusters in table) : {n_clusters}")
    print(f"  # units with >= 1 spike     : {int(np.count_nonzero(counts))}")
    print(f"  # spikes (total)            : {n_spikes:,}")
    print(f"  recording duration          : {duration:.1f} s "
          f"({duration / 60:.1f} min)")
    print(f"  # brain regions (acronyms)  : {len(regions)}")
    print(f"  brain regions present       : {', '.join(regions)}")
    return duration, firing_rate, acronym, counts


def select_units(firing_rate, acronym, counts, label=None):
    """Deterministically pick up to N_UNITS units for the PSTH.

    Strategy (deterministic for fixed data):
      * keep units in the FR_BAND with a real brain region and >=1 spike,
      * prefer curated "good" units (label == 1) when there are enough,
      * pick units evenly spaced across the firing-rate range so the figure
        shows a representative spread rather than only the fastest cells.
    """
    acronym = np.asarray([str(a) for a in acronym])
    real_region = ~np.isin(acronym, ["void", "root", ""])
    in_band = (firing_rate >= FR_BAND[0]) & (firing_rate <= FR_BAND[1])
    base = real_region & in_band & (counts > 0)

    mask = base
    if label is not None:
        good = base & (np.asarray(label) >= 1.0)  # IBL "good" quality units
        if int(good.sum()) >= N_UNITS:
            mask = good

    cand = np.where(mask)[0]
    if cand.size == 0:  # relax region constraint if needed
        cand = np.where(in_band & (counts > 0))[0]
    if cand.size == 0:
        return cand

    cand = cand[np.argsort(firing_rate[cand], kind="stable")]  # ascending
    if cand.size <= N_UNITS:
        return cand
    idx = np.unique(np.linspace(0, cand.size - 1, N_UNITS).round().astype(int))
    return cand[idx]


def compute_psth(spike_times, spike_clusters, unit_id, events):
    """Mean firing rate (Hz) per time bin, aligned to events, for one unit."""
    edges = np.arange(WINDOW[0], WINDOW[1] + BIN_SIZE / 2, BIN_SIZE)
    st = np.sort(spike_times[spike_clusters == unit_id])
    total = np.zeros(edges.size - 1)
    for ev in events:
        i0 = np.searchsorted(st, ev + WINDOW[0])
        i1 = np.searchsorted(st, ev + WINDOW[1])
        hist, _ = np.histogram(st[i0:i1] - ev, bins=edges)
        total += hist
    psth_hz = total / (events.size * BIN_SIZE)
    centers = edges[:-1] + BIN_SIZE / 2
    return centers, psth_hz


def load_trials(one, eid):
    """Load the trials table (Task 5) and report available columns."""
    sl = SessionLoader(one=one, eid=eid)
    sl.load_trials()
    trials = sl.trials  # pandas DataFrame

    print("\n========== Task 5: trials table ==========")
    print(f"  # trials : {len(trials)}")
    print(f"  columns  : {list(trials.columns)}")
    wanted = [
        "stimOn_times", "firstMovement_times", "choice",
        "contrastLeft", "contrastRight", "feedbackType",
        "probabilityLeft",
    ]
    print("  requested-column check:")
    for col in wanted:
        present = col in trials.columns
        print(f"    {'OK ' if present else 'MISSING'}  {col}")
    return trials


def make_figure(spikes, clusters, units, firing_rate, acronym, events,
                eid, pname, pid):
    times = _get(spikes, "times")
    sclusters = _get(spikes, "clusters")
    acronym = np.asarray([str(a) for a in acronym])

    n = len(units)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.0 * n), sharex=True)
    if n == 1:
        axes = [axes]

    print("\n========== Task 6: PSTH units ==========")
    for ax, uid in zip(axes, units):
        centers, psth = compute_psth(times, sclusters, uid, events)
        ax.bar(centers, psth, width=BIN_SIZE, align="center",
               color="#3b7dd8", edgecolor="none")
        ax.axvline(0.0, color="k", lw=1, ls="--")
        ax.set_ylabel("Hz")
        ax.set_title(
            f"unit {uid}  |  {acronym[uid]}  |  {firing_rate[uid]:.1f} Hz mean",
            fontsize=9, loc="left",
        )
        ax.margins(x=0)
        print(f"  unit {uid:<6}  region={acronym[uid]:<8}  "
              f"mean_fr={firing_rate[uid]:.2f} Hz")

    axes[-1].set_xlabel("time from stimulus onset (s)")
    fig.suptitle(
        f"PSTH smoke test  |  stimOn-aligned  |  {len(events)} trials\n"
        f"eid={eid}  probe={pname}\npid={pid}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nSaved figure -> {FIG_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", default=DEFAULT_PID,
                        help="probe insertion UUID (defaults to pinned PID)")
    args = parser.parse_args()

    one = get_one()
    print(f"Connected to: {one.alyx.base_url}")
    print(f"Using probe insertion pid: {args.pid}")

    ssl, eid, pname, spikes, clusters = load_spikes(one, args.pid)
    print(f"  session eid : {eid}")
    print(f"  probe name  : {pname}")

    duration, firing_rate, acronym, counts = summarize(spikes, clusters)
    trials = load_trials(one, eid)

    events = _get(trials, "stimOn_times").astype(float)
    events = events[np.isfinite(events)]

    try:
        label = _get(clusters, "label")
    except KeyError:
        label = None
    units = select_units(firing_rate, acronym, counts, label=label)
    if len(units) == 0:
        raise SystemExit("No units passed the firing-rate selection.")
    make_figure(spikes, clusters, units, firing_rate, acronym, events,
                eid, pname, args.pid)


if __name__ == "__main__":
    main()

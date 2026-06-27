"""Task 3 smoke test: confirm we can reach the IBL PUBLIC data.

Connects ONE to the open Alyx instance and lists a handful of Brain Wide Map
sessions / probe insertions that have spike-sorting products available.

Run:
    python src/list_sessions.py
"""
from __future__ import annotations

from ibl_one import BWM_PROJECT, get_one


def main(n: int = 5) -> None:
    one = get_one()
    print(f"Connected to: {one.alyx.base_url}\n")

    # Sessions in the Brain Wide Map that have spike times available.
    eids = one.search(project=BWM_PROJECT, datasets="spikes.times.npy")
    print(f"Brain Wide Map sessions with spikes.times.npy: {len(eids)} total")
    print(f"First {n} session eids:")
    for eid in eids[:n]:
        print("   ", eid)

    # Probe insertions (pids) are what SpikeSortingLoader consumes.
    pids = one.search_insertions(project=BWM_PROJECT, datasets="spikes.times.npy")
    print(f"\nBrain Wide Map probe insertions with spikes.times.npy: {len(pids)} total")
    print(f"First {n} insertion pids:")
    for pid in pids[:n]:
        eid, pname = one.pid2eid(pid)
        print(f"    {pid}  ({pname})  eid={eid}")


if __name__ == "__main__":
    main()

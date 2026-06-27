"""Shared connection helper for the IBL public (open) Alyx instance.

We connect to the PUBLIC, read-only Open Alyx server and stream only the small
spike-sorting / trials products. We never download raw AP/LFP voltage data.

The ONE cache is pointed at the project-local ``data/`` directory (gitignored)
so every download is reproducible and self-contained inside the repo.
"""
from __future__ import annotations

from pathlib import Path

from one.api import ONE

# --- paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ONE_CACHE_DIR = DATA_DIR / "ONE"

# --- public Open Alyx connection details ------------------------------------
OPENALYX_BASE_URL = "https://openalyx.internationalbrainlab.org"
# IBL's PUBLIC, read-only password for the open Brain-Wide Map release — NOT a private secret.
# It is the credential IBL publishes for anyone to access the open Alyx instance; see the ONE
# docs: https://int-brain-lab.github.io/ONE/  (intentionally hard-coded, not loaded from env).
OPENALYX_PASSWORD = "international"

# Brain Wide Map project tag used by ONE.search(project=...)
BWM_PROJECT = "brainwide"

# Pinned probe insertion (pid) for the reproducible PSTH smoke test.
# Chosen once (see README) for good ephys coverage; override with --pid.
# churchlandlab_ucla/MFD_09/2023-10-19/001 probe00 -- 1557 units, 16 regions,
# 80.9M spikes, 120 min, 569 trials.
DEFAULT_PID = "8c732bf2-639d-496c-bf82-464bc9c2d54b"


def get_one() -> ONE:
    """Return a ONE client bound to the public Open Alyx instance.

    Uses ``silent=True`` so it never prompts, and stores the data cache inside
    the repo's ``data/`` directory.
    """
    ONE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ONE(
        base_url=OPENALYX_BASE_URL,
        password=OPENALYX_PASSWORD,
        silent=True,
        cache_dir=str(ONE_CACHE_DIR),
    )


if __name__ == "__main__":
    one = get_one()
    print("Connected to:", one.alyx.base_url)
    print("Cache dir   :", one.cache_dir)

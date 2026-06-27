"""Launcher for the recovery simulation.

Running ``phase1_recovery.py`` directly puts its functions in ``__main__``, which
joblib's loky workers cannot re-import on Windows ("Can't get attribute ... on
__main__"). Importing ``main`` here keeps every worker function in the importable
``phase1_recovery`` module, so parallel sims pickle cleanly.

    python src/run_recovery.py --stage validate
    python src/run_recovery.py --stage sweep
"""
from phase1_recovery import main

if __name__ == "__main__":
    main()

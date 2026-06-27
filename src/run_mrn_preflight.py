"""Launcher for the MRN population step-vs-ramp recovery pre-flight.

Like run_recovery.py / run_hardened.py: importing main keeps every worker function in
the importable phase2_mrn_recovery_preflight module, so joblib's loky workers can
re-import them on Windows (functions defined in __main__ cannot be unpickled by workers).

    python src/run_mrn_preflight.py --stage all
    python src/run_mrn_preflight.py --stage all --quick
"""
from phase2_mrn_recovery_preflight import main

if __name__ == "__main__":
    main()

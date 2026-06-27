"""Launcher for the hardened recovery sweep.

Like run_recovery.py: importing main keeps every worker function in the importable
phase1_recovery_hardened module, so joblib's loky workers can re-import them on
Windows (functions defined in __main__ cannot be unpickled by workers).

    python src/run_hardened.py --stage all
"""
from phase1_recovery_hardened import main

if __name__ == "__main__":
    main()

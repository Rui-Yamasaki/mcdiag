"""Launcher for the Steinmetz population step-vs-ramp recovery pre-flight.

Like run_mrn_preflight.py: importing main keeps every worker function in the importable
steinmetz_population_preflight module, so joblib's loky workers can re-import them on
Windows (functions defined in __main__ cannot be unpickled by workers).

    python src/run_steinmetz_preflight.py --stage all
    python src/run_steinmetz_preflight.py --stage all --quick
"""
from steinmetz_population_preflight import main

if __name__ == "__main__":
    main()

"""Launcher for audit_ramp_validate (keeps worker fns importable for loky on Windows)."""
from audit_ramp_validate import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the quantitative Python sources bundled with this kit."""
from __future__ import annotations
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYROOT = ROOT / "source" / "python"
sys.path.insert(0, str(PYROOT))
SCRIPTS = [
    "f04_primary_paired_counts.py",
    "f05_sensitivity_intervals.py",
    "f06_gaussian_calibration.py",
    "f07_serial_dependence.py",
    "s01_candidate_identity.py",
    "s02_robustness_mechanisms.py",
]

def main() -> None:
    fig_dir = PYROOT / "figures"
    for script in SCRIPTS:
        print(f"running {script}")
        runpy.run_path(str(fig_dir / script), run_name="__main__")

if __name__ == "__main__":
    main()

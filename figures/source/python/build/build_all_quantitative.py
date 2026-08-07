
import runpy
from pathlib import Path
FIG_DIR = Path(__file__).resolve().parents[1] / 'figures'
SCRIPTS=['f04_primary_paired_counts.py','f05_sensitivity_intervals.py','f06_gaussian_calibration.py','f07_serial_dependence.py','s01_candidate_identity.py','s02_robustness_mechanisms.py']

def main():
    for s in SCRIPTS:
        runpy.run_path(str(FIG_DIR / s), run_name='__main__')
if __name__=='__main__': main()

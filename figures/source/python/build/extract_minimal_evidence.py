
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / 'data' / 'evidence_extracts'
EVIDENCE_ZIP = Path(r"/mnt/data/PCPI_SIMPAT_GPT56_LOAD_BEARING_EVIDENCE_2026-08-04(1).zip")
EXTRACT_MAP = {
  "application_primary_raw.csv": "PCPI_SIMPAT_GPT56_LOAD_BEARING_EVIDENCE_2026-08-04/03_LOAD_BEARING_EXPERIMENTS/PHASE_07_APPLICATION_CONFIRMATORY/application_primary_raw.csv",
  "application_primary_summary.json": "PCPI_SIMPAT_GPT56_LOAD_BEARING_EVIDENCE_2026-08-04/03_LOAD_BEARING_EXPERIMENTS/PHASE_07_APPLICATION_CONFIRMATORY/application_primary_summary.json",
  "application_primary_candidate_distribution.csv": "PCPI_SIMPAT_GPT56_LOAD_BEARING_EVIDENCE_2026-08-04/03_LOAD_BEARING_EXPERIMENTS/PHASE_07_APPLICATION_CONFIRMATORY/application_primary_candidate_distribution.csv",
  "application_sensitivity_summary.json": "PCPI_SIMPAT_GPT56_LOAD_BEARING_EVIDENCE_2026-08-04/03_LOAD_BEARING_EXPERIMENTS/PHASE_07_APPLICATION_CONFIRMATORY/application_sensitivity_summary.json",
  "null_calibration_summary.json": "PCPI_SIMPAT_GPT56_LOAD_BEARING_EVIDENCE_2026-08-04/03_LOAD_BEARING_EXPERIMENTS/PHASE_08_CALIBRATION_ROBUSTNESS/null_calibration_summary.json",
  "robustness_summary.json": "PCPI_SIMPAT_GPT56_LOAD_BEARING_EVIDENCE_2026-08-04/03_LOAD_BEARING_EXPERIMENTS/PHASE_08_CALIBRATION_ROBUSTNESS/robustness_summary.json"
}

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(EVIDENCE_ZIP) as z:
        for out_name, in_name in EXTRACT_MAP.items():
            (DATA_DIR / out_name).write_bytes(z.read(in_name))
            print(f'extracted {out_name}')
if __name__=='__main__': main()

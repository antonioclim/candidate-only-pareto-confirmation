"""Aggregate generated calibration and robustness chunks."""
from pathlib import Path
import csv
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcpi_candidate_tas.calibration import (
    bonferroni_core_upper,
    one_sided_clopper_pearson_upper,
)

OUTPUT = ROOT / "generated/calibration"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str) -> bool:
    return value.lower() == "true"


def main() -> None:
    summaries = []
    for path in sorted(OUTPUT.glob("NULL-*.csv")):
        rows = read_csv(path)
        false_count = sum(as_bool(row["false_certified"]) for row in rows)
        trials = len(rows)
        rate = false_count / trials
        summaries.append({
            "cell_id": path.stem,
            "method": rows[0]["method"],
            "replications": trials,
            "false_certifications": false_count,
            "false_certification_rate": rate,
            "monte_carlo_se": math.sqrt(rate * (1.0 - rate) / trials),
            "one_sided_upper_95": one_sided_clopper_pearson_upper(
                false_count, trials, 0.95
            ),
            "familywise_adjusted_upper_95": bonferroni_core_upper(
                false_count, trials
            ),
            "gate_threshold": 0.065,
            "gate_passed": bonferroni_core_upper(
                false_count, trials
            ) <= 0.065,
        })
    if summaries:
        write_csv(OUTPUT / "calibration_summary.csv", summaries)

    robustness = []
    for path in sorted(OUTPUT.glob("ROB-*.csv")):
        rows = read_csv(path)
        for method in ("coordinate", "hotelling", "hybrid"):
            selected = [row for row in rows if row["method"] == method]
            false_count = sum(as_bool(row["false_certified"]) for row in selected)
            trials = len(selected)
            rate = false_count / trials
            robustness.append({
                "cell_id": path.stem,
                "method": method,
                "replications": trials,
                "false_certifications": false_count,
                "false_certification_rate": rate,
                "one_sided_upper_95": one_sided_clopper_pearson_upper(
                    false_count, trials, 0.95
                ),
            })
    if robustness:
        write_csv(OUTPUT / "robustness_summary.csv", robustness)
    print(json.dumps({
        "calibration_cells": len(summaries),
        "robustness_method_cells": len(robustness),
    }, indent=2))


if __name__ == "__main__":
    main()

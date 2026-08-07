
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def test_figure_registry_complete():
    reg = json.loads((FIG / "figure_registry.json").read_text(encoding="utf-8"))
    assert [f["id"] for f in reg["figures"]] == ["F1","F2","F3","F4","F5","F6","F7","GA","S1","S2"]


def test_exact_ooxml_sources_present():
    for fid in ["F1","F2","F3","F4","F5","F6","F7","GA","S1","S2"]:
        assert (FIG / "source" / "exact_ooxml" / fid / "[Content_Types].xml").is_file()


def test_candidate_distribution_total():
    path = FIG / "data" / "evidence_extracts" / "application_primary_candidate_distribution.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        assert sum(int(row["count"]) for row in csv.DictReader(handle)) == 400


def test_calibration_and_robustness_load_bearing_values():
    null = json.loads((FIG / "data" / "evidence_extracts" / "null_calibration_summary.json").read_text(encoding="utf-8"))
    assert len(null) == 12
    assert all(row["gate_passed"] for row in null)
    robust = json.loads((FIG / "data" / "evidence_extracts" / "robustness_summary.json").read_text(encoding="utf-8"))
    expected = {"coordinate": 0.306, "hotelling": 0.233, "hybrid": 0.282}
    for method, value in expected.items():
        got = next(row["false_certification_rate"] for row in robust if row["cell_id"] == "ROB-06" and row["method"] == method)
        assert got == value

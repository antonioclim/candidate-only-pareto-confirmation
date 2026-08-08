#!/usr/bin/env python3
"""Validate figure sources, evidence extracts, reference hashes and built artefacts."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REG = json.loads((ROOT / "figure_registry.json").read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    failures = []
    for fig in REG["figures"]:
        for key in ["reference_pptx", "reference_odg", "reference_png"]:
            p = ROOT / fig[key]
            if not p.exists(): failures.append(f"missing:{p}")
        ooxml = ROOT / fig["exact_ooxml_dir"]
        if not (ooxml / "[Content_Types].xml").exists(): failures.append(f"bad_ooxml:{ooxml}")
        out = ROOT / "outputs" / "journal_png_600dpi" / f"{fig['id']}.png"
        if out.exists():
            with Image.open(out) as im:
                if im.size != (fig["target_png_width"], fig["target_png_height"]):
                    failures.append(f"size:{fig['id']}:{im.size}")
    # load-bearing data checks
    data = ROOT / "data" / "evidence_extracts"
    with (data / "application_primary_raw.csv").open(newline="", encoding="utf-8") as f:
        if sum(1 for _ in csv.DictReader(f)) != 400: failures.append("primary_rows")
    candidate = list(csv.DictReader((data / "application_primary_candidate_distribution.csv").open(newline="", encoding="utf-8")))
    if sum(int(r["count"]) for r in candidate) != 400: failures.append("candidate_total")
    null = json.loads((data / "null_calibration_summary.json").read_text(encoding="utf-8"))
    if len(null) != 12 or not all(r["gate_passed"] for r in null): failures.append("null_calibration")
    robust = json.loads((data / "robustness_summary.json").read_text(encoding="utf-8"))
    expected = {"coordinate":0.306,"hotelling":0.233,"hybrid":0.282}
    for method, value in expected.items():
        got = next(r["false_certification_rate"] for r in robust if r["cell_id"]=="ROB-06" and r["method"]==method)
        if abs(got-value)>1e-12: failures.append(f"rob06:{method}")
    report = {"valid": not failures, "failures": failures, "figures": len(REG["figures"])}
    (ROOT / "qa" / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures: raise SystemExit(1)

if __name__ == "__main__":
    main()

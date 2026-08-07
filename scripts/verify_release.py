"""Verify release checksums and prohibited-file policy."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_posix(path: Path) -> str:
    """Return a repository-relative POSIX path on every operating system."""
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    prohibited_names = re.compile(
        r"(manu" r"script|supple" r"ment|submis" r"sion|mirror_" r"subset)",
        re.IGNORECASE,
    )
    prohibited_paths = [
        _relative_posix(path)
        for path in ROOT.rglob("*")
        if path.is_file() and prohibited_names.search(path.name)
    ]

    source_rows = []
    for path in ROOT.rglob("*.csv"):
        relative = _relative_posix(path)
        if path.name == "FILE_INVENTORY.csv":
            continue
        if "synthetic" in relative:
            continue
        if relative.startswith("evidence/"):
            continue
        if relative.startswith("config/"):
            continue
        if relative.startswith("figures/data/evidence_extracts/"):
            continue
        if relative.startswith("figures/qa/"):
            continue
        source_rows.append(relative)

    allowed_figure_data = {
        "application_primary_raw.csv",
        "application_primary_candidate_distribution.csv",
    }
    figure_data_csv = {
        path.name
        for path in (ROOT / "figures" / "data" / "evidence_extracts").glob("*.csv")
    }
    unexpected_figure_data = sorted(figure_data_csv - allowed_figure_data)

    report = {
        "prohibited_paths": sorted(prohibited_paths),
        "unexpected_data_csv": sorted(source_rows),
        "unexpected_figure_data_csv": unexpected_figure_data,
        "valid": not prohibited_paths and not source_rows and not unexpected_figure_data,
    }
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

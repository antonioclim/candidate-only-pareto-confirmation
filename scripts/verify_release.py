"""Verify release manifests and prohibited-file policy."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "FILE_INVENTORY.csv"
CHECKSUM_PATH = ROOT / "SHA256SUMS.txt"
ROOT_MANIFESTS = {"FILE_INVENTORY.csv", "SHA256SUMS.txt"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_posix(path: Path) -> str:
    """Return a repository-relative POSIX path on every operating system."""
    return path.relative_to(ROOT).as_posix()


def _safe_repository_path(relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe repository path: {relative!r}")
    candidate = ROOT.joinpath(*pure.parts).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {relative!r}") from exc
    return candidate


def _load_inventory() -> tuple[dict[str, tuple[int, str]], list[str]]:
    records: dict[str, tuple[int, str]] = {}
    errors: list[str] = []
    if not INVENTORY_PATH.is_file():
        return records, ["missing FILE_INVENTORY.csv"]

    with INVENTORY_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "bytes", "sha256"]:
            errors.append(
                "FILE_INVENTORY.csv header must be exactly path,bytes,sha256"
            )
            return records, errors
        for line_number, row in enumerate(reader, start=2):
            relative = row["path"]
            if relative in records:
                errors.append(f"duplicate inventory path at line {line_number}: {relative}")
                continue
            try:
                size = int(row["bytes"])
            except ValueError:
                errors.append(f"invalid byte count at line {line_number}: {row['bytes']!r}")
                continue
            checksum = row["sha256"]
            if size < 0:
                errors.append(f"negative byte count at line {line_number}: {relative}")
                continue
            if not SHA256_PATTERN.fullmatch(checksum):
                errors.append(f"invalid SHA-256 at line {line_number}: {relative}")
                continue
            try:
                _safe_repository_path(relative)
            except ValueError as exc:
                errors.append(f"line {line_number}: {exc}")
                continue
            records[relative] = (size, checksum)
    return records, errors


def _load_checksums() -> tuple[dict[str, str], list[str]]:
    records: dict[str, str] = {}
    errors: list[str] = []
    if not CHECKSUM_PATH.is_file():
        return records, ["missing SHA256SUMS.txt"]

    for line_number, line in enumerate(
        CHECKSUM_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if "  " not in line:
            errors.append(f"malformed checksum line {line_number}")
            continue
        checksum, relative = line.split("  ", 1)
        if relative in records:
            errors.append(f"duplicate checksum path at line {line_number}: {relative}")
            continue
        if not SHA256_PATTERN.fullmatch(checksum):
            errors.append(f"invalid SHA-256 at line {line_number}: {relative}")
            continue
        try:
            _safe_repository_path(relative)
        except ValueError as exc:
            errors.append(f"line {line_number}: {exc}")
            continue
        records[relative] = checksum
    return records, errors


def _git_archive_records() -> dict[str, tuple[int, str]] | None:
    """Return canonical committed bytes, independent of checkout line endings."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "archive", "--format=tar", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    records: dict[str, tuple[int, str]] = {}
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: extracted.read(1 << 20), b""):
                size += len(block)
                digest.update(block)
            records[member.name] = (size, digest.hexdigest())
    return records


def _filesystem_record(relative: str) -> tuple[int, str] | None:
    path = _safe_repository_path(relative)
    if not path.is_file():
        return None
    return path.stat().st_size, sha256_file(path)


def _verify_manifests() -> dict[str, object]:
    inventory, inventory_errors = _load_inventory()
    checksums, checksum_format_errors = _load_checksums()
    errors = [*inventory_errors, *checksum_format_errors]

    inventory_paths = set(inventory)
    checksum_paths = set(checksums)
    for relative in sorted(inventory_paths - checksum_paths):
        errors.append(f"missing from SHA256SUMS.txt: {relative}")
    for relative in sorted(checksum_paths - inventory_paths):
        errors.append(f"missing from FILE_INVENTORY.csv: {relative}")

    canonical = _git_archive_records()
    source = "git_archive" if canonical is not None else "filesystem"
    if canonical is not None:
        expected_tracked = inventory_paths | ROOT_MANIFESTS
        canonical_paths = set(canonical)
        for relative in sorted(expected_tracked - canonical_paths):
            errors.append(f"manifest path is absent from committed tree: {relative}")
        for relative in sorted(canonical_paths - expected_tracked):
            errors.append(f"committed path is absent from release manifests: {relative}")

    for relative in sorted(inventory_paths & checksum_paths):
        expected_size, inventory_checksum = inventory[relative]
        checksum_checksum = checksums[relative]
        if inventory_checksum != checksum_checksum:
            errors.append(f"manifest checksum disagreement: {relative}")
            continue

        actual = canonical.get(relative) if canonical is not None else _filesystem_record(relative)
        if actual is None:
            errors.append(f"manifested file is missing: {relative}")
            continue
        actual_size, actual_checksum = actual
        if actual_size != expected_size:
            errors.append(
                f"size mismatch: {relative}: expected {expected_size}, got {actual_size}"
            )
        if actual_checksum != inventory_checksum:
            errors.append(
                f"SHA-256 mismatch: {relative}: expected {inventory_checksum}, "
                f"got {actual_checksum}"
            )

    return {
        "inventory_records": len(inventory),
        "checksum_records": len(checksums),
        "verification_source": source,
        "errors": errors,
        "valid": not errors,
    }


def _verify_policy() -> dict[str, object]:
    prohibited_names = re.compile(
        r"(manu" r"script|supple" r"ment|submis" r"sion|mirror_" r"subset)",
        re.IGNORECASE,
    )
    prohibited_paths = [
        _relative_posix(path)
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(ROOT).parts
        and prohibited_names.search(path.name)
    ]

    unexpected_csv: list[str] = []
    for path in ROOT.rglob("*.csv"):
        relative = _relative_posix(path)
        if ".git" in path.relative_to(ROOT).parts:
            continue
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
        unexpected_csv.append(relative)

    allowed_figure_data = {
        "application_primary_raw.csv",
        "application_primary_candidate_distribution.csv",
    }
    figure_data_csv = {
        path.name
        for path in (ROOT / "figures" / "data" / "evidence_extracts").glob("*.csv")
    }
    unexpected_figure_data = sorted(figure_data_csv - allowed_figure_data)

    return {
        "prohibited_paths": sorted(prohibited_paths),
        "unexpected_data_csv": sorted(unexpected_csv),
        "unexpected_figure_data_csv": unexpected_figure_data,
        "valid": not prohibited_paths
        and not unexpected_csv
        and not unexpected_figure_data,
    }


def main() -> None:
    manifests = _verify_manifests()
    policy = _verify_policy()
    report = {
        "manifests": manifests,
        "policy": policy,
        "valid": bool(manifests["valid"] and policy["valid"]),
    }
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

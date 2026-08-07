"""Adapter for a user-supplied OPSD 2020-10-06 file.

The upstream dataset is not redistributed. A user-supplied file may be
checked against the pinned SHA-256 value before extraction.
"""
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
from datetime import datetime

OFFICIAL_VERSION = "2020-10-06"
OFFICIAL_DOI = "10.25832/time_series/2020-10-06"
OFFICIAL_FILENAME = "opsd_time_series_60min_singleindex_2020-10-06.csv"
OFFICIAL_SHA256 = "6A7F2BC571314CBF9C321CC03437691CD4BE95C3A6F075E60FF99E8035C704C8"
PRICE_COLUMN = "DK_1_price_day_ahead"
LOAD_COLUMN = "DK_1_load_actual_entsoe_transparency"
TIMESTAMP_COLUMN = "utc_timestamp"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_official_source(path: Path | str) -> dict:
    source = Path(path)
    if not source.exists():
        return {
            "exists": False,
            "valid": False,
            "path": str(source),
            "reason": "file_missing",
        }
    observed = sha256_file(source)
    return {
        "exists": True,
        "valid": observed == OFFICIAL_SHA256,
        "path": str(source),
        "sha256": observed,
        "expected_sha256": OFFICIAL_SHA256,
        "reason": "ok" if observed == OFFICIAL_SHA256 else "sha256_mismatch",
    }


def extract_official_market_year(
    path: Path | str,
    *,
    verify_hash: bool = True,
) -> list[dict]:
    source = Path(path)
    if verify_hash:
        verification = verify_official_source(source)
        if not verification["valid"]:
            raise ValueError(verification["reason"])
    rows = []
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {TIMESTAMP_COLUMN, PRICE_COLUMN, LOAD_COLUMN}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("required DK1 columns are absent")
        for row in reader:
            timestamp = row[TIMESTAMP_COLUMN]
            if not timestamp.startswith("2019-"):
                continue
            if row[PRICE_COLUMN] in {"", "nan"} or row[LOAD_COLUMN] in {"", "nan"}:
                continue
            moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            split = "development" if moment.month <= 6 else "confirmation"
            rows.append(
                {
                    "utc_timestamp": timestamp,
                    "split": split,
                    "price_eur_per_mwh": float(row[PRICE_COLUMN]),
                    "load_mwh": float(row[LOAD_COLUMN]),
                }
            )
    if not rows:
        raise ValueError("no complete 2019 DK1 rows extracted")
    return rows


def chronological_split(
    rows: list[dict],
    development_fraction: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    """Sort timestamped rows and create a deterministic chronological split."""

    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must lie in (0,1)")
    if len(rows) < 2:
        raise ValueError("at least two rows are required")
    if any("utc_timestamp" not in row for row in rows):
        raise ValueError("utc_timestamp is required")
    ordered = sorted(rows, key=lambda row: str(row["utc_timestamp"]))
    cut = int(round(len(ordered) * development_fraction))
    cut = min(max(cut, 1), len(ordered) - 1)
    return ordered[:cut], ordered[cut:]

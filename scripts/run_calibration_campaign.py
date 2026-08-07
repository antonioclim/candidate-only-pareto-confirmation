"""Execute the deterministic calibration and robustness campaigns."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import csv
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcpi_candidate_tas.calibration import (
    CalibrationDesign,
    generate_exact_null_path,
    generate_robustness_path,
    sequential_method_diagnostics,
)

CONFIG = ROOT / "config"
OUTPUT = ROOT / "generated/calibration"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_cell(kind: str, cell: dict, replications: int) -> tuple[str, list[dict], dict]:
    design = CalibrationDesign()
    cell_index = int(cell["cell_id"].split("-")[1])
    rows: list[dict] = []
    started = time.perf_counter()
    for replication in range(replications):
        if kind == "calibration":
            seed = 3_000_000 + cell_index * 3_000 + replication
            path = generate_exact_null_path(
                seed,
                int(cell["dimension"]),
                cell["covariance"],
                count=design.sample_cap,
                easy_effect=design.easy_effect,
                negative_coordinate=design.easy_negative_coordinate,
            )
            methods = (cell["method"],)
        else:
            seed = 10_000_000 + cell_index * 1_000 + replication
            path = generate_robustness_path(
                seed,
                cell["data_generating_mechanism"],
                count=design.sample_cap,
                dimension=design.robustness_dimension,
                easy_effect=design.easy_effect,
                negative_coordinate=design.easy_negative_coordinate,
            )
            methods = ("coordinate", "hotelling", "hybrid")
        diagnostics = sequential_method_diagnostics(
            path,
            design.delta,
            methods=methods,
            min_count=design.minimum_count,
            max_count=design.sample_cap,
            check_every=design.stopping_grid,
        )
        for method, result in diagnostics.items():
            rows.append({
                "campaign": kind,
                "cell_id": cell["cell_id"],
                "replication": replication,
                "seed": seed,
                "method": method,
                "false_certified": result["certified"],
                "stopping_count": result["stopping_count"],
                "boundary_ever_crossed": result["boundary_ever_crossed"],
                "easy_controls_jointly_crossed": result[
                    "easy_controls_jointly_crossed"
                ],
                "full_rank": result["final_full_rank_all_challengers"],
            })
    manifest = {
        "campaign": kind,
        "cell_id": cell["cell_id"],
        "replication_paths": replications,
        "method_rows": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return cell["cell_id"], rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        choices=("calibration", "robustness", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--replications",
        type=int,
        default=None,
        help="Optional small smoke-test override.",
    )
    args = parser.parse_args()

    tasks: list[tuple[str, dict, int]] = []
    if args.campaign in {"calibration", "all"}:
        for cell in read_csv(CONFIG / "calibration_cells.csv"):
            tasks.append((
                "calibration",
                cell,
                args.replications or int(cell["replications"]),
            ))
    if args.campaign in {"robustness", "all"}:
        for cell in read_csv(CONFIG / "robustness_cells.csv"):
            tasks.append((
                "robustness",
                cell,
                args.replications or int(cell["replications"]),
            ))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifests = []
    with ProcessPoolExecutor(
        max_workers=max(1, min(args.workers, len(tasks)))
    ) as executor:
        futures = {
            executor.submit(run_cell, *task): task[1]["cell_id"]
            for task in tasks
        }
        for future in as_completed(futures):
            expected = futures[future]
            cell_id, rows, manifest = future.result()
            if cell_id != expected:
                raise RuntimeError("worker returned a different cell")
            write_csv(OUTPUT / f"{cell_id}.csv", rows)
            (OUTPUT / f"{cell_id}.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            manifests.append(manifest)
            print(json.dumps(manifest))

    (OUTPUT / "execution_record.json").write_text(
        json.dumps(
            {"cells": len(manifests), "manifests": sorted(
                manifests, key=lambda item: item["cell_id"]
            )},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

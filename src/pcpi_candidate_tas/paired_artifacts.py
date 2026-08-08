"""Strict paired-certificate artefact with raw-data binding and replay."""
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from jsonschema import Draft202012Validator

from ._version import __version__
from .paired import evaluate_archive


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_identity_contract(
    candidate_id, challenger_ids, objective_names
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id must be a non-empty string")
    challengers = tuple(challenger_ids)
    objectives = tuple(objective_names)
    if not challengers or any(not isinstance(x, str) or not x for x in challengers):
        raise ValueError("challenger_ids must be non-empty strings")
    if len(set(challengers)) != len(challengers):
        raise ValueError("challenger_ids must be unique")
    if candidate_id in challengers:
        raise ValueError("candidate/challenger collision")
    if not objectives or any(not isinstance(x, str) or not x for x in objectives):
        raise ValueError("objective_names must be non-empty strings")
    if len(set(objectives)) != len(objectives):
        raise ValueError("objective_names must be unique")
    return challengers, objectives


def _validate_differences(
    differences, challenger_ids, objective_names
) -> np.ndarray:
    array = np.asarray(differences, dtype=float)
    if array.ndim != 3:
        raise ValueError("differences must be challenger-by-count-by-objective")
    if array.shape[0] != len(challenger_ids):
        raise ValueError("challenger count mismatch")
    if array.shape[1] < 2:
        raise ValueError("at least two paired scenarios are required")
    if array.shape[2] != len(objective_names):
        raise ValueError("objective count mismatch")
    if not np.all(np.isfinite(array)):
        raise ValueError("differences must be finite")
    return array


def write_raw_paired(path, differences, challenger_ids, objective_names):
    challengers = tuple(challenger_ids)
    objectives = tuple(objective_names)
    if not challengers or len(set(challengers)) != len(challengers):
        raise ValueError("unique challenger_ids are required")
    if not objectives or len(set(objectives)) != len(objectives):
        raise ValueError("unique objective_names are required")
    array = _validate_differences(differences, challengers, objectives)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["challenger_id", "scenario_index", *objectives])
        for i, challenger_id in enumerate(challengers):
            for scenario, row in enumerate(array[i]):
                writer.writerow([
                    challenger_id,
                    scenario,
                    *[format(float(value), ".17g") for value in row],
                ])


def read_raw_paired(path, challenger_ids, objective_names):
    challengers = tuple(challenger_ids)
    objectives = tuple(objective_names)
    rows = {challenger_id: [] for challenger_id in challengers}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            "challenger_id", "scenario_index", *objectives
        ]:
            raise ValueError("raw columns mismatch")
        for record in reader:
            challenger_id = record["challenger_id"]
            if challenger_id not in rows:
                raise ValueError("unknown challenger")
            rows[challenger_id].append((
                int(record["scenario_index"]),
                [float(record[objective]) for objective in objectives],
            ))
    counts = {len(values) for values in rows.values()}
    if len(counts) != 1 or not counts or min(counts) < 2:
        raise ValueError("balanced paired scenarios required")
    output = []
    for challenger_id in challengers:
        sequence = sorted(rows[challenger_id])
        if [item[0] for item in sequence] != list(range(len(sequence))):
            raise ValueError("scenario index mismatch")
        output.append([item[1] for item in sequence])
    return np.asarray(output, dtype=float)


def build_paired_certificate(
    differences,
    delta,
    candidate_id,
    challenger_ids,
    objective_names,
    raw_path,
    *,
    method="hybrid",
    hybrid_hotelling_share=0.5,
):
    challengers, objectives = _validate_identity_contract(
        candidate_id, challenger_ids, objective_names
    )
    array = _validate_differences(differences, challengers, objectives)
    raw_array = read_raw_paired(raw_path, challengers, objectives)
    if raw_array.shape != array.shape or not np.allclose(
        raw_array, array, rtol=0.0, atol=1e-14
    ):
        raise ValueError("raw evidence does not match supplied differences")

    certified, decisions = evaluate_archive(
        array, delta, method=method,
        hybrid_hotelling_share=hybrid_hotelling_share,
    )
    semantic = {
        "specification": "pcpi-paired-candidate-certificate/0.6",
        "claim": {
            "type": "candidateStrictlyParetoNondominatedInMean",
            "candidate_id": candidate_id,
            "challenger_ids": list(challengers),
            "objective_names": list(objectives),
            "orientation": "minimise",
            "delta": float(delta),
            "model": "iidPairedMultivariateNormalUnknownPositiveDefiniteCovariance",
        },
        "stopping_rule": {
            "method": method,
            "spending": "log_telescoping",
            "hybrid_hotelling_share": float(hybrid_hotelling_share),
        },
        "sufficient_statistics": {
            "count": int(array.shape[1]),
            "means": [np.mean(item, axis=0).tolist() for item in array],
            "covariances": [
                np.cov(item, rowvar=False, ddof=1)
                .reshape(array.shape[2], array.shape[2]).tolist()
                for item in array
            ],
        },
        "certificate": {
            "verdict": "certified" if certified else "notCertified",
            "crossed": [item.crossed for item in decisions],
            "hotelling_distances": [item.hotelling_distance for item in decisions],
            "hotelling_thresholds": [item.hotelling_threshold for item in decisions],
            "coordinate_t_max": [item.coordinate_t_max for item in decisions],
            "coordinate_thresholds": [item.coordinate_threshold for item in decisions],
            "witness_objective_indices": [item.witness_objective for item in decisions],
            "covariance_ranks": [item.covariance_rank for item in decisions],
        },
        "raw_evidence": {
            "media_type": "text/csv",
            "sha256": file_sha256(raw_path),
            "relative_path": Path(raw_path).name,
        },
    }
    return {
        "semantic_core": semantic,
        "semantic_sha256": sha256_bytes(canonical_json(semantic)),
        "execution_metadata": {
            "generator": "pcpi-candidate-certification",
            "generator_version": __version__,
            "algorithm_id": "paired_sequential_confirmation",
        },
    }


def replay_paired_certificate(artifact, raw_path, schema_path=None):
    if schema_path:
        Draft202012Validator(
            json.loads(Path(schema_path).read_text())
        ).validate(artifact)
    semantic = artifact["semantic_core"]
    if sha256_bytes(canonical_json(semantic)) != artifact["semantic_sha256"]:
        raise ValueError("semantic hash mismatch")
    if file_sha256(raw_path) != semantic["raw_evidence"]["sha256"]:
        raise ValueError("raw evidence hash mismatch")
    claim = semantic["claim"]
    challengers, objectives = _validate_identity_contract(
        claim["candidate_id"], claim["challenger_ids"],
        claim["objective_names"],
    )
    differences = read_raw_paired(raw_path, challengers, objectives)
    if differences.shape[1] != semantic["sufficient_statistics"]["count"]:
        raise ValueError("count mismatch")
    means = np.asarray([item.mean(axis=0) for item in differences])
    covariances = np.asarray([
        np.cov(item, rowvar=False, ddof=1) for item in differences
    ])
    if not np.allclose(
        means, semantic["sufficient_statistics"]["means"],
        rtol=0.0, atol=1e-10,
    ):
        raise ValueError("mean mismatch")
    if not np.allclose(
        covariances, semantic["sufficient_statistics"]["covariances"],
        rtol=0.0, atol=1e-10,
    ):
        raise ValueError("covariance mismatch")
    rule = semantic["stopping_rule"]
    certified, decisions = evaluate_archive(
        differences,
        claim["delta"],
        method=rule["method"],
        hybrid_hotelling_share=rule["hybrid_hotelling_share"],
    )
    certificate = semantic["certificate"]
    if certificate["verdict"] != (
        "certified" if certified else "notCertified"
    ):
        raise ValueError("verdict mismatch")
    exact = {
        "crossed": [item.crossed for item in decisions],
        "witness_objective_indices": [
            item.witness_objective for item in decisions
        ],
        "covariance_ranks": [item.covariance_rank for item in decisions],
    }
    for key, values in exact.items():
        if certificate[key] != values:
            raise ValueError(key + " mismatch")
    numeric = {
        "hotelling_distances": [
            item.hotelling_distance for item in decisions
        ],
        "hotelling_thresholds": [
            item.hotelling_threshold for item in decisions
        ],
        "coordinate_t_max": [item.coordinate_t_max for item in decisions],
        "coordinate_thresholds": [
            item.coordinate_threshold for item in decisions
        ],
    }
    for key, values in numeric.items():
        declared = certificate[key]
        if len(declared) != len(values):
            raise ValueError(key + " length mismatch")
        for left, right in zip(declared, values):
            if left is None or right is None:
                if left is not None or right is not None:
                    raise ValueError(key + " mismatch")
            elif not np.isclose(
                float(left), float(right), rtol=0.0, atol=1e-10
            ):
                raise ValueError(key + " mismatch")
    return {
        "valid": True,
        "certified": certified,
        "method": rule["method"],
        "stopping_count": int(differences.shape[1]),
        "algorithm_id": "independent_paired_verifier",
    }

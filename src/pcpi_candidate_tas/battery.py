"""Backward-compatible imports for the stylised battery application model.

The main model is implemented in :mod:`pcpi_candidate_tas.application`.
Compact fixture-loading helpers remain for regression tests.
"""
from __future__ import annotations

from pathlib import Path
import csv
import numpy as np

from .application import (
    ApplicationCalibration,
    BatteryPolicy,
    BatterySystemParameters,
    OBJECTIVE_NAMES,
    PolicyThresholds,
    SelectionWeights,
    application_contract,
    calibrate_application,
    candidate_differences,
    compromise_scores,
    evaluate_archive,
    pareto_mask,
    policy_archive,
    policy_design_diagnostics,
    select_compromise_candidate,
    simulate_policy,
    simulate_policy_with_trace,
)


def load_opsd_fixture(path: Path | str) -> dict:
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty fixture")
    generic = {"utc_timestamp", "price_eur_per_mwh", "load_mwh", "split"}
    legacy = {
        "utc_timestamp",
        "DK_1_price_day_ahead_EUR_MWh",
        "DK_1_load_actual_entsoe_transparency_MWh",
        "split",
    }
    columns = set(rows[0])
    if columns == generic:
        price_key, load_key = "price_eur_per_mwh", "load_mwh"
    elif columns == legacy:
        price_key = "DK_1_price_day_ahead_EUR_MWh"
        load_key = "DK_1_load_actual_entsoe_transparency_MWh"
    else:
        raise ValueError("unexpected fixture columns")
    return {
        "price": np.asarray([float(row[price_key]) for row in rows]),
        "load": np.asarray([float(row[load_key]) for row in rows]),
        "split": np.asarray([row["split"] for row in rows]),
        "timestamps": np.asarray([row["utc_timestamp"] for row in rows]),
    }


def _base_days(data: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(data["split"]) == split
    price = np.asarray(data["price"])[mask]
    load = np.asarray(data["load"])[mask]
    if len(price) % 24:
        raise ValueError("whole days required")
    return price.reshape(-1, 24), load.reshape(-1, 24)


def generate_scenarios(
    data: dict,
    split: str,
    count: int,
    seed: int,
    *,
    price_noise: float = 0.055,
    load_noise: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy six-day fixture generator retained for regression only."""

    price_days, load_days = _base_days(data, split)
    rng = np.random.default_rng(seed)
    prices = []
    loads = []
    for _ in range(count):
        day = int(rng.integers(len(price_days)))
        innovation = rng.normal(size=24)
        smooth = np.convolve(innovation, np.ones(3) / 3, mode="same")
        prices.append(
            price_days[day] * (1 + price_noise * smooth)
            + rng.normal(0, 1.5, size=24)
        )
        loads.append(
            np.clip(
                load_days[day] * (1 + load_noise * smooth)
                + rng.normal(0, 15, size=24),
                1,
                None,
            )
        )
    return np.asarray(prices), np.asarray(loads)

"""Multi-season scenario generation for a stylised application study.

The repository includes deterministic synthetic fixtures only. Users may
supply compatible data under their own data-use obligations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import numpy as np


@dataclass(frozen=True)
class ScenarioGeneratorParameters:
    window_hours: int = 24
    smoothing_window: int = 5
    shared_smooth_weight: float = 0.60
    shared_level_sd: float = 0.02
    multiplicative_price_sd: float = 0.045
    multiplicative_load_sd: float = 0.020
    additive_price_sd_eur_per_mwh: float = 1.25
    additive_load_sd_system_mwh: float = 12.0
    season_sampling: str = "balanced"

    def __post_init__(self) -> None:
        if self.window_hours < 2:
            raise ValueError("window_hours >= 2 required")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window >= 1 required")
        if not 0 <= self.shared_smooth_weight <= 1:
            raise ValueError("shared_smooth_weight must lie in [0,1]")
        for name, value in (
            ("shared_level_sd", self.shared_level_sd),
            ("multiplicative_price_sd", self.multiplicative_price_sd),
            ("multiplicative_load_sd", self.multiplicative_load_sd),
            ("additive_price_sd_eur_per_mwh", self.additive_price_sd_eur_per_mwh),
            ("additive_load_sd_system_mwh", self.additive_load_sd_system_mwh),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.season_sampling not in {"balanced", "window_uniform"}:
            raise ValueError("unsupported season_sampling")


def load_multiseason_subset(path: Path | str) -> dict:
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    generic = {
        "utc_timestamp",
        "season",
        "split",
        "price_eur_per_mwh",
        "load_mwh",
    }
    legacy = {
        "utc_timestamp",
        "season",
        "split",
        "DK_1_price_day_ahead_EUR_MWh",
        "DK_1_load_actual_entsoe_transparency_MWh",
    }
    if not rows:
        raise ValueError("empty multi-season input")
    columns = set(rows[0])
    if columns == generic:
        price_key = "price_eur_per_mwh"
        load_key = "load_mwh"
    elif columns == legacy:
        price_key = "DK_1_price_day_ahead_EUR_MWh"
        load_key = "DK_1_load_actual_entsoe_transparency_MWh"
    else:
        raise ValueError("unexpected multi-season columns")
    price = np.asarray([float(row[price_key]) for row in rows])
    load = np.asarray([float(row[load_key]) for row in rows])
    season = np.asarray([row["season"] for row in rows])
    split = np.asarray([row["split"] for row in rows])
    timestamps = np.asarray([row["utc_timestamp"] for row in rows])
    if not np.all(np.isfinite(price)) or not np.all(np.isfinite(load)):
        raise ValueError("finite price and load are required")
    if np.any(load <= 0):
        raise ValueError("strictly positive load is required")
    return {
        "price": price,
        "load": load,
        "season": season,
        "split": split,
        "timestamps": timestamps,
    }


def seasonal_windows(
    data: dict,
    split: str,
    window: int = 24,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    if window < 2:
        raise ValueError("window >= 2 required")
    blocks = []
    for season in sorted(set(data["season"])):
        mask = (np.asarray(data["season"]) == season) & (
            np.asarray(data["split"]) == split
        )
        prices = np.asarray(data["price"])[mask]
        loads = np.asarray(data["load"])[mask]
        if len(prices) < window:
            continue
        starts = list(range(0, len(prices) - window + 1, window))
        final = len(prices) - window
        if final not in starts:
            starts.append(final)
        for start in sorted(set(starts)):
            blocks.append(
                (
                    str(season),
                    prices[start : start + window].copy(),
                    loads[start : start + window].copy(),
                )
            )
    if not blocks:
        raise ValueError(f"no {window}-hour windows for split {split!r}")
    return tuple(blocks)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(values, kernel, mode="same")


def generate_multiseason_scenarios(
    data: dict,
    split: str,
    count: int,
    seed: int,
    *,
    parameters: ScenarioGeneratorParameters = ScenarioGeneratorParameters(),
    return_metadata: bool = False,
):
    """Generate independent scenario blocks with frozen, declared perturbations."""

    if count < 1:
        raise ValueError("positive scenario count required")
    blocks = seasonal_windows(data, split, parameters.window_hours)
    by_season: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for season, prices, loads in blocks:
        by_season.setdefault(season, []).append((prices, loads))
    seasons_available = sorted(by_season)
    rng = np.random.default_rng(seed)

    if parameters.season_sampling == "balanced":
        labels = np.asarray(
            [
                seasons_available[index % len(seasons_available)]
                for index in range(count)
            ],
            dtype=object,
        )
        rng.shuffle(labels)
    else:
        block_labels = np.asarray([season for season, _, _ in blocks], dtype=object)
        labels = block_labels[rng.integers(len(block_labels), size=count)]

    prices_out = []
    loads_out = []
    selected_windows = []
    clipped_load_values = 0
    independent_weight = float(np.sqrt(1.0 - parameters.shared_smooth_weight**2))

    for season in labels:
        windows = by_season[str(season)]
        window_index = int(rng.integers(len(windows)))
        base_price, base_load = windows[window_index]

        shared = _smooth(rng.normal(size=parameters.window_hours), parameters.smoothing_window)
        independent_price = _smooth(
            rng.normal(size=parameters.window_hours), parameters.smoothing_window
        )
        independent_load = _smooth(
            rng.normal(size=parameters.window_hours), parameters.smoothing_window
        )
        price_profile_shock = (
            parameters.shared_smooth_weight * shared
            + independent_weight * independent_price
        )
        load_profile_shock = (
            parameters.shared_smooth_weight * shared
            + independent_weight * independent_load
        )
        shared_scale = rng.lognormal(
            mean=-0.5 * parameters.shared_level_sd**2,
            sigma=parameters.shared_level_sd,
        )
        price = (
            base_price
            * shared_scale
            * (1.0 + parameters.multiplicative_price_sd * price_profile_shock)
            + rng.normal(
                0.0,
                parameters.additive_price_sd_eur_per_mwh,
                size=parameters.window_hours,
            )
        )
        raw_load = (
            base_load
            * shared_scale
            * (1.0 + parameters.multiplicative_load_sd * load_profile_shock)
            + rng.normal(
                0.0,
                parameters.additive_load_sd_system_mwh,
                size=parameters.window_hours,
            )
        )
        clipped_load_values += int(np.count_nonzero(raw_load < 1.0))
        load = np.clip(raw_load, 1.0, None)

        prices_out.append(price)
        loads_out.append(load)
        selected_windows.append(window_index)

    result = (
        np.asarray(prices_out),
        np.asarray(loads_out),
        np.asarray(labels, dtype=str),
    )
    if not return_metadata:
        return result
    metadata = {
        "split": split,
        "count": count,
        "seed": seed,
        "parameters": asdict(parameters),
        "season_counts": {
            season: int(np.count_nonzero(np.asarray(labels) == season))
            for season in seasons_available
        },
        "selected_window_counts": {
            f"{season}:{index}": int(
                sum(
                    label == season and selected == index
                    for label, selected in zip(labels, selected_windows)
                )
            )
            for season in seasons_available
            for index in range(len(by_season[season]))
        },
        "clipped_load_values": clipped_load_values,
        "scenario_independence_contract": (
            "random innovations and base-window choices are independent across "
            "scenario blocks; within-block hourly values are intentionally dependent"
        ),
    }
    return (*result, metadata)

"""Validated model and result objects for candidate-only confirmation experiments."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class GaussianCandidateInstance:
    """Row zero is the candidate; rows 1..n are challengers."""

    means: np.ndarray
    variances: np.ndarray
    system_ids: tuple[str, ...]
    objective_names: tuple[str, ...]

    def __post_init__(self) -> None:
        means = np.asarray(self.means, dtype=float)
        variances = np.asarray(self.variances, dtype=float)
        if means.ndim != 2 or variances.ndim != 2 or means.shape != variances.shape:
            raise ValueError("means and variances must be same-shape matrices")
        if means.shape[0] < 2 or means.shape[1] < 1:
            raise ValueError("candidate, challenger and objective required")
        if not np.all(np.isfinite(means)):
            raise ValueError("means must be finite")
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
            raise ValueError("variances must be finite and positive")
        if len(self.system_ids) != means.shape[0] or len(set(self.system_ids)) != len(self.system_ids):
            raise ValueError("system_ids must be unique and match rows")
        if len(self.objective_names) != means.shape[1] or len(set(self.objective_names)) != len(self.objective_names):
            raise ValueError("objective_names must be unique and match columns")
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "variances", variances)

    @property
    def n_challengers(self) -> int:
        return self.means.shape[0] - 1

    @property
    def n_objectives(self) -> int:
        return self.means.shape[1]

    @property
    def gaps(self) -> np.ndarray:
        return self.means[1:] - self.means[0]

    @property
    def strictly_nondominated_candidate(self) -> bool:
        return bool(np.all(np.max(self.gaps, axis=1) > 0))

    @classmethod
    def from_arrays(
        cls, means: Sequence[Sequence[float]], variances: Sequence[Sequence[float]],
        system_ids: Sequence[str] | None = None,
        objective_names: Sequence[str] | None = None,
    ) -> "GaussianCandidateInstance":
        matrix = np.asarray(means, dtype=float)
        ids = tuple(system_ids or ["candidate", *[
            f"challenger_{i}" for i in range(1, matrix.shape[0])
        ]])
        names = tuple(objective_names or [
            f"objective_{j}" for j in range(matrix.shape[1])
        ])
        return cls(matrix, np.asarray(variances, dtype=float), ids, names)


@dataclass(frozen=True)
class AllocationResult:
    weights: np.ndarray
    rate: float
    q_star: float
    equalised_information: np.ndarray
    numerical_tolerance: float
    iterations: int


@dataclass(frozen=True)
class CandidateRunResult:
    certified: bool
    stopping_time: int
    counts: np.ndarray
    sums: np.ndarray
    sample_means: np.ndarray
    crossed: np.ndarray
    cone_glr_values: np.ndarray
    cone_thresholds: np.ndarray
    max_z_values: np.ndarray
    coordinate_thresholds: np.ndarray
    witness_objectives: np.ndarray
    oracle_failures: int
    target_weights: np.ndarray
    boundary_method: str
    spending: str
    trace: tuple[dict, ...]


@dataclass(frozen=True)
class FullPSIRunResult:
    completed: bool
    stopping_time: int
    counts: np.ndarray
    sums: np.ndarray
    sample_means: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    dominated: np.ndarray
    nondominated: np.ndarray
    estimated_pareto_set: tuple[int, ...]
    trace: tuple[dict, ...]

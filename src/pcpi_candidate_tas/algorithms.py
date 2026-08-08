"""Machine-readable algorithm contracts for candidate-only confirmation.

The registry separates mathematical reference algorithms, practical engineering
policies, diagnostic baselines, paired-output confirmation, and independent
replay.  A contract describes what an implementation answers, when it checks a
stopping rule, how it handles failures, and whether a theorem applies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AlgorithmContract:
    algorithm_id: str
    name: str
    purpose: str
    decision_scope: str
    output_semantics: str
    observation_model: str
    sampling_granularity: str
    target_source: str
    update_schedule: str
    stopping_grid: str
    theorem_coverage: str
    finite_precision_status: str
    oracle_failure_policy: str
    complexity_summary: str
    nonclaims: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


_CONTRACTS = (
    AlgorithmContract(
        "reference_unit_pull_c_tracking",
        "Reference unit-pull cumulative-target confirmation",
        "Theory-to-code reference for the known-variance one-sided procedure.",
        "Confirm one fixed candidate relative to one fixed finite archive.",
        "certified or undecided",
        "Independent Gaussian objectives with known positive diagonal variances.",
        "One system pull per iteration.",
        "Plug-in candidate-specific scalar oracle with vanishing gap regularisation.",
        "Immediate update after mandatory exploration, then every fixed number of pulls.",
        "After every pull.",
        "Pathwise allocation convergence and almost-sure termination on strictly positive instances under the stated ideal-oracle assumptions.",
        "Executable is a finite-precision approximation; the theorem concerns exact or vanishing-error oracle evaluations.",
        "Fail closed by default; optional uniform fallback is explicitly outside theorem coverage.",
        "Per target update: nested scalar oracle; per pull: O(Km) boundary evaluation plus O(K) tracking selection.",
        (
            "No expected-time optimality claim.",
            "No guarantee on false or equality-boundary instances.",
            "No serial-dependence guarantee.",
            "No theorem for the batched policy.",
        ),
    ),
    AlgorithmContract(
        "batched_plugin_tracking",
        "Batched plug-in candidate-only confirmation",
        "Practical engineering policy with periodic oracle updates and batch allocation.",
        "Confirm one fixed candidate relative to one fixed finite archive.",
        "certified at a batch boundary or undecided at the cap",
        "Independent Gaussian objectives with known positive diagonal variances.",
        "A deterministic batch of pulls per iteration.",
        "Periodically updated plug-in candidate-specific scalar oracle.",
        "At the first batch and after the configured number of accumulated samples.",
        "At initialisation and after each completed batch.",
        "No inherited tracking theorem; empirical engineering policy only.",
        "Finite-precision numerical policy.",
        "Uniform fallback or fail-closed mode, chosen explicitly.",
        "Per batch: O(Km) boundary evaluation, optional nested oracle, and O(K + batch_size log K) integer allocation.",
        (
            "No pathwise theorem.",
            "No expected-time theorem.",
            "Batch-boundary stopping can overshoot an every-pull stopping time.",
        ),
    ),
    AlgorithmContract(
        "uniform_candidate_only",
        "Uniform candidate-only baseline",
        "Allocation baseline isolating the value of adaptive sampling.",
        "Same one-candidate archive claim.",
        "certified at a batch boundary or undecided",
        "Independent Gaussian objectives with known positive diagonal variances.",
        "Batched.",
        "Fixed uniform target.",
        "No oracle updates.",
        "At batch boundaries.",
        "Boundary validity only; no allocation-efficiency theorem.",
        "Exact target, finite-precision statistics.",
        "No oracle.",
        "O(Km) boundary work per batch plus O(K + batch_size log K) allocation.",
        ("Not an efficiency benchmark for every instance.",),
    ),
    AlgorithmContract(
        "half_candidate_baseline",
        "Half-candidate static baseline",
        "Diagnostic static allocation reserving half the effort for the shared candidate.",
        "Same one-candidate archive claim.",
        "certified at a batch boundary or undecided",
        "Independent Gaussian objectives with known positive diagonal variances.",
        "Batched.",
        "Fixed target: one half to the candidate and the remainder uniformly across challengers.",
        "No oracle updates.",
        "At batch boundaries.",
        "Boundary validity only.",
        "Exact target, finite-precision statistics.",
        "No oracle.",
        "O(Km) boundary work per batch.",
        ("Not claimed to be optimal.",),
    ),
    AlgorithmContract(
        "gap_racing_diagnostic",
        "Boundary-deficit racing diagnostic",
        "Heuristic baseline allocating challenger effort by unresolved boundary deficit.",
        "Same one-candidate archive claim.",
        "certified at a batch boundary or undecided",
        "Independent Gaussian objectives with known positive diagonal variances.",
        "Batched.",
        "Boundary-deficit heuristic, with half of the target assigned to the candidate.",
        "Recomputed at every batch.",
        "At batch boundaries.",
        "No allocation theorem; diagnostic only.",
        "Finite-precision heuristic.",
        "No oracle.",
        "O(Km) boundary work and O(K) target construction per batch.",
        ("Not a state-of-the-art racing method.", "Not theorem covered."),
    ),
    AlgorithmContract(
        "oracle_static_diagnostic",
        "Static true-oracle diagnostic",
        "Separates oracle-estimation cost from target-tracking and stopping behaviour.",
        "Same one-candidate archive claim.",
        "certified at a batch boundary or undecided",
        "Independent Gaussian objectives with known positive diagonal variances.",
        "Batched.",
        "True-parameter or externally supplied candidate-specific target.",
        "No plug-in updates.",
        "At batch boundaries.",
        "Diagnostic only; uses information unavailable to a real procedure.",
        "Finite-precision target computation.",
        "Fail closed when supplied target is invalid.",
        "One oracle computation plus O(Km) boundary work per batch.",
        ("Not deployable without true parameters.", "Not a fair practical comparator."),
    ),
    AlgorithmContract(
        "paired_sequential_confirmation",
        "Paired common-scenario archive confirmation",
        "Sequentially inspect common-scenario paired differences using Hotelling, coordinate, or hybrid evidence.",
        "Confirm one fixed candidate relative to one fixed archive.",
        "certified on a prespecified intrinsic-count grid or undecided",
        "IID multivariate-normal paired differences; positive-definite covariance for the Hotelling branch.",
        "One common scenario adds one vector for every challenger.",
        "No allocation oracle; observations are supplied as a paired sequence.",
        "No target updates.",
        "At the configured intrinsic-count grid, including the final count.",
        "Finite-time one-sided soundness under the declared paired Gaussian model.",
        "Finite-precision covariance, NNLS, t, and F calculations.",
        "Singular covariance is non-certification, not evidence.",
        "For C challengers, S counts and m objectives: O(CSm^2) cumulative statistics in the current replay implementation and O(Cm^3) per checked Hotelling grid point.",
        ("No validity under serial dependence or arbitrary heavy tails.",),
    ),
    AlgorithmContract(
        "independent_paired_verifier",
        "Independent raw-bound paired-certificate verifier",
        "Reconstruct a declared paired decision from schema-valid metadata and hashed raw CSV evidence.",
        "Verify internal consistency of one evidence object.",
        "valid/invalid artefact and reproduced certified/not-certified verdict",
        "The model named by the evidence object; verifier does not test model adequacy.",
        "Offline replay.",
        "No sampling target.",
        "No updates.",
        "Single complete replay.",
        "Software conformance contract, not a statistical theorem beyond the replayed rule.",
        "Independent finite-precision implementation.",
        "Fail closed on schema, identity, hash, raw/statistic, boundary, or verdict mismatch.",
        "O(CSm + Cm^3) time and O(CSm) raw-data memory in the current implementation.",
        (
            "Does not authenticate the simulator or random-number source.",
            "Does not validate distributional assumptions.",
            "Does not prove global Pareto optimality.",
        ),
    ),
)


def algorithm_contracts() -> tuple[AlgorithmContract, ...]:
    return _CONTRACTS


def algorithm_contract(algorithm_id: str) -> AlgorithmContract:
    for contract in _CONTRACTS:
        if contract.algorithm_id == algorithm_id:
            return contract
    raise KeyError(f"unknown algorithm_id: {algorithm_id}")


def algorithm_contract_dicts() -> tuple[dict, ...]:
    return tuple(contract.to_dict() for contract in _CONTRACTS)

from .study_design import (
    SeedPlan, HardArchiveDesign, ConfirmatoryDesign, StudyDesign,
    local_policy_pool, construct_hard_archive, validate_hard_archive,
    replication_precision_table,
)
from ._version import __version__
from .models import GaussianCandidateInstance, AllocationResult, CandidateRunResult, FullPSIRunResult
from .oracle import solve_scalar_allocation
from .candidate import (
    run_candidate_certification,
    run_theory_aligned_c_tracking,
    run_reference_c_tracking,
)
from .paired import sequential_archive_confirmation, evaluate_archive
from .algorithms import (
    AlgorithmContract,
    algorithm_contract,
    algorithm_contracts,
    algorithm_contract_dicts,
)

__all__ = [
    "__version__",
    "GaussianCandidateInstance",
    "AllocationResult",
    "CandidateRunResult",
    "FullPSIRunResult",
    "solve_scalar_allocation",
    "run_candidate_certification",
    "run_theory_aligned_c_tracking",
    "run_reference_c_tracking",
    "sequential_archive_confirmation",
    "evaluate_archive",
    "AlgorithmContract",
    "algorithm_contract",
    "algorithm_contracts",
    "algorithm_contract_dicts",
    "SeedPlan", "HardArchiveDesign", "ConfirmatoryDesign",
    "StudyDesign", "local_policy_pool",
    "construct_hard_archive", "validate_hard_archive",
    "replication_precision_table",
    "OneSidedTheoryStatus", "one_sided_theory_status",
    "CalibrationDesign",
]

from .theory_status import OneSidedTheoryStatus, one_sided_theory_status
from .calibration import CalibrationDesign

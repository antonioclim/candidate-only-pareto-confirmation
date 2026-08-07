"""Reference-equivalence checks for accelerated execution kernels."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcpi_candidate_tas.application import (
    calibrate_application,
    candidate_differences,
    evaluate_archive,
    policy_archive,
)
from pcpi_candidate_tas.multiseason import (
    generate_multiseason_scenarios,
    load_multiseason_subset,
)
from pcpi_candidate_tas.paired import sequential_archive_confirmation
from pcpi_candidate_tas.accelerated import (
    evaluate_archive_fast,
    sequential_candidate_fast,
    sequential_full_psi_coordinate_fast,
)
from pcpi_candidate_tas.psi_comparators import sequential_full_psi_coordinate
from pcpi_candidate_tas.calibration import (
    generate_exact_null_path,
    generate_robustness_path,
    sequential_method_diagnostics,
)


def main() -> None:
    data = load_multiseason_subset(
        ROOT / "data/synthetic_multiseason/synthetic_multiseason_fixture.csv"
    )
    policies = policy_archive()
    calibration = calibrate_application(data, policies)
    application_cases = 0
    application_mismatches = 0
    maximum_absolute = 0.0
    maximum_relative = 0.0

    for seed in range(6):
        prices, loads, _ = generate_multiseason_scenarios(
            data, "confirmation", 120 + 10 * seed, 710_000 + seed
        )
        scalar = evaluate_archive(policies, prices, loads, calibration)
        fast = evaluate_archive_fast(policies, prices, loads, calibration)
        difference = np.abs(scalar - fast)
        maximum_absolute = max(maximum_absolute, float(np.max(difference)))
        maximum_relative = max(
            maximum_relative,
            float(np.max(difference / np.maximum(np.abs(scalar), 1.0))),
        )
        reference_full = sequential_full_psi_coordinate(
            scalar, 0.05, min_count=12, max_count=len(scalar), check_every=5
        )
        accelerated_full = sequential_full_psi_coordinate_fast(
            scalar, 0.05, min_count=12, max_count=len(scalar), check_every=5
        )
        application_cases += 1
        application_mismatches += int(
            reference_full["completed"] != accelerated_full["completed"]
            or reference_full["stopping_count"] != accelerated_full["stopping_count"]
            or not np.array_equal(reference_full["dominated"], accelerated_full["dominated"])
            or not np.array_equal(reference_full["nondominated"], accelerated_full["nondominated"])
        )

    prices, loads, _ = generate_multiseason_scenarios(
        data, "confirmation", 160, 717_001
    )
    scalar = evaluate_archive(policies, prices, loads, calibration)
    paired, _ = candidate_differences(scalar, 1)
    for method in ("coordinate", "hotelling", "hybrid"):
        reference = sequential_archive_confirmation(
            paired, 0.05, method=method, min_count=8,
            max_count=160, check_every=5
        )
        accelerated = sequential_candidate_fast(
            paired, 0.05, method=method, min_count=8,
            max_count=160, check_every=5
        )
        application_cases += 1
        application_mismatches += int(
            reference["certified"] != accelerated["certified"]
            or reference["stopping_count"] != accelerated["stopping_count"]
        )

    calibration_cases = 0
    calibration_mismatches = 0
    for dimension in (2, 4):
        for covariance in ("identity", "compound_rho_0.5"):
            path = generate_exact_null_path(
                800_000 + 10 * dimension, dimension, covariance, count=300
            )
            diagnostics = sequential_method_diagnostics(
                path, 0.05,
                methods=("coordinate", "hotelling", "hybrid"),
                max_count=300, check_every=5,
            )
            for method in ("coordinate", "hotelling", "hybrid"):
                reference = sequential_archive_confirmation(
                    path, 0.05, method=method, min_count=8,
                    max_count=300, check_every=5
                )
                calibration_cases += 1
                calibration_mismatches += int(
                    reference["certified"] != diagnostics[method]["certified"]
                    or reference["stopping_count"] != diagnostics[method]["stopping_count"]
                )

    for mechanism_index, mechanism in enumerate((
        "iid Gaussian reference",
        "multivariate t, df=5",
        "multivariate t, df=3",
        "centred lognormal coordinates",
        "1% ten-SD contamination",
        "AR(1) rho=0.3",
        "AR(1) rho=0.6",
        "near-singular covariance",
    )):
        path = generate_robustness_path(
            900_000 + mechanism_index, mechanism, count=300, dimension=3
        )
        diagnostics = sequential_method_diagnostics(
            path, 0.05,
            methods=("coordinate", "hotelling", "hybrid"),
            max_count=300, check_every=5,
        )
        for method in ("coordinate", "hotelling", "hybrid"):
            reference = sequential_archive_confirmation(
                path, 0.05, method=method, min_count=8,
                max_count=300, check_every=5
            )
            calibration_cases += 1
            calibration_mismatches += int(
                reference["certified"] != diagnostics[method]["certified"]
                or reference["stopping_count"] != diagnostics[method]["stopping_count"]
            )

    report = {
        "specification": "pcpi-reference-equivalence/1.0",
        "application_cases": application_cases,
        "application_mismatches": application_mismatches,
        "calibration_cases": calibration_cases,
        "calibration_mismatches": calibration_mismatches,
        "maximum_application_absolute_difference": maximum_absolute,
        "maximum_application_relative_difference": maximum_relative,
        "all_gates_passed": (
            application_mismatches == 0
            and calibration_mismatches == 0
            and maximum_absolute < 1e-8
            and maximum_relative < 1e-12
        ),
    }
    output = ROOT / "generated/equivalence_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

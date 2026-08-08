"""Run a compact deterministic synthetic application example."""
from pathlib import Path
import json
import numpy as np

from pcpi_candidate_tas.application import (
    calibrate_application,
    evaluate_archive,
    policy_archive,
    select_compromise_candidate,
)
from pcpi_candidate_tas.multiseason import (
    generate_multiseason_scenarios,
    load_multiseason_subset,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/synthetic_multiseason/synthetic_multiseason_fixture.csv"


def main() -> None:
    data = load_multiseason_subset(DATA)
    policies = policy_archive()
    calibration = calibrate_application(data, policies)
    prices, loads, seasons = generate_multiseason_scenarios(
        data, "confirmation", 32, 120_001
    )
    outcomes = evaluate_archive(policies, prices, loads, calibration)
    candidate = select_compromise_candidate(outcomes)
    result = {
        "scenarios": int(len(prices)),
        "policy_count": len(policies),
        "candidate_policy_id": policies[candidate].policy_id,
        "seasons": sorted(set(seasons.tolist())),
        "finite": bool(np.all(np.isfinite(outcomes))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

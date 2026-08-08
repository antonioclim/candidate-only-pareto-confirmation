from __future__ import annotations

import argparse
import json
from pathlib import Path

from .algorithms import algorithm_contract_dicts
from .paired_artifacts import replay_paired_certificate


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pcpi-candidate-certification")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verifier = subparsers.add_parser(
        "verify-paired", help="replay a raw-bound paired certificate"
    )
    verifier.add_argument("--artifact", required=True)
    verifier.add_argument("--raw", required=True)
    verifier.add_argument("--schema", required=True)

    describe = subparsers.add_parser(
        "describe-algorithms",
        help="emit machine-readable algorithm contracts",
    )
    describe.add_argument(
        "--pretty", action="store_true", help="indent JSON output"
    )

    args = parser.parse_args(argv)
    if args.command == "verify-paired":
        result = replay_paired_certificate(
            json.loads(Path(args.artifact).read_text()),
            args.raw,
            args.schema,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "describe-algorithms":
        print(json.dumps(
            algorithm_contract_dicts(),
            sort_keys=True,
            indent=2 if args.pretty else None,
        ))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

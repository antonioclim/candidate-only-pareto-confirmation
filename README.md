# One-Sided Candidate-Only Pareto Confirmation

[![Software version](https://img.shields.io/badge/version-1.1.0-blue.svg)](RELEASE_NOTES_v1.1.0.md)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

This repository implements and evaluates a one-sided procedure for confirming
that one externally selected candidate is Pareto-nondominated in mean relative
to one fixed finite archive. The procedure may issue a positive archive-relative
certificate or remain undecided. It is not a total fixed-confidence
identification algorithm, and non-certification is not a domination verdict.

## Scientific scope

The release contains:

- the candidate-specific Gaussian dominance-cone information projection;
- a scalar allocation specialisation built on established equalisation and
  inverse-root machinery;
- theorem-aligned unit-pull tracking and a separate batched engineering policy;
- cone, coordinate, hybrid and paired Hotelling evidence procedures;
- a raw-data-bound certificate and an independent standalone verifier;
- powered application, robustness and tracking evidence;
- manuscript and supplementary sources.

## Explicit non-claims

- no global delta-PAC identification;
- no expected-time first-order optimality;
- no novelty claim for the generic scalar equalisation/inverse-root architecture;
- no complete Pareto-front recovery;
- no validity under unmodelled temporal dependence;
- no authentication of the simulator or upstream selection process.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Core verification

```bash
python -m pytest -q
python -m coverage run -m pytest -q
python -m coverage report --fail-under=90
python tools/manual_mutation_suite.py
python tools/verify_paired_certificate_standalone.py \
  --artifact evidence/artefact/example_certificate.json \
  --raw evidence/artefact/example_raw.csv \
  --schema schemas/pcpi_paired_candidate_certificate.schema.json
```

The current release reports 50 passing tests, 95% statement coverage, 9/9
mutation probes killed and 14/14 artefact conformance cases.

## Reproducing the reported studies

The deterministic experiment drivers are under `scripts/`. Aggregated evidence
is stored under `evidence/`; source data and machine-readable provenance are
under `data/`.

## Citation and persistent identifier

See `CITATION.cff`. The Zenodo DOI should be added after the record is actually
published; no placeholder DOI is asserted in this release bundle.

## Data provenance

The seasonal application data are a derived public-mirror subset with pinned
repository commit, upstream blob identifiers, source line ranges, merge rules
and SHA-256 recorded in `data/multiseason/MULTISEASON_PROVENANCE.json`. They are
not the complete official OPSD package.

## Licence

Software is released under the BSD 3-Clause licence. Manuscript and
supplementary text are supplied for scholarly review and citation.

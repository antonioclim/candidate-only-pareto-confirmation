# One-Sided Candidate-Only Pareto Confirmation

[![Software version](https://img.shields.io/badge/version-1.1.0-blue.svg)](RELEASE_NOTES_v1.1.0.md)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21762634.svg)](https://doi.org/10.5281/zenodo.21762634)

This repository contains the software and reproducibility core for a one-sided
procedure that may confirm that one externally selected candidate is
Pareto-nondominated in mean relative to one fixed finite archive. The procedure
may issue a positive archive-relative certificate or remain undecided. It is not
a total fixed-confidence identification algorithm, and non-certification is not
a domination verdict.

## Public repository scope

The repository contains only the software and reproducibility core:

- Python source code and command-line interface;
- automated tests, coverage records and deterministic mutation probes;
- strict JSON Schema and example raw-bound evidence artefacts;
- standalone certificate verifier;
- reproducibility scripts and minimal derived data with machine-readable
  provenance;
- build, container, SBOM and continuous-integration configuration.

The unpublished manuscript, supplementary manuscript, submission documents and
article PDFs are intentionally **not included** in this repository or in the
associated Zenodo software record.

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

The v1.1.0 core reports 50 passing tests, 95% statement coverage, 9/9
mutation probes killed and 14/14 artefact conformance cases.

## Reproducibility resources

Deterministic experiment drivers are under `scripts/`. Minimal example evidence
is stored under `evidence/artefact/`; source data and machine-readable
provenance are under `data/`. Larger unpublished study outputs are not required
for installing or testing the software core and are not included here.

## Citation and persistent identifier

Archived software core v1.1.0: [10.5281/zenodo.21762634](https://doi.org/10.5281/zenodo.21762634).

See `CITATION.cff` for machine-readable citation metadata. The DOI was assigned
after the `v1.1.0` tag and binary assets were published; the tag and release
assets remain unchanged, while this default-branch metadata update records the
persistent identifier.

## Data provenance

The included seasonal data are a derived public-mirror subset with pinned
repository commit, upstream blob identifiers, source line ranges, merge rules
and SHA-256 recorded in `data/multiseason/MULTISEASON_PROVENANCE.json`. They are
not the complete official OPSD package.

## Licence

Software source code is released under the BSD 3-Clause licence. Textual
documentation and non-code metadata are licensed under CC BY 4.0 unless a file
states otherwise.

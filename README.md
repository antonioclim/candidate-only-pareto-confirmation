# One-Sided Candidate-Only Pareto Confirmation

[![Software version](https://img.shields.io/badge/version-1.2.0-blue.svg)](RELEASE_NOTES_v1.2.0.md)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

This repository contains a software and reproducibility core for confirming
strict positive-witness archive separation for one externally selected candidate
relative to one fixed finite archive. For every declared challenger, the
certificate requires evidence that the challenger is strictly worse than the
candidate in at least one minimised objective.

This proposition implies ordinary archive-relative Pareto nondominance, but is
strictly stronger on equality boundaries: an exact tie remains non-certifiable.
The procedure may issue the positive certificate or remain undecided.
Non-certification is not a domination verdict, and the software is not a total
fixed-confidence Pareto-front identification algorithm.

## Version 1.2.0

This release adds:

- an explicit strict-positive-witness semantic contract;
- paired coordinate, Hotelling and hybrid confirmation;
- reference-equivalent accelerated execution kernels;
- a prospectively frozen exact-Gaussian calibration design;
- machine-readable applicability boundaries;
- deterministic synthetic fixtures;
- a strict source-data non-redistribution policy;
- 95% statement-coverage enforcement;
- a source-based reproducible figure-generation kit using PlantUML semantic sources, Python quantitative sources and a normalised OOXML exact-layout layer.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Verification

```bash
python -m pytest -q
python -m coverage run -m pytest -q
python -m coverage report --fail-under=95
python tools/manual_mutation_suite.py
python tools/verify_paired_certificate_standalone.py --artifact evidence/examples/example_certificate.json --raw evidence/examples/example_raw.csv --schema schemas/pcpi_paired_candidate_certificate.schema.json
python scripts/verify_accelerated_equivalence.py
python scripts/verify_release.py
```

## Data policy

Only deterministic synthetic fixtures are included. Upstream observation rows
are not redistributed. Users may provide compatible data under their own
data-use obligations.

Project-generated study outputs are distributed separately as a research-data
record. The unpublished article, supplement and submission files are not
included in this software repository.

## Explicit non-claims

- no certificate for equality-boundary ordinary nondominance;
- no global delta-PAC Pareto-front identification;
- no expected-time first-order optimality;
- no distribution-free validity;
- no validity under unmodelled temporal dependence;
- no universal algorithmic superiority;
- no authentication of a simulator or upstream selection process.

Strong AR(1) dependence is an observed failure boundary and remains outside the
validity contract.

## Citation and related records

Version-specific software DOI:
`10.5281/zenodo.21801863`.

Research-outputs dataset DOI (separate record): `10.5281/zenodo.21802337`.

Previous public software version 1.1.0: `10.5281/zenodo.21762634`.

`CITATION.cff` deliberately omits `date-released` while this object remains
unpublished. The actual first-public-availability date must be inserted only at
publication.


## Reproducible figure generation

The `figures/` directory contains the source-based figure-generation kit.

```bash
python -m pip install -r figures/requirements-figures-lock.txt
python figures/build/repack_ooxml.py
python figures/build/render_exact_reference.py
python figures/build/build_quantitative.py
python figures/build/validate.py
```

Conceptual semantics are available as PlantUML sources. Quantitative figures are generated from compact project-derived evidence extracts. Normalised OOXML packages preserve the exact approved editorial layout and can be repacked deterministically by Python. PlantUML itself is not bundled; the required version and hash are recorded in `figures/toolchain/plantuml.lock.json`.

The figure kit is distributed with the repository/source archive and Zenodo software bundle; it is not installed by the core Python wheel.

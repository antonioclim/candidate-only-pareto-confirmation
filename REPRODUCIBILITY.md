# Reproducibility

## Environment

Python 3.10 or later is required. The exact packages used for the verified
release are listed in `requirements-lock.txt`. The repository also includes
Docker and Apptainer recipes, a CycloneDX SBOM and a multi-platform GitHub
Actions workflow.

## Core checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m coverage run -m pytest -q
python -m coverage report --fail-under=90
python tools/manual_mutation_suite.py
python tools/verify_paired_certificate_standalone.py \
  --artifact evidence/artefact/example_certificate.json \
  --raw evidence/artefact/example_raw.csv \
  --schema schemas/pcpi_paired_candidate_certificate.schema.json
```

## Experiment drivers

- `scripts/run_multiseason_application.py`
- `scripts/run_tracker_alignment.py`
- `scripts/run_robustness_study.py`

The scripts use fixed seeds and produce the aggregate files already stored in
`evidence/`. The data provenance contract is under `data/`.

## Verified release status

- 50 tests passed;
- 95% statement coverage;
- 9/9 deterministic mutation probes killed;
- 14/14 artefact conformance cases passed;
- standalone verifier reproduced the example verdict and stopping count;
- clean-extraction package test passed.

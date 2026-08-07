# Reproducibility

## Fast verification

```bash
python -m pip install -e ".[dev]"
python -m coverage run -m pytest -q
python -m coverage report --fail-under=95
python tools/manual_mutation_suite.py
python scripts/verify_accelerated_equivalence.py
python scripts/verify_release.py
```

## Synthetic application example

```bash
python scripts/run_synthetic_application.py
```

## Full calibration campaign

```bash
python scripts/run_calibration_campaign.py --campaign all --workers 4
python scripts/aggregate_calibration_results.py
```

The release contains compact summary evidence. Full project-generated raw
outputs are available in the separate research-data package.

## Associated research outputs

The project-generated raw outputs, summaries and figure-source data are archived
in the separate dataset record `10.5281/zenodo.21802337` after its controlled publication.


## Figure-generation verification

```bash
python -m pip install -r figures/requirements-figures-lock.txt
python figures/build/repack_ooxml.py
python figures/build/render_exact_reference.py
python figures/build/build_quantitative.py
python figures/build/validate.py
```

The figure kit includes only minimal derived inputs required for figure reproduction. The complete project-generated research outputs remain in the associated dataset record. The `figures/` directory is part of the repository/source archive and is not installed by the core wheel.

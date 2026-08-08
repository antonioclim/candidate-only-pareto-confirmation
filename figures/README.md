# PCPI / SIMPAT reproducible figure-generation kit

Version 1.0.0

This directory provides three complementary source layers:

1. **PlantUML semantic sources** for the conceptual figures F1, F2, F3 and the graphical abstract.
2. **Python/data sources** for the quantitative figures F4–F7, S1 and S2.
3. **Normalised OOXML layout sources** for exact reconstruction of the current editorial PPTX layout, followed by deterministic LibreOffice export.

## Why there are three layers

PlantUML is excellent for versionable semantic diagrams but does not guarantee pixel-identical placement across every renderer and version. Python is the natural source for quantitative plots. The normalised OOXML layer preserves the exact approved editorial layout as text/XML and permits deterministic repacking through Python.

## Fast build

```bash
python -m pip install -r requirements-figures-lock.txt
python build/build_all.py
python build/validate.py
```

The exact editorial build also requires LibreOffice in `PATH`. PlantUML rendering is optional for the semantic mirror and requires a separately downloaded jar matching `toolchain/plantuml.lock.json`.

## Output roles

- `outputs/rebuilt_pptx`: regenerated editable presentations;
- `outputs/rebuilt_odg`: LibreOffice Draw exports;
- `outputs/journal_png_600dpi`: journal-scale PNG exports;
- `outputs/quantitative_demo_*`: data-driven Python figures used to audit the numerical route.

## Governance

The unpublished article and supplementary manuscript are not contained in this kit. The kit contains figure sources, minimal derived figure-input data and reproducibility controls only.

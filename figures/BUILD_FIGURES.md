# Build instructions

## Exact editorial reference layer

```bash
python build/repack_ooxml.py
python build/render_exact_reference.py
```

## Quantitative data-driven layer

```bash
python build/build_quantitative.py
```

## PlantUML semantic layer

1. Download the exact PlantUML version listed in `toolchain/plantuml.lock.json`.
2. Verify its SHA-256.
3. Set `PLANTUML_JAR`.
4. Run:

```bash
python build/render_plantuml_semantic.py
```

## Full validation

```bash
python build/validate.py
```

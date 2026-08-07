#!/usr/bin/env python3
"""Build exact editorial references and quantitative demos."""
from __future__ import annotations
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for script in ["repack_ooxml.py", "render_exact_reference.py", "build_quantitative.py"]:
    print(f"=== {script} ===")
    runpy.run_path(str(ROOT / script), run_name="__main__")

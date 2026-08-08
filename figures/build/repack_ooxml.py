#!/usr/bin/env python3
"""Repack normalised OOXML slide packages into deterministic PPTX files."""
from __future__ import annotations
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "figure_registry.json").read_text(encoding="utf-8"))
OUT = ROOT / "outputs" / "rebuilt_pptx"

FIXED_DATE = (1980, 1, 1, 0, 0, 0)

def repack_one(fig: dict) -> Path:
    src = ROOT / fig["exact_ooxml_dir"]
    out = OUT / f"{fig['id']}.pptx"
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(p for p in src.rglob("*") if p.is_file()):
            rel = path.relative_to(src).as_posix()
            info = zipfile.ZipInfo(rel, FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, path.read_bytes())
    return out

def main() -> None:
    for fig in REGISTRY["figures"]:
        print(repack_one(fig))

if __name__ == "__main__":
    main()

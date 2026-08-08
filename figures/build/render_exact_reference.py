#!/usr/bin/env python3
"""Render rebuilt PPTX packages through LibreOffice and normalise PNG metadata."""
from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "figure_registry.json").read_text(encoding="utf-8"))
PPTX_DIR = ROOT / "outputs" / "rebuilt_pptx"
ODG_DIR = ROOT / "outputs" / "rebuilt_odg"
PREVIEW_DIR = ROOT / "outputs" / "preview_png"
JOURNAL_DIR = ROOT / "outputs" / "journal_png_600dpi"


def run_lo(src: Path, fmt: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "libreoffice", "--headless", "--convert-to", fmt,
        "--outdir", str(out_dir), str(src)
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return out_dir / (src.stem + "." + fmt)


def main() -> None:
    for fig in REGISTRY["figures"]:
        fid = fig["id"]
        pptx = PPTX_DIR / f"{fid}.pptx"
        if not pptx.exists():
            raise SystemExit(f"missing rebuilt pptx: {pptx}")
        odg = run_lo(pptx, "odg", ODG_DIR)
        preview = run_lo(pptx, "png", PREVIEW_DIR)
        target = JOURNAL_DIR / f"{fid}.png"
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        with Image.open(preview).convert("RGB") as im:
            im = im.resize((fig["target_png_width"], fig["target_png_height"]), Image.Resampling.LANCZOS)
            im.save(target, dpi=(fig["target_dpi"], fig["target_dpi"]))
        print(fid, odg, preview, target)

if __name__ == "__main__":
    main()

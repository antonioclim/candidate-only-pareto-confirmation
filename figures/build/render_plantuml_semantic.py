#!/usr/bin/env python3
"""Render PlantUML semantic sources when PLANTUML_JAR is available."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "toolchain" / "plantuml.lock.json").read_text(encoding="utf-8"))
SRC = ROOT / "source" / "plantuml"
OUT = ROOT / "outputs" / "plantuml_semantic"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    jar_raw = os.environ.get("PLANTUML_JAR")
    if not jar_raw:
        raise SystemExit("Set PLANTUML_JAR to the verified PlantUML jar path.")
    jar = Path(jar_raw)
    if sha256(jar) != LOCK["sha256"]:
        raise SystemExit("PlantUML jar SHA-256 does not match toolchain lock.")
    OUT.mkdir(parents=True, exist_ok=True)
    for puml in sorted(p for p in SRC.glob("*.puml") if p.is_file()):
        subprocess.run([
            "java", "-jar", str(jar), "-tsvg", "-o", str(OUT), str(puml)
        ], check=True)
        print(puml)

if __name__ == "__main__":
    main()

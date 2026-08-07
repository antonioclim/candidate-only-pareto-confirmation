import json,subprocess,sys
from pathlib import Path

def test_standalone_verifier_matches_package():
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,str(root/"tools/verify_paired_certificate_standalone.py"),
             "--artifact",str(root/"evidence/examples/example_certificate.json"),
             "--raw",str(root/"evidence/examples/example_raw.csv"),
             "--schema",str(root/"schemas/pcpi_paired_candidate_certificate.schema.json")]
    completed=subprocess.run(command,text=True,capture_output=True,check=True)
    result=json.loads(completed.stdout)
    assert result["valid"] and result["certified"]

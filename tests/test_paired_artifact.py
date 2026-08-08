from pathlib import Path
import json,numpy as np,pytest
from pcpi_candidate_tas.paired_artifacts import *
from pcpi_candidate_tas import __version__
ROOT=Path(__file__).resolve().parents[1]

def fixture(tmp_path):
 rng=np.random.default_rng(6);d=np.stack([rng.multivariate_normal([.7,-.1],np.eye(2),size=80),rng.multivariate_normal([-.1,.7],np.eye(2),size=80)])
 raw=tmp_path/'raw.csv';write_raw_paired(raw,d,['a','b'],['f1','f2'])
 art=build_paired_certificate(d,.1,'candidate',['a','b'],['f1','f2'],raw)
 return raw,art

def test_replay(tmp_path):
 raw,a=fixture(tmp_path);schema=ROOT/'schemas/pcpi_paired_candidate_certificate.schema.json'
 assert a['execution_metadata']['generator_version']==__version__
 assert replay_paired_certificate(a,raw,schema)['valid']

def test_raw_tamper(tmp_path):
 raw,a=fixture(tmp_path);schema=ROOT/'schemas/pcpi_paired_candidate_certificate.schema.json'
 raw.write_text(raw.read_text().replace(',0,',',9,',1))
 with pytest.raises(ValueError):replay_paired_certificate(a,raw,schema)

def test_cli_replay(tmp_path,capsys):
    from pcpi_candidate_tas.cli import main
    raw,art=fixture(tmp_path);schema=ROOT/'schemas/pcpi_paired_candidate_certificate.schema.json'
    artifact_path=tmp_path/'artifact.json';artifact_path.write_text(json.dumps(art))
    assert main(['verify-paired','--artifact',str(artifact_path),'--raw',str(raw),'--schema',str(schema)])==0
    output=json.loads(capsys.readouterr().out)
    assert output['valid']

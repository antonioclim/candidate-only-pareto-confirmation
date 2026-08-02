import argparse,json
from pathlib import Path
from .paired_artifacts import replay_paired_certificate

def main(argv=None):
    p=argparse.ArgumentParser(prog='pcpi-candidate-tas');s=p.add_subparsers(dest='command',required=True)
    v=s.add_parser('verify-paired');v.add_argument('--artifact',required=True);v.add_argument('--raw',required=True);v.add_argument('--schema',required=True)
    a=p.parse_args(argv)
    if a.command=='verify-paired':
        result=replay_paired_certificate(json.loads(Path(a.artifact).read_text()),a.raw,a.schema);print(json.dumps(result,sort_keys=True));return 0
    return 2
if __name__=='__main__':raise SystemExit(main())

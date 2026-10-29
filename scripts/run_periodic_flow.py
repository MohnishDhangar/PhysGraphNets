"""Run a manufactured incompressible-flow reference and export tracer animation."""
import argparse
import json
from pathlib import Path
from physgraph.experiments.periodic_flow import run_flow


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=run_flow(**json.loads(args.config.read_text()),output=args.output)
    print({k:v for k,v in result['metrics'].items() if k not in ('errors','diagnostics')})
    return 0 if result['metrics']['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())

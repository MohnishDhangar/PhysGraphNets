"""Run the predeclared GNN accuracy qualification."""
import argparse
import json
from pathlib import Path
from physgraph.experiments.gnn_study import run_study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=run_study(**json.loads(args.config.read_text()),output=args.output)
    print('Qualification:',result['metrics']['status'])
    return 0 if result['metrics']['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())

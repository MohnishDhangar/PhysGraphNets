"""Run a declared accuracy-versus-cost study on CPU."""
import argparse
import json
from pathlib import Path
from physgraph.experiments.pinn_cost import run_cost_study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    try:
        result=run_cost_study(**json.loads(args.config.read_text()),output=args.output)
    except (ValueError,TypeError,OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result['metrics']['selected'],indent=2))
    if result['metrics']['status']!='completed':
        raise SystemExit(1)


if __name__=='__main__':
    main()

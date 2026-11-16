"""Execute a predeclared mesh/timestep study for the complete DFG 2D-3 case."""
import argparse
import json
from pathlib import Path

from physgraph.experiments.cylinder_study import run_cylinder_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path('configs/experiments/cylinder_flow_qualification.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run_cylinder_study(output=args.output, **json.loads(args.config.read_text()))
    print(report['metrics']['status'])
    print(json.dumps(report['metrics']['checks'], indent=2))
    return 0 if report['metrics']['status'] == 'qualified' else 1


if __name__ == '__main__':
    raise SystemExit(main())

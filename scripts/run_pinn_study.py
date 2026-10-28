"""Run a predeclared PINN robustness study."""
import argparse
import json
from pathlib import Path
from physgraph.experiments.pinn_study import run_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        settings = json.loads(args.config.read_text())
        result = run_study(**settings, output=args.output)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result['metrics']['groups'], indent=2))
    if result['metrics']['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

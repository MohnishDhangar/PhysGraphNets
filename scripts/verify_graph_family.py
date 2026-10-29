"""Verify the parameterized square family before graph training."""
import argparse
import json
from pathlib import Path
from physgraph.experiments.parametric_square import verify_family


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resolutions', type=int, nargs='+', default=[8, 16, 32])
    args = parser.parse_args()
    try:
        result = verify_family(**vars(args))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result['metrics'], indent=2))
    return 0 if result['metrics']['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

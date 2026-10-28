"""Run the optional FEniCSx comparison; missing dependencies exit nonzero."""
import argparse
from pathlib import Path
from physgraph.experiments.fenicsx import run_comparisons
from physgraph.verification.cases import CASE_IDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=CASE_IDS)
    parser.add_argument('--resolutions', nargs='+', type=int, default=[4, 8])
    parser.add_argument('--tolerance', type=float, default=1e-8)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        manifest = run_comparisons(**vars(args))
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        parser.error(str(exc))
    print(manifest['metrics']['status'])
    print(manifest['metrics']['reason'] or '')
    print('Manifest:', manifest['config']['output'])
    return 0 if manifest['metrics']['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())

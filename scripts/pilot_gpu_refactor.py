"""Run one fine-mesh vendor sparse-refactorization pilot."""
import argparse
from physgraph.profiling.refactor_pilot import run_refactor_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--case',required=True)
    report = run_refactor_pilot(**vars(parser.parse_args()))
    print(report['metrics']['status'], report['metrics'].get('reason',''))
    return 0 if report['metrics']['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

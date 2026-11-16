"""Run real GPU sparse linear solves against CPU reference fixtures."""
import argparse
from physgraph.experiments.gpu_solver_checks import run_gpu_solver_checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',required=True)
    parser.add_argument('--output',required=True)
    report=run_gpu_solver_checks(**vars(parser.parse_args()))
    return 0 if report['metrics']['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())

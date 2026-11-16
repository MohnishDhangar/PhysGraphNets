"""Synchronized per-operation profile; deliberately separate from timing trials."""
import argparse
from physgraph.profiling.gpu_solver_profile import profile_gpu_solver


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',required=True)
    parser.add_argument('--output',required=True)
    report=profile_gpu_solver(**vars(parser.parse_args()))
    return 0 if report['metrics']['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())

"""Compare bounded complete cylinder trajectories with SciPy and MUMPS."""
import argparse
from physgraph.experiments.flow_backend_comparison import run_backend_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--level',type=int,default=16)
    parser.add_argument('--dt',type=float,default=.005)
    parser.add_argument('--steps',type=int,default=6)
    parser.add_argument('--repeats',type=int,default=3)
    report = run_backend_comparison(**vars(parser.parse_args()))
    print(report['metrics']['status'])
    return 0 if report['metrics']['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

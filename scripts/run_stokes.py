"""Run the CPU Stokes verification milestone."""
import argparse
from physgraph.experiments.stokes import run_stokes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--viscosity', type=float, default=1.)
    parser.add_argument('--fenicsx', action='store_true')
    args = parser.parse_args()
    report = run_stokes(output=args.output, sizes=args.sizes, viscosity=args.viscosity, external=args.fenicsx)
    for row in report['metrics']['rows']:
        print(f"{row['case']} n={row['n']}: passed={row['passed']}, u L2={row['velocity_l2']:.3e}, p L2={row['pressure_l2']:.3e}")
    print(report['metrics']['status'])
    return 0 if report['metrics']['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

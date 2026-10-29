"""Run manufactured transient Navier-Stokes qualification on CPU."""
import argparse
from physgraph.experiments.navier_stokes import run_navier_stokes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[4, 8, 16])
    parser.add_argument('--time-steps', nargs='+', type=int, default=[8, 16, 32])
    parser.add_argument('--fenicsx', action='store_true')
    args = parser.parse_args()
    report = run_navier_stokes(output=args.output, sizes=args.sizes, time_steps=args.time_steps, external=args.fenicsx)
    for row in report['metrics']['rows']:
        print(f"{row['phase']} {row['case']['name']} {row['scheme']} {row['traction_tag'] or 'full'} "
              f"n={row['n']} dt={row['dt']}: passed={row['passed']}, errors={row['final_errors']}")
    for row in report['metrics']['energy_rows']:
        print(f"energy {row['scheme']}: passed={row['passed']}, balance={row['max_absolute_balance']:.3e}")
    print(report['metrics']['status'])
    return 0 if report['metrics']['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

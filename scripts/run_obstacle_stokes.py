"""Run the finite-domain no-slip obstacle Stokes qualification."""
import argparse
from physgraph.experiments.obstacle_stokes import run_obstacle_stokes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--levels', nargs='+', type=int, default=[4, 8, 16])
    parser.add_argument('--cases', nargs='+', default=['annular_cylinder', 'channel_cylinder', 'airfoil'])
    parser.add_argument('--fenicsx', action='store_true')
    parser.add_argument('--no-domain-study', action='store_true')
    parser.add_argument('--airfoil-angular-bias', type=float, default=.2,
                        help='endpoint ray clustering for the airfoil (default: 0.2; uniform: 1)')
    args = parser.parse_args()
    report = run_obstacle_stokes(output=args.output, levels=args.levels, cases=args.cases,
                                external=args.fenicsx, domain_study=not args.no_domain_study,
                                airfoil_angular_bias=args.airfoil_angular_bias)
    for row in report['metrics']['rows']+report['metrics']['domain_rows']:
        print(f"{row['case']} level={row['level']} R={row['parameters']['outer_radius']}: "
              f"passed={row['passed']}, resistance={row['reaction_resistance']}, "
              f"force disagreement={row['relative_force_disagreement']:.3e}")
    print(report['metrics']['status'])
    return 0 if report['metrics']['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Qualify mixed velocity/traction Stokes fields, boundary fluxes and loads."""
import argparse
from physgraph.experiments.stokes_mixed import run_mixed_stokes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--viscosity', type=float, default=.3)
    parser.add_argument('--pressure-offset', type=float, default=2.5)
    parser.add_argument('--fenicsx', action='store_true')
    args = parser.parse_args()
    report = run_mixed_stokes(output=args.output, sizes=args.sizes, viscosity=args.viscosity,
                             pressure_offset=args.pressure_offset, external=args.fenicsx)
    for row in report['metrics']['rows']:
        print(f"{row['case']} traction={row['traction_tag']} n={row['n']}: passed={row['passed']}, "
              f"u L2={row['velocity_l2']:.3e}, p L2={row['pressure_l2']:.3e}, "
              f"net flux={row['boundary_integrals']['net_flux']:.3e}")
    print(report['metrics']['status'])
    return 0 if report['metrics']['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

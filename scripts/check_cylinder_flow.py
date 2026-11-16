"""Short cylinder development check; does not establish wake qualification."""
import argparse
from physgraph.experiments.cylinder_flow import run_cylinder_flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fenicsx', action='store_true', help='independently compare all ten smoke-test steps')
    parser.add_argument('--linear-backend', choices=('scipy', 'mumps', 'rocm_gmres','cpu_cached','rocm_cached'), default='scipy')
    args = parser.parse_args()
    report = run_cylinder_flow(output=args.output, level=2, dt=.02, final_time=.2,
                               external=args.fenicsx, linear_backend=args.linear_backend)
    print(report['metrics']['status'])
    print('Published wake benchmark:', report['metrics']['published']['status'])
    print('Seconds by phase:', report['metrics']['timings_seconds'])
    return 1 if report['metrics']['status'].startswith('failed') else 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Run or resume a manufactured CPU flow experiment with bounded trajectory memory."""
import argparse
from physgraph.experiments.flow_stream import run_manufactured_flow_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--case', dest='name', choices=['accelerating_shear','vortex','translated_vortex'],
                        default='accelerating_shear')
    parser.add_argument('--time-profile', choices=['linear','exponential'], default='linear')
    parser.add_argument('--dt', type=float, default=.025)
    parser.add_argument('--total-steps', type=int, default=16)
    parser.add_argument('--chunk-steps', type=int, default=4)
    parser.add_argument('--scheme', choices=['be','bdf2'], default='bdf2')
    parser.add_argument('--traction-tag', choices=['left','right','top','bottom'])
    parser.add_argument('--resume', action='store_true')
    report = run_manufactured_flow_stream(**vars(parser.parse_args()))
    print(report['metrics']['status'])
    print(report['metrics']['final_errors'])
    return 0 if report['metrics']['status'] == 'passed_discrete_checks' else 1


if __name__ == '__main__':
    raise SystemExit(main())

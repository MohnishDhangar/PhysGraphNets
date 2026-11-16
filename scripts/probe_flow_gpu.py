"""Probe optional ROCm float64 sparse arithmetic; does not run a GPU flow solver."""
import argparse
from physgraph.experiments.flow_gpu_probe import run_gpu_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sdk-site', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=30)
    report = run_gpu_probe(**vars(parser.parse_args()))
    print(report['metrics']['status'])
    return 0 if report['metrics']['status']=='passed_capability_probe' else 1


if __name__ == '__main__':
    raise SystemExit(main())

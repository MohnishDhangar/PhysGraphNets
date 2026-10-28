"""Train and evaluate the CPU PINN baseline, with optional exact-state resume."""
import argparse
import json
from pathlib import Path
from physgraph.experiments.pinn import run_experiment
from physgraph.physics.pinn import CASE_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--case', choices=CASE_NAMES)
    parser.add_argument('--backend', choices=('numpy', 'torch'))
    parser.add_argument('--torch-threads', type=int)
    for name in ('width', 'depth', 'collocation', 'seed', 'steps', 'nx', 'checkpoint-every', 'neumann-points', 'validation-seed', 'holdout-points', 'boundary-points', 'evaluation-quadrature-order'):
        parser.add_argument('--'+name, type=int)
    parser.add_argument('--acceptance', type=json.loads, help='JSON object of maximum permitted errors')
    parser.add_argument('--neumann-weight', type=float)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path)
    args = vars(parser.parse_args())
    config_path = args.pop('config')
    try:
        config = json.loads(config_path.read_text()) if config_path else {}
        if not isinstance(config, dict) or set(config)-set(args):
            raise ValueError('config must contain only supported CLI settings')
        config.update({key: value for key, value in args.items() if value is not None})
        manifest = run_experiment(**config)
    except (ValueError, TypeError, OSError, FloatingPointError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest['metrics'], indent=2))
    print('Saved experiment:', manifest['config']['output'])


if __name__ == '__main__':
    main()

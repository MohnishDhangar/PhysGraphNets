"""Run a Phase 2A CPU elliptic case, with optional mesh convergence."""
import argparse
import json
from pathlib import Path

from physgraph.experiments.elliptic import run_experiment
from physgraph.physics.diffusion import CASE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON defaults; explicit CLI values override")
    parser.add_argument("--case", choices=CASE_NAMES)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int, help="Defaults to the resolved nx")
    parser.add_argument("--grading", type=float, help="1 = uniform; >1 clusters near x=y=0")
    parser.add_argument("--output", type=Path, help="Empty directory; default creates a unique run")
    parser.add_argument("--convergence", nargs="+", type=int, help="Square-mesh refinement sizes")
    parser.add_argument("--quadrature-order", type=int)
    parser.add_argument("--error-quadrature-order", type=int)
    args = vars(parser.parse_args())
    config_path = args.pop("config")
    try:
        config = json.loads(config_path.read_text()) if config_path else {}
        if not isinstance(config, dict) or set(config) - set(args):
            raise ValueError("config must be a JSON object containing only supported CLI settings")
        config.update({key: value for key, value in args.items() if value is not None})
        manifest = run_experiment(**config)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest["metrics"], indent=2))
    print(f"Saved experiment: {manifest['config']['output']}")


if __name__ == "__main__":
    main()

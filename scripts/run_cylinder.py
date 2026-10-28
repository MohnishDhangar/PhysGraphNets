"""Verify diffusion/reaction on a rectangle containing a cylinder."""
import argparse
import json
from pathlib import Path

from physgraph.experiments.cylinder import run_experiment
from physgraph.physics.cylinder import BOUNDARY_MODES, CASES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON defaults; explicit CLI settings override")
    parser.add_argument("--nr", type=int, help="Radial cell count")
    parser.add_argument("--nside", type=int, help="Segments per rectangle side; defaults to side-ratio * nr")
    parser.add_argument("--side-ratio", type=int, help="Side segments per radial cell in refinement studies")
    parser.add_argument("--radius", type=float)
    parser.add_argument("--bounds", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--grading", type=float)
    parser.add_argument("--diffusivity", type=float)
    parser.add_argument("--boundary-mode", choices=BOUNDARY_MODES)
    parser.add_argument("--convergence", type=int, nargs="+", help="Refine radial layers and side segments together")
    parser.add_argument("--quadrature-order", type=int)
    parser.add_argument("--error-quadrature-order", type=int)
    parser.add_argument("--output", type=Path, help="Empty directory; default creates a unique run")
    args = vars(parser.parse_args())
    config_path = args.pop("config")
    try:
        config = json.loads(config_path.read_text()) if config_path else {}
        if not isinstance(config, dict) or set(config)-set(args):
            raise ValueError("config must be a JSON object containing only supported CLI settings")
        config.update({key: value for key, value in args.items() if value is not None})
        manifest = run_experiment(**config)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest["metrics"], indent=2))
    print(f"Saved experiment: {manifest['config']['output']}")


if __name__ == "__main__":
    main()

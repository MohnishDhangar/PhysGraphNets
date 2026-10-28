"""Phase 1 CLI: solves Poisson, validates against exact field, exports plots."""
import argparse
import json
from pathlib import Path

from physgraph.experiments.pipeline import run_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON experiment config")
    parser.add_argument("--nx", type=int, help="Mesh cells along x")
    parser.add_argument("--ny", type=int, help="Mesh cells along y; defaults to nx")
    parser.add_argument("--case", choices=["sine", "polynomial", "affine"])
    parser.add_argument("--output", type=Path, help="Results directory")
    parser.add_argument("--show-edges", action="store_true")
    args = parser.parse_args()
    settings = {}
    if args.config:
        settings = json.loads(args.config.read_text())
    nx = args.nx if args.nx is not None else settings.get("nx", 16)
    ny = args.ny if args.ny is not None else settings.get("ny", nx)
    case = args.case or settings.get("case", "sine")
    output = args.output or Path(settings.get("output", "outputs/phase1_local"))
    metrics = run_one(nx, ny, case, output, show_edges=args.show_edges)
    print(json.dumps(metrics, indent=2))
    print(f"Saved field.npz, metrics.json, comparison.png in: {output.resolve()}")


if __name__ == "__main__":
    main()

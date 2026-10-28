"""Convergence study on a sequence of uniformly refined unit-square meshes."""
import argparse
import json
from pathlib import Path

from physgraph.experiments.pipeline import run_convergence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--case", choices=["sine", "polynomial", "affine"],
                        default="sine")
    parser.add_argument("--output", type=Path, default=Path("outputs/convergence"))
    args = parser.parse_args()
    rows = run_convergence(args.sizes, args.case, args.output)
    print(json.dumps(rows, indent=2))
    print(f"Saved convergence.csv and convergence.png in: {args.output.resolve()}")


if __name__ == "__main__":
    main()

"""Check block-preconditioner convergence on archived CPU FEM systems."""
import argparse
from physgraph.profiling.block_amg_pilot import run_pilot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',required=True);p.add_argument('--output',required=True)
    rows=run_pilot(**vars(p.parse_args()))
    return 0 if all(r['passed'] for r in rows) else 1


if __name__=='__main__':raise SystemExit(main())

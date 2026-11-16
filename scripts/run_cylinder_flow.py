"""Run a DFG 2D-3 cylinder trajectory and preserve all acceptance results."""
import argparse
from physgraph.experiments.cylinder_flow import run_cylinder_flow


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--level',type=int,default=4)
    parser.add_argument('--dt',type=float,default=.02)
    parser.add_argument('--final-time',type=float,default=8.)
    parser.add_argument('--fenicsx',action='store_true')
    parser.add_argument('--linear-backend',choices=('scipy','mumps','rocm_gmres','cpu_cached','rocm_cached'),default='scipy')
    args=parser.parse_args()
    report=run_cylinder_flow(output=args.output,level=args.level,dt=args.dt,
        final_time=args.final_time,external=args.fenicsx,linear_backend=args.linear_backend)
    metrics=report['metrics']
    print(metrics['status'])
    print({k:v for k,v in metrics['history'].items() if k!='rows'})
    print(metrics['published'])
    return 1 if metrics['status'].startswith('failed') else 0


if __name__=='__main__':
    raise SystemExit(main())

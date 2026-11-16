"""Prepare archived Newton systems or benchmark isolated CPU solver candidates."""
import argparse
from physgraph.profiling.flow_linear import prepare_systems
from physgraph.experiments.flow_acceleration import run_linear_study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    prepare=sub.add_parser('prepare')
    prepare.add_argument('--trajectory',required=True)
    prepare.add_argument('--output',required=True)
    prepare.add_argument('--steps',type=int,nargs='+',default=[1,1201])
    run=sub.add_parser('run')
    run.add_argument('--fixtures',required=True)
    run.add_argument('--output',required=True)
    run.add_argument('--backends',nargs='+',default=['scipy','mumps','superlu_dist'])
    run.add_argument('--threads',type=int,nargs='+',default=[1,2,4,6])
    run.add_argument('--repeats',type=int,default=3)
    args=vars(parser.parse_args()); action=args.pop('action')
    report=prepare_systems(**args) if action=='prepare' else run_linear_study(**args)
    print(report['metrics']['status'])
    return 0 if report['metrics']['status'] in ('prepared','completed') else 1


if __name__=='__main__':
    raise SystemExit(main())

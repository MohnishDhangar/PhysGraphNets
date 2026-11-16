"""Prepare bounded seeds, run one segment, or compare CPU/GPU candidates."""
import argparse
from physgraph.profiling.flow_segments import prepare_flow_segments
from physgraph.experiments.gpu_flow_segment import run_flow_segment
from physgraph.experiments.gpu_flow_study import run_gpu_flow_study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    prepare=sub.add_parser('prepare')
    prepare.add_argument('--trajectory',required=True)
    prepare.add_argument('--output',required=True)
    prepare.add_argument('--starts',type=int,nargs='+',default=[0,1200])
    prepare.add_argument('--steps',type=int,default=3)
    segment=sub.add_parser('segment')
    segment.add_argument('--seed',required=True)
    segment.add_argument('--output',required=True)
    segment.add_argument('--backend',choices=('scipy','mumps','rocm_gmres','cpu_cached','rocm_cached'),required=True)
    study=sub.add_parser('study')
    study.add_argument('--seeds',required=True)
    study.add_argument('--output',required=True)
    study.add_argument('--repeats',type=int,default=3)
    study.add_argument('--baseline',choices=('mumps','cpu_cached'),default='mumps')
    study.add_argument('--candidate',choices=('rocm_gmres','rocm_cached'),default='rocm_gmres')
    args=vars(parser.parse_args());action=args.pop('action')
    report={'prepare':prepare_flow_segments,'segment':run_flow_segment,'study':run_gpu_flow_study}[action](**args)
    print(report['metrics']['status'])
    return 0 if report['metrics']['status'] in ('prepared','passed','candidate_accepted') else 1


if __name__=='__main__':
    raise SystemExit(main())

"""Run fixed validation-only GNN repair ablations."""
import argparse
from physgraph.experiments.gnn_repair import run_ablations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','normalization','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--epochs',type=int,default=300)
    args=parser.parse_args()
    print(run_ablations(**vars(args))['metrics']['status'])


if __name__=='__main__':main()

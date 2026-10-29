"""Train the supervised CPU graph baseline."""
import argparse
from physgraph.training.gnn import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'normalization', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--resume')
    args = parser.parse_args()
    result = train(**vars(args))
    print(result['metrics']['history'][-1])


if __name__ == '__main__':
    main()

"""Check graph data, FEM labels and preprocessing before model training."""
import argparse
import json
from pathlib import Path
from physgraph.experiments.graph_readiness import audit_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, help='JSON audit thresholds and instance minima')
    args = parser.parse_args()
    try:
        options = {} if args.config is None else json.loads(args.config.read_text())
        result = audit_dataset(dataset=args.dataset, output=args.output, **options)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    metrics = result['metrics']
    print(json.dumps({k: v for k, v in metrics.items() if k != 'samples'}, indent=2))
    return 0 if metrics['status'] == 'ready' else 1


if __name__ == '__main__':
    raise SystemExit(main())

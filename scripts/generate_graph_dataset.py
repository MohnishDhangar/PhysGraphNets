"""Generate declared PDE instances with CPU FEM graph labels."""
import argparse
import json
from pathlib import Path
from physgraph.datasets.generate import generate_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    try:
        result=generate_dataset(**json.loads(args.config.read_text()),output=args.output)
    except (ValueError,TypeError,OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result['metrics']['instance_counts'],indent=2))
    print('Saved samples:',len(result['metrics']['samples']))


if __name__=='__main__':
    main()

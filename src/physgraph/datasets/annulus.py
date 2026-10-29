"""Compatibility entry point for the annulus FEM graph dataset generator."""
from physgraph.datasets.generate import generate_dataset as _generate


def generate_dataset(*, instances, resolutions, output, split_seed=0):
    return _generate(instances=instances, resolutions=resolutions, output=output,
                     split_seed=split_seed, family='annulus_log')

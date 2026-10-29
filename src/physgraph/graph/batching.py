"""Dependency-free disjoint graph batching with separate physical FEM targets."""
from dataclasses import dataclass
import numpy as np

from physgraph.graph.mesh import frozen_array


@dataclass(frozen=True)
class GraphBatch:
    node_features: np.ndarray
    edge_features: np.ndarray
    edge_index: np.ndarray
    triangles: np.ndarray
    graph_ptr: np.ndarray
    edge_ptr: np.ndarray
    triangle_ptr: np.ndarray
    graph_index: np.ndarray
    node_area: np.ndarray
    dirichlet_mask: np.ndarray
    dirichlet_values: np.ndarray
    fem_solution: np.ndarray
    instance_ids: tuple[str, ...]
    resolutions: tuple[int, ...]
    split: str


def batch_graphs(samples, normalizer):
    samples = tuple(samples)
    if not samples or len({s.split for s in samples}) != 1:
        raise ValueError('a batch requires samples from exactly one split')
    keys = [(s.instance_id, s.resolution) for s in samples]
    if len(set(keys)) != len(keys):
        raise ValueError('a batch cannot duplicate an instance/resolution sample')
    graph_ptr = np.cumsum([0]+[len(s.graph.nodes) for s in samples], dtype=np.int64)
    edge_ptr = np.cumsum([0]+[len(s.graph.length) for s in samples], dtype=np.int64)
    triangle_ptr = np.cumsum([0]+[len(s.mesh.triangles) for s in samples], dtype=np.int64)
    features = [normalizer.transform(s) for s in samples]
    arrays = [np.concatenate([f[0] for f in features]), np.concatenate([f[1] for f in features]),
              np.concatenate([s.graph.edge_index+offset for s, offset in zip(samples, graph_ptr)], axis=1),
              np.concatenate([s.mesh.triangles+offset for s, offset in zip(samples, graph_ptr)]),
              graph_ptr, edge_ptr, triangle_ptr, np.repeat(np.arange(len(samples)), np.diff(graph_ptr)),
              np.concatenate([s.graph.node_area for s in samples]),
              np.concatenate([s.inputs['dirichlet_mask'] for s in samples]),
              np.concatenate([s.inputs['dirichlet_values'] for s in samples]),
              np.concatenate([s.fem_solution for s in samples])]
    return GraphBatch(*(frozen_array(a) for a in arrays),
                      tuple(s.instance_id for s in samples), tuple(s.resolution for s in samples), samples[0].split)


def iter_batches(dataset, split, normalizer, *, batch_size=4, shuffle=False, seed=0, epoch=0, resolutions=None):
    """Deterministic epoch shuffling using a private RNG; retains the final batch."""
    if (type(batch_size) is not int or batch_size < 1 or type(shuffle) is not bool
            or any(type(v) is not int or v < 0 for v in (seed, epoch))):
        raise ValueError('require positive batch size, boolean shuffle and nonnegative seed/epoch')
    if split not in ('train', 'validation', 'test'):
        raise ValueError('unknown dataset split')
    if shuffle and split != 'train':
        raise ValueError('only training batches may be shuffled')
    available = dataset.manifest['config']['resolutions']
    if resolutions is None:
        resolutions = normalizer.provenance['fit_resolutions'] if split == 'train' else available
    if (not resolutions or any(type(n) is not int or n not in available for n in resolutions)
            or len(set(resolutions)) != len(resolutions)):
        raise ValueError('batch resolutions must be a nonempty subset of the dataset')
    if shuffle and not set(resolutions) <= set(normalizer.provenance['fit_resolutions']):
        raise ValueError('training shuffle cannot include reserved evaluation resolutions')
    indices = np.array([i for i, row in enumerate(dataset.rows)
                        if row['split'] == split and row['resolution'] in resolutions], dtype=np.int64)
    if not len(indices):
        raise ValueError(f'{split} split is empty')
    if shuffle:
        np.random.default_rng(np.random.SeedSequence([seed, epoch])).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield batch_graphs((dataset.load(int(i)) for i in indices[start:start+batch_size]), normalizer)

"""Training-only physical-feature statistics; geometry and FEM targets stay raw."""
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
import numpy as np

from physgraph.datasets.loading import GraphDataset, strict_json
from physgraph.graph.mesh import frozen_array

NODE_COLUMNS = ('x', 'y', 'diffusion', 'reaction', 'forcing', 'dirichlet_value',
                'dirichlet_mask', 'boundary', 'node_area')
EDGE_COLUMNS = ('dx', 'dy', 'length', 'normal_x', 'normal_y', 'neumann_flux',
                'neumann_mask', 'boundary')
GROUP_WIDTHS = {'coefficients': 3, 'dirichlet_values': 1, 'neumann_flux': 1}


def fit_provenance(dataset, resolutions=None):
    available = dataset.manifest['config']['resolutions']
    resolutions = sorted(available if resolutions is None else resolutions)
    if (not resolutions or any(type(n) is not int or n not in available for n in resolutions)
            or len(set(resolutions)) != len(resolutions)):
        raise ValueError('normalization resolutions must be a nonempty subset of the dataset')
    rows = [r for r in dataset.rows if r['split'] == 'train' and r['resolution'] in resolutions]
    return {'dataset_sha256': dataset.sha256,
            'fit_resolutions': resolutions,
            'fit_instance_ids': sorted({r['instance_id'] for r in rows}),
            'fit_archives': {r['file']: r['sha256'] for r in rows}}


@dataclass(frozen=True)
class FeatureNormalizer:
    statistics: MappingProxyType
    provenance: dict
    instance_splits: MappingProxyType

    def transform(self, sample):
        if self.instance_splits.get(sample.instance_id) != sample.split:
            raise ValueError('sample identity/split is outside the normalization dataset')
        transformed = {}
        for key in GROUP_WIDTHS:
            mean, scale = self.statistics[key]
            values = sample.inputs[key]
            transformed[key] = (values-mean)/scale
        # Missing BC values remain exactly zero; presence is carried explicitly.
        prescribed = sample.inputs['dirichlet_mask']
        neumann = sample.inputs['neumann_mask']
        graph = sample.graph
        node = np.column_stack((graph.nodes, transformed['coefficients'],
            np.where(prescribed, transformed['dirichlet_values'], 0.), prescribed,
            graph.boundary, graph.node_area))
        edge = np.column_stack((graph.displacement, graph.length, graph.outward_normal,
            np.where(neumann, transformed['neumann_flux'], 0.), neumann, graph.edge_boundary))
        if not np.isfinite(node).all() or not np.isfinite(edge).all():
            raise ValueError('nonfinite normalized features')
        return frozen_array(node), frozen_array(edge)

    def save(self, path):
        payload = {'schema_version': 1, 'artifact': 'graph_feature_normalization',
                   'node_columns': list(NODE_COLUMNS), 'edge_columns': list(EDGE_COLUMNS),
                   'weighting': 'equal instance, equal resolution; nodal area / boundary support',
                   'targets': 'raw FEM values, never used to fit input normalization',
                   **self.provenance,
                   'statistics': {k: {'mean': m.tolist(), 'scale': s.tolist()}
                                  for k, (m, s) in self.statistics.items()}}
        # Exclusive creation preserves an earlier fit as experiment evidence.
        with Path(path).open('x') as stream:
            stream.write(json.dumps(payload, indent=2, allow_nan=False)+'\n')


def fit_normalizer(dataset: GraphDataset, *, resolutions=None):
    provenance = fit_provenance(dataset, resolutions)
    if not provenance['fit_instance_ids']:
        raise ValueError('normalization requires training instances')
    counts = {identifier: sum(r['instance_id'] == identifier and r['resolution'] in provenance['fit_resolutions'] for r in dataset.rows)
              for identifier in provenance['fit_instance_ids']}
    groups = {key: [] for key in GROUP_WIDTHS}
    for index, row in enumerate(dataset.rows):
        if row['split'] != 'train' or row['resolution'] not in provenance['fit_resolutions']:
            continue
        sample = dataset.load(index)
        inputs, graph = sample.inputs, sample.graph
        supported = {
            'coefficients': (inputs['coefficients'], graph.node_area),
            'dirichlet_values': (inputs['dirichlet_values'][inputs['dirichlet_mask'], None],
                                 np.ones(np.count_nonzero(inputs['dirichlet_mask']))),
            'neumann_flux': (inputs['neumann_flux'][inputs['neumann_mask'], None],
                             graph.length[inputs['neumann_mask']])}
        for key, (values, weights) in supported.items():
            if len(weights):
                weights = weights/weights.sum()/counts[sample.instance_id]
                groups[key].append((values, weights))
    statistics = {}
    for key, width in GROUP_WIDTHS.items():
        if not groups[key]:
            mean, scale = np.zeros(width), np.ones(width)
        else:
            total = sum(w.sum() for _, w in groups[key])
            mean = sum((v*w[:, None]).sum(axis=0) for v, w in groups[key])/total
            variance = sum(((v-mean)**2*w[:, None]).sum(axis=0) for v, w in groups[key])/total
            scale = np.sqrt(variance)
            scale = np.where(scale < 1e-12, 1., scale)
        statistics[key] = (frozen_array(mean), frozen_array(scale))
    return FeatureNormalizer(MappingProxyType(statistics), provenance, dataset.instance_splits)


def load_normalizer(path, dataset: GraphDataset):
    payload = strict_json(Path(path).read_text())
    try:
        provenance = fit_provenance(dataset, payload['fit_resolutions'])
        if (type(payload['schema_version']) is not int or payload['schema_version'] != 1
                or payload['artifact'] != 'graph_feature_normalization'
                or payload['node_columns'] != list(NODE_COLUMNS) or payload['edge_columns'] != list(EDGE_COLUMNS)
                or any(payload[k] != v for k, v in provenance.items()) or not provenance['fit_instance_ids']
                or set(payload['statistics']) != set(GROUP_WIDTHS)):
            raise ValueError('normalization schema/provenance does not match training instances')
        statistics = {}
        for key, width in GROUP_WIDTHS.items():
            mean = np.asarray(payload['statistics'][key]['mean'], dtype=float)
            scale = np.asarray(payload['statistics'][key]['scale'], dtype=float)
            if (mean.shape != (width,) or scale.shape != (width,) or not np.isfinite(mean).all()
                    or not np.isfinite(scale).all() or np.any(scale <= 0)):
                raise ValueError('invalid normalization statistics')
            statistics[key] = (frozen_array(mean), frozen_array(scale))
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid normalization file structure') from exc
    return FeatureNormalizer(MappingProxyType(statistics), provenance, dataset.instance_splits)

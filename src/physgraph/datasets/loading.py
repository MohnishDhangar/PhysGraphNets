"""Strict, pickle-free loading of completed, checksum-protected graph datasets."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import MappingProxyType
import numpy as np

from physgraph.core.types import Mesh2D
from physgraph.datasets.catalog import resolve_instance, instance_mesh
from physgraph.datasets.generate import file_sha256
from physgraph.datasets.instances import PDEInstance, split_instances
from physgraph.graph.mesh import MeshGraph, build_graph, physical_features, frozen_array

SPLITS = ('train', 'validation', 'test')
ERROR_KEYS = {'relative_l2', 'relative_h1_seminorm', 'max_nodal_error', 'absolute_l2', 'absolute_h1_seminorm'}


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f'nonfinite JSON value: {value}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def positive_int(value, minimum=1):
    return type(value) is int and value >= minimum


def finite_nonnegative(value):
    return type(value) in (int, float) and np.isfinite(value) and value >= 0


@dataclass(frozen=True)
class GraphSample:
    instance: PDEInstance
    split: str
    resolution: int
    mesh: Mesh2D
    graph: MeshGraph
    inputs: MappingProxyType
    fem_solution: np.ndarray

    @property
    def instance_id(self):
        return self.instance.instance_id


class GraphDataset:
    """Validate the manifest eagerly and archives on access; never unpickle.

    Schema-2 dataset manifests require checksums. Earlier schema-1 sample arrays
    remain unchanged, but old manifests must be regenerated for training use.
    Reconstructing the declared mesh and physical features checks their meaning,
    not just their shapes. FEM residual validation lives in the readiness audit.
    """
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.sha256 = file_sha256(self.path)
        self.manifest = strict_json(self.path.read_text())
        try:
            self._validate_manifest()
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError('invalid graph dataset manifest structure') from exc

    def _validate_manifest(self):
        config, metrics = self.manifest['config'], self.manifest['metrics']
        if type(config.get('dataset_schema_version')) is not int or config['dataset_schema_version'] != 2:
            raise ValueError('require dataset schema 2 with archive checksums; regenerate legacy datasets')
        if metrics['status'] != 'completed':
            raise ValueError('dataset generation must be completed')
        family = config['family']
        if (config['label'] != 'fem_solution' or config['units'] != 'dimensionless'
                or config['coefficient_columns'] != ['diffusion', 'reaction', 'forcing']
                or config['assembly_order'] != (7 if family == 'parametric_square' else 4)
                or config['error_quadrature_order'] != 5
                or config['split_fractions'] != {'train': .7, 'validation': .15, 'test': .15}):
            raise ValueError('unsupported dataset physics/label/quadrature/split contract')
        resolutions = config['resolutions']
        if (not isinstance(resolutions, list) or not resolutions
                or not all(positive_int(n, 2) for n in resolutions) or len(set(resolutions)) != len(resolutions)):
            raise ValueError('invalid dataset resolutions')
        self._instances = {}
        if not isinstance(config['instances'], list) or not config['instances']:
            raise ValueError('dataset requires instances')
        for payload in config['instances']:
            if (set(payload) != {'schema_version', 'family', 'geometry', 'physics'}
                    or type(payload['schema_version']) is not int or payload['schema_version'] != 1
                    or payload['family'] != family):
                raise ValueError('invalid physical instance schema')
            spec, case = resolve_instance(family, {'geometry': payload['geometry'], 'physics': payload['physics']})
            if json.loads(spec.canonical_json) != payload or spec.instance_id in self._instances:
                raise ValueError('instance parameters must be complete, canonical and unique')
            self._instances[spec.instance_id] = (spec, case)
        splits = split_instances([s for s, _ in self._instances.values()], seed=config['split_seed'])
        counts = {s: sum(v == s for v in splits.values()) for s in SPLITS}
        if (metrics['instance_splits'] != splits or metrics['instance_counts'] != counts
                or metrics['splits_nonempty'] is not all(counts.values())):
            raise ValueError('manifest split assignments/counts disagree with physical identities')
        self.instance_splits = MappingProxyType(splits)
        rows = metrics['samples']
        if not isinstance(rows, list) or not rows:
            raise ValueError('dataset must list samples')
        seen, files = set(), set()
        for row in rows:
            key = row['instance_id'], row['resolution']
            filename = row['file']
            if (not isinstance(filename, str) or Path(filename).name != filename
                    or Path(filename).suffix != '.npz' or '\\' in filename
                    or (self.path.parent/filename).resolve().parent != self.path.parent):
                raise ValueError('sample path must stay inside the dataset directory')
            if (not positive_int(row['resolution'], 2) or key in seen or filename in files
                    or key[0] not in splits or key[1] not in resolutions or row['split'] != splits[key[0]]):
                raise ValueError('duplicate, unknown or split-inconsistent sample')
            checksum = row['sha256']
            if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
                raise ValueError('every sample requires a SHA256 checksum')
            if (not all(positive_int(row[k]) for k in ('nodes', 'triangles', 'directed_edges'))
                    or not finite_nonnegative(row['relative_residual']) or set(row['errors']) != ERROR_KEYS
                    or not all(finite_nonnegative(v) for v in row['errors'].values())):
                raise ValueError('invalid sample counts/residual/error metrics')
            seen.add(key)
            files.add(filename)
        if seen != {(identifier, n) for identifier in splits for n in resolutions}:
            raise ValueError('dataset is missing declared instance/resolution samples')
        self.rows = tuple(MappingProxyType(row) for row in rows)

    def __len__(self):
        return len(self.rows)

    def samples(self, split=None):
        if split is not None and split not in SPLITS:
            raise ValueError(f'split must be one of {SPLITS}')
        for index, row in enumerate(self.rows):
            if split is None or row['split'] == split:
                yield self.load(index)

    def load(self, index):
        row = self.rows[index]
        path = self.path.parent/row['file']
        if path.resolve().parent != self.path.parent or file_sha256(path) != row['sha256']:
            raise ValueError(f'archive checksum/path mismatch: {row["file"]}')
        spec, case = self._instances[row['instance_id']]
        mesh = instance_mesh(spec, row['resolution'])
        graph = build_graph(mesh)
        inputs = physical_features(mesh, graph, case.problem)
        expected = asdict(graph) | inputs | {'triangles': mesh.triangles}
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)) or set(archive.files) != set(expected) | {'metadata', 'fem_solution'}:
                raise ValueError('archive contains missing, duplicate or unexpected arrays')
            raw_metadata = archive['metadata']
            if raw_metadata.shape != () or raw_metadata.dtype.kind != 'U':
                raise ValueError('metadata must be a scalar Unicode JSON string')
            metadata = strict_json(raw_metadata.item())
            wanted = {'schema_version': 1, 'artifact': 'fem_graph_sample', 'instance_id': spec.instance_id,
                      'split': row['split'], 'resolution': row['resolution'], 'instance': json.loads(spec.canonical_json)}
            if (metadata != wanted or type(metadata.get('schema_version')) is not int
                    or not positive_int(metadata.get('resolution'), 2)):
                raise ValueError('sample metadata disagrees with manifest')
            for key, value in expected.items():
                actual = archive[key]
                floating = value.dtype.kind == 'f'
                if (actual.shape != value.shape or actual.dtype != value.dtype
                        or (floating and (not np.isfinite(actual).all() or not np.allclose(actual, value, rtol=2e-13, atol=2e-14)))
                        or (not floating and not np.array_equal(actual, value))):
                    raise ValueError(f'invalid {key}: differs from declared mesh/physics')
            target = archive['fem_solution']
            if target.shape != (len(mesh.nodes),) or target.dtype != np.float64 or not np.isfinite(target).all():
                raise ValueError('fem_solution must be a finite float64 nodal field')
            mask = inputs['dirichlet_mask']
            if not np.allclose(target[mask], inputs['dirichlet_values'][mask], rtol=0, atol=1e-12):
                raise ValueError('FEM target violates nonzero Dirichlet data')
            target = frozen_array(target)
        if (row['nodes'] != len(mesh.nodes) or row['triangles'] != len(mesh.triangles)
                or row['directed_edges'] != len(graph.length)):
            raise ValueError('sample counts disagree with mesh')
        # Return independent read-only arrays, never views into an archive file.
        mesh = Mesh2D(*(None if a is None else frozen_array(a) for a in
                        (mesh.nodes, mesh.triangles, mesh.boundary, mesh.boundary_edges, mesh.boundary_edge_tags)))
        return GraphSample(spec, row['split'], row['resolution'], mesh, graph,
                           MappingProxyType({k: frozen_array(v) for k, v in inputs.items()}), target)

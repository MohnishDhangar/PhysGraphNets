"""Parameterized graph datasets labelled exclusively by the CPU FEM solver."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from physgraph.datasets.catalog import resolve_instance, instance_mesh
from physgraph.datasets.instances import split_instances
from physgraph.graph.mesh import build_graph, physical_features
from physgraph.fem.elliptic import solve
from physgraph.fem.error import field_errors
from physgraph.io.manifest import write_manifest


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def generate_dataset(*, instances, resolutions, output, split_seed=0, family='annulus_log'):
    if not isinstance(instances, list) or not instances:
        raise ValueError('instances must be a nonempty list')
    if (not isinstance(resolutions, list) or not resolutions
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in resolutions)
            or len(set(resolutions)) != len(resolutions)):
        raise ValueError('resolutions must be distinct integers >=2')
    specs = [resolve_instance(family, item) for item in instances]
    if len({s.instance_id for s, _ in specs}) != len(specs):
        raise ValueError('duplicate physical instances; put mesh refinements in resolutions')
    splits = split_instances([s for s, _ in specs], seed=split_seed)
    dest = Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('dataset output must be empty')
    dest.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    assembly_order = 7 if family == 'parametric_square' else 4
    config = {'dataset_schema_version': 2, 'family': family,
              'instances': [json.loads(s.canonical_json) for s, _ in specs],
              'resolutions': resolutions, 'split_seed': split_seed,
              'split_fractions': {'train': .7, 'validation': .15, 'test': .15},
              'units': 'dimensionless', 'assembly_order': assembly_order, 'error_quadrature_order': 5,
              'label': 'fem_solution', 'coefficient_columns': ['diffusion', 'reaction', 'forcing'],
              'neumann_flux_sampling': 'facet midpoint, actual polygon outward normal; not integrated FEM load'}
    rows = []

    def save(status, reason=None):
        return write_manifest(dest/'dataset_manifest.json', config, {
            'status': status, 'reason': reason, 'samples': rows,
            'splits_nonempty': all(s in splits.values() for s in ('train', 'validation', 'test')),
            'instance_splits': splits,
            'instance_counts': {s: sum(v == s for v in splits.values()) for s in ('train', 'validation', 'test')}
        }, perf_counter()-started)

    save('running')
    try:
        for spec, case in specs:
            for n in resolutions:
                mesh = instance_mesh(spec, n)
                graph = build_graph(mesh)
                inputs = physical_features(mesh, graph, case.problem)
                reference = solve(mesh, case.problem, assembly_order)
                metrics = field_errors(mesh, reference.values, case.reference)
                filename = f'{spec.instance_id}_n{n}.npz'
                metadata = {'schema_version': 1, 'artifact': 'fem_graph_sample', 'instance_id': spec.instance_id,
                            'split': splits[spec.instance_id], 'resolution': n, 'instance': json.loads(spec.canonical_json)}
                np.savez_compressed(dest/filename, **asdict(graph), **inputs, triangles=mesh.triangles,
                                    fem_solution=reference.values, metadata=np.array(json.dumps(metadata, allow_nan=False)))
                rows.append({'file': filename, 'sha256': file_sha256(dest/filename),
                             'instance_id': spec.instance_id, 'split': splits[spec.instance_id],
                             'resolution': n, 'nodes': len(mesh.nodes), 'triangles': len(mesh.triangles),
                             'directed_edges': len(graph.length), 'errors': metrics,
                             'relative_residual': reference.relative_residual})
                save('running')
    except Exception as exc:
        save('failed', str(exc))
        raise
    return save('completed')

"""Pre-training data audit: numerical labels, split integrity and batching."""
from pathlib import Path
from time import perf_counter
import numpy as np

from physgraph.datasets.catalog import resolve_instance
from physgraph.datasets.loading import GraphDataset, SPLITS, strict_json
from physgraph.datasets.normalization import fit_normalizer, load_normalizer
from physgraph.fem.assembly import assemble
from physgraph.fem.operators import reduce_system
from physgraph.fem.error import field_errors
from physgraph.graph.batching import iter_batches
from physgraph.io.manifest import write_manifest


def audit_dataset(*, dataset, output, minimum_instances=None, max_relative_residual=1e-10,
                  max_relative_l2=.05, max_relative_h1=.35, batch_size=4,
                  max_quadrature_residual=1e-7, train_resolutions=None):
    """Audit against declared gates; this function does not construct/train a model."""
    minimum_instances = {'train': 8, 'validation': 3, 'test': 3} if minimum_instances is None else minimum_instances
    if (set(minimum_instances) != set(SPLITS)
            or any(type(v) is not int or v < 1 for v in minimum_instances.values())
            or type(batch_size) is not int or batch_size < 1
            or any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0
                   for v in (max_relative_residual, max_relative_l2, max_relative_h1, max_quadrature_residual))):
        raise ValueError('require positive audit tolerances, instance minima and batch size')
    dest = Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('audit output must be empty')
    dest.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    config = {'dataset': str(Path(dataset).resolve()), 'minimum_instances': minimum_instances,
              'max_relative_residual': max_relative_residual, 'max_relative_l2': max_relative_l2,
              'max_relative_h1': max_relative_h1, 'batch_size': batch_size,
              'max_quadrature_residual': max_quadrature_residual,
              'purpose': 'pre-training numerical and data-integrity gates; no model training'}
    metrics = {'status': 'running', 'failures': [], 'samples': [], 'training_started': False}
    write_manifest(dest/'manifest.json', config, metrics, 0.)
    try:
        data = GraphDataset(dataset)
        config['dataset_sha256'] = data.sha256
        counts = {s: sum(v == s for v in data.instance_splits.values()) for s in SPLITS}
        metrics['instance_counts'] = counts
        if any(counts[s] < minimum_instances[s] for s in SPLITS):
            raise ValueError(f'insufficient held-out/training instances: {counts}')
        if len(data.manifest['config']['resolutions']) < 2:
            raise ValueError('readiness requires multiple resolutions per physical instance')
        available = data.manifest['config']['resolutions']
        fitted = [min(available)] if train_resolutions is None else train_resolutions
        if (not fitted or any(type(n) is not int or n not in available for n in fitted)
                or len(set(fitted)) != len(fitted) or not set(available)-set(fitted)):
            raise ValueError('reserve at least one resolution outside the declared training resolutions')
        config['train_resolutions'] = sorted(fitted)
        config['reserved_resolutions'] = sorted(set(available)-set(fitted))
        assembly_order = data.manifest['config']['assembly_order']
        for row, sample in zip(data.rows, data.samples()):
            payload = strict_json(sample.instance.canonical_json)
            _, case = resolve_instance(payload['family'], {'geometry': payload['geometry'], 'physics': payload['physics']})
            system = reduce_system(sample.mesh, case.problem, assemble(sample.mesh, case.problem, assembly_order))
            residual = float(np.linalg.norm(system.residual(sample.fem_solution[system.interior]))
                             / max(np.linalg.norm(system.rhs), 1e-12))
            errors = field_errors(sample.mesh, sample.fem_solution, case.reference,
                                  data.manifest['config']['error_quadrature_order'])
            higher = reduce_system(sample.mesh, case.problem, assemble(sample.mesh, case.problem, assembly_order+2))
            quadrature_residual = float(np.linalg.norm(higher.residual(sample.fem_solution[higher.interior]))
                                        / max(np.linalg.norm(higher.rhs), 1e-12))
            if not np.isclose(residual, row['relative_residual'], rtol=1e-8, atol=1e-12):
                raise ValueError('recorded residual disagrees with reassembled FEM equations')
            if any(not np.isclose(v, row['errors'][k], rtol=1e-10, atol=1e-12) for k, v in errors.items()):
                raise ValueError('recorded errors disagree with recomputed field errors')
            metrics['samples'].append({'instance_id': sample.instance_id, 'split': sample.split,
                'resolution': sample.resolution, 'nodes': len(sample.mesh.nodes),
                'triangles': len(sample.mesh.triangles), 'relative_residual': residual,
                'higher_quadrature_residual': quadrature_residual, 'errors': errors})
            if (residual > max_relative_residual or errors['relative_l2'] > max_relative_l2
                    or errors['relative_h1_seminorm'] > max_relative_h1 or quadrature_residual > max_quadrature_residual):
                metrics['failures'].append(f'numerical label gate failed: {row["file"]}')
        if metrics['failures']:
            raise ValueError('one or more FEM label accuracy/residual gates failed')
        normalizer = fit_normalizer(data, resolutions=fitted)
        normalizer.save(dest/'normalization.json')
        restored = load_normalizer(dest/'normalization.json', data)
        metrics['batches'] = {}
        for split in SPLITS:
            count = 0
            for batch in iter_batches(data, split, restored, batch_size=batch_size):
                src, dst = batch.edge_index
                if not np.array_equal(batch.graph_index[src], batch.graph_index[dst]):
                    raise ValueError('batch contains an edge between different meshes')
                if not np.all(batch.graph_index[batch.triangles] == batch.graph_index[batch.triangles[:, :1]]):
                    raise ValueError('batch contains a triangle between different meshes')
                if not np.allclose(batch.fem_solution[batch.dirichlet_mask],
                                   batch.dirichlet_values[batch.dirichlet_mask], atol=1e-12, rtol=0):
                    raise ValueError('batch lost prescribed Dirichlet data')
                count += len(batch.instance_ids)
            if count != sum(row['split'] == split and (split != 'train' or row['resolution'] in fitted) for row in data.rows):
                raise ValueError('batching lost samples')
            metrics['batches'][split] = {'samples': count, 'batch_size': batch_size}
        metrics['normalization_fit_instance_ids'] = normalizer.provenance['fit_instance_ids']
        metrics['max_relative_residual'] = max(r['relative_residual'] for r in metrics['samples'])
        metrics['max_relative_l2'] = max(r['errors']['relative_l2'] for r in metrics['samples'])
        metrics['max_relative_h1'] = max(r['errors']['relative_h1_seminorm'] for r in metrics['samples'])
        metrics['max_quadrature_residual'] = max(r['higher_quadrature_residual'] for r in metrics['samples'])
        metrics['status'] = 'ready'
    except Exception as exc:
        metrics['status'] = 'failed'
        metrics['reason'] = f'{type(exc).__name__}: {exc}'
    return write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)

"""Predeclared multi-seed GNN qualification, with no test-based selection."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from physgraph.core.torch_runtime import cpu_runtime
from physgraph.datasets.generate import file_sha256
from physgraph.datasets.loading import GraphDataset
from physgraph.datasets.normalization import load_normalizer
from physgraph.evaluation.graph import prediction_metrics
from physgraph.graph.batching import batch_graphs
from physgraph.models.gnn import make_model, batch_tensors
from physgraph.training.gnn import train, source_fingerprint
from physgraph.io.manifest import write_manifest


def boundary_affine(sample):
    """Least-squares affine extension using prescribed boundary data only."""
    design = np.column_stack((np.ones(len(sample.graph.nodes)), sample.graph.nodes))
    mask = sample.inputs['dirichlet_mask']
    coefficients = np.linalg.lstsq(design[mask], sample.inputs['dirichlet_values'][mask], rcond=None)[0]
    values = design @ coefficients
    values[mask] = sample.inputs['dirichlet_values'][mask]
    return values


def summarize(rows):
    if not rows:
        raise ValueError('cannot summarize an empty evaluation group')
    return {key: {'mean': float(np.mean([r[key] for r in rows])),
                  'median': float(np.median([r[key] for r in rows])),
                  'max': float(np.max([r[key] for r in rows])),
                  'std': float(np.std([r[key] for r in rows]))} for key in rows[0]}


def assess(summary, baseline, criteria):
    failures = []
    for metric, statistics in criteria.items():
        for statistic, limit in statistics.items():
            if summary[metric][statistic] > limit:
                failures.append(f'{metric}.{statistic} exceeds {limit}')
    for metric in ('relative_fem_l2', 'relative_fem_h1_seminorm'):
        if summary[metric]['mean'] >= baseline[metric]['mean']:
            failures.append(f'{metric}: does not improve boundary-affine baseline')
    return failures


def evaluate_checkpoint(*, checkpoint, dataset, normalization, output, criteria, split="test", resolutions=None):
    data = GraphDataset(dataset)
    normal = load_normalizer(normalization, data)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = state['config']
    if (state['schema_version'] != 2 or config['dataset_sha256'] != data.sha256
            or config['normalization_sha256'] != file_sha256(normalization)
            or config['training_source_sha256'] != source_fingerprint()
            or config['torch_version'] != str(torch.__version__)
            or config['train_resolutions'] != normal.provenance['fit_resolutions']):
        raise ValueError('checkpoint provenance does not match evaluation inputs/code/runtime')
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=False)
    rows = []
    with cpu_runtime(1), torch.random.fork_rng(devices=[]), torch.no_grad():
        model = make_model(config['width'], config['depth'], config['architecture'])
        model.load_state_dict(state['best_model'])
        model.eval()
        # No validation on reserved resolution: model selection already ended.
        for sample in data.samples(split):
            if resolutions is not None and sample.resolution not in resolutions:
                continue
            prediction = model(*batch_tensors(batch_graphs([sample], normal))).numpy()
            baseline = boundary_affine(sample)
            metrics = prediction_metrics(sample, prediction)
            baseline_metrics = prediction_metrics(sample, baseline)
            rows.append({'instance_id': sample.instance_id, 'resolution': sample.resolution,
                         'nodes': len(sample.graph.nodes), 'triangles': len(sample.mesh.triangles),
                         'gnn': metrics, 'boundary_affine': baseline_metrics})
            np.savez_compressed(dest/f'{sample.instance_id}_n{sample.resolution}.npz',
                nodes=sample.graph.nodes, triangles=sample.mesh.triangles,
                fem_solution=sample.fem_solution, gnn_prediction=prediction, boundary_affine=baseline,
                metadata=np.array(json.dumps({'artifact':'gnn_evaluation','schema_version':1,
                    'instance_id':sample.instance_id,'split':split,'resolution':sample.resolution})))
    groups = {}
    for n in sorted({r['resolution'] for r in rows}):
        selected = [r for r in rows if r['resolution'] == n]
        summary = summarize([r['gnn'] for r in selected])
        baseline = summarize([r['boundary_affine'] for r in selected])
        failures = assess(summary, baseline, criteria)
        groups[str(n)] = {'instances':len(selected), 'gnn':summary, 'boundary_affine':baseline,
                          'passed':not failures, 'failures':failures}
    return {'selected_epoch':min(state['history'],key=lambda r:r['validation_loss'])['epoch'],
            'best_validation_loss':state['best_validation'], 'groups':groups, 'samples':rows,
            'passed':bool(groups) and all(g['passed'] for g in groups.values()),
            'checkpoint_sha256':file_sha256(checkpoint)}


def run_study(*, dataset, normalization, output, seeds, epochs, width, depth, batch_size,
              learning_rate, criteria, architecture="local", objective="mse", h1_weight=1.,
              decay_every=0, decay_factor=.5):
    if (not isinstance(seeds,list) or not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s < 0 for s in seeds)):
        raise ValueError('require distinct nonnegative seeds')
    allowed_metrics = {'relative_fem_l2','relative_fem_h1_seminorm','relative_equation_residual','max_dirichlet_error'}
    if set(criteria) != allowed_metrics:
        raise ValueError('declare L2, H1, residual and boundary criteria')
    for statistics in criteria.values():
        if not statistics or set(statistics)-{'mean','max'} or any(type(v) not in (float,int) or not np.isfinite(v) or v < 0 for v in statistics.values()):
            raise ValueError('invalid acceptance criteria')
    dest = Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('study output must be empty')
    dest.mkdir(parents=True,exist_ok=True)
    config = dict(dataset=str(Path(dataset).resolve()), normalization=str(Path(normalization).resolve()),
                  dataset_sha256=file_sha256(dataset), normalization_sha256=file_sha256(normalization),
                  seeds=seeds, epochs=epochs,width=width,depth=depth,batch_size=batch_size,
                  learning_rate=learning_rate,criteria=criteria,architecture=architecture,objective=objective,
                  h1_weight=h1_weight,decay_every=decay_every,decay_factor=decay_factor,
                  selection='minimum declared validation objective at fit resolution within each seed; all seeds reported',
                  stopping='fixed epoch budget; no test-based restarts')
    started = perf_counter()
    results=[]
    def save(status):
        return write_manifest(dest/'manifest.json',config,{'status':status,'runs':results,
            'passed':len(results)==len(seeds) and all(r.get('passed',False) for r in results)},perf_counter()-started)
    save('running')
    # Train ALL declared seeds before examining any test predictions.
    for seed in seeds:
        try:
            print(f'Training seed {seed}',flush=True)
            train(dataset=dataset,normalization=normalization,output=dest/f'seed_{seed}',epochs=epochs,
                  width=width,depth=depth,batch_size=batch_size,seed=seed,learning_rate=learning_rate,architecture=architecture,objective=objective,
                  h1_weight=h1_weight,decay_every=decay_every,decay_factor=decay_factor,
                  progress=lambda row: print(row,flush=True) if row['epoch']%20==0 else None)
            results.append({'seed':seed,'status':'trained'})
        except Exception as exc:
            results.append({'seed':seed,'status':'failed','reason':f'{type(exc).__name__}: {exc}','passed':False})
        save('running')
    for row in results:
        if row['status'] != 'trained':
            continue
        try:
            row.update(evaluate_checkpoint(checkpoint=dest/f'seed_{row["seed"]}'/'checkpoint.pt',
                dataset=dataset,normalization=normalization,output=dest/f'evaluation_{row["seed"]}',criteria=criteria))
            row['status']='evaluated'
        except Exception as exc:
            row.update(status='failed',reason=f'{type(exc).__name__}: {exc}',passed=False)
        save('running')
    return save('passed' if all(r.get('passed',False) for r in results) else 'failed_acceptance')

"""Supervised CPU graph training; checkpoints are saved at complete epochs."""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from physgraph.core.torch_runtime import cpu_runtime
from physgraph.datasets.loading import GraphDataset
from physgraph.datasets.normalization import load_normalizer
from physgraph.graph.batching import iter_batches
from physgraph.models.gnn import make_model, batch_tensors
from physgraph.losses.gnn import sobolev_loss, p1_errors
from physgraph.io.manifest import write_manifest


def source_fingerprint():
    source = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.rglob('*.py')):
        source.update(str(path.relative_to(root)).encode()+path.read_bytes())
    return source.hexdigest()


class _CachedTrainingData:
    """Cache validated immutable samples only in the allowed fit/selection sets."""
    def __init__(self, dataset, resolutions):
        self.rows, self.manifest = dataset.rows, dataset.manifest
        self.samples = {i: dataset.load(i) for i, row in enumerate(dataset.rows)
                        if row['split'] in ('train', 'validation') and row['resolution'] in resolutions}

    def load(self, index):
        return self.samples[index]


def supervised_loss(prediction, batch):
    target = torch.tensor(batch.fem_solution)
    area = torch.tensor(batch.node_area)
    return torch.stack([torch.sum(area[a:b]*(prediction[a:b]-target[a:b])**2)/area[a:b].sum()
                        for a, b in zip(batch.graph_ptr[:-1], batch.graph_ptr[1:])]).mean()


def train(*, dataset, normalization, output, epochs=10, width=32, depth=3,
          batch_size=4, seed=0, learning_rate=.001, resume=None, progress=None, architecture="local",
          objective="mse", h1_weight=1., decay_every=0, decay_factor=.5):
    for name, value, minimum in [('epochs', epochs, 1), ('batch_size', batch_size, 1), ('seed', seed, 0)]:
        if type(value) is not int or value < minimum:
            raise ValueError(f'invalid {name}')
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be positive and finite')
    if objective not in ('mse','sobolev') or architecture not in ('local','global_square'):
        raise ValueError('unsupported objective/architecture')
    if (type(decay_every) is not int or decay_every<0 or not 0<decay_factor<=1
            or not np.isfinite(h1_weight) or h1_weight<0):
        raise ValueError('invalid decay or H1 weight')
    data = GraphDataset(dataset)
    normal = load_normalizer(normalization, data)
    if len(normal.provenance['fit_resolutions']) != 1:
        raise ValueError('initial trainer requires exactly one training resolution per instance')
    if not all(s in data.instance_splits.values() for s in ('train', 'validation', 'test')):
        raise ValueError('all instance splits must be nonempty')
    dest = Path(output)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('output must be empty')
    dest.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    config = dict(width=width, depth=depth, batch_size=batch_size, seed=seed,
                  learning_rate=learning_rate, dataset_sha256=data.sha256,
                  normalization_sha256=hashlib.sha256(Path(normalization).read_bytes()).hexdigest(),
                  training_source_sha256=source_fingerprint(), torch_version=str(torch.__version__),
                  train_resolutions=normal.provenance['fit_resolutions'], architecture=architecture,
                  objective=objective,h1_weight=h1_weight,decay_every=decay_every,decay_factor=decay_factor)
    data = _CachedTrainingData(data, normal.provenance['fit_resolutions'])
    loss_function = (supervised_loss if objective=='mse' else
                     lambda prediction,batch: sobolev_loss(prediction,batch,h1_weight))
    history = []
    with cpu_runtime(1), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = make_model(width, depth, architecture)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, foreach=False)
        start = 0
        best = float('inf')
        if resume is not None:
            state = torch.load(resume, map_location='cpu', weights_only=True)
            if state['schema_version'] != 2 or state['config'] != config:
                raise ValueError('checkpoint configuration/provenance mismatch')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['rng'])
            start, history, best = state['epoch'], state['history'], state['best_validation']
            if epochs <= start:
                raise ValueError('target epochs must exceed saved epoch')
            best_model = state['best_model']
        else:
            best_model = None
        for epoch in range(start, epochs):
            lr=learning_rate*(decay_factor**(epoch//decay_every) if decay_every else 1.)
            for group in optimizer.param_groups:
                group['lr']=lr
            model.train()
            total = count = 0
            for batch in iter_batches(data, 'train', normal, batch_size=batch_size, shuffle=True, seed=seed, epoch=epoch):
                optimizer.zero_grad()
                loss = loss_function(model(*batch_tensors(batch)), batch)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('nonfinite gradients')
                optimizer.step()
                total += loss.item()*len(batch.instance_ids)
                count += len(batch.instance_ids)
            model.eval()
            val = number = 0
            l2_total=h1_total=0.
            with torch.no_grad():
                for batch in iter_batches(data, 'validation', normal, batch_size=batch_size,
                                          resolutions=normal.provenance['fit_resolutions']):
                    prediction=model(*batch_tensors(batch))
                    value = loss_function(prediction, batch).item()
                    metrics=p1_errors(prediction,batch).sqrt().sum(0)
                    l2_total+=metrics[0].item();h1_total+=metrics[1].item()
                    if not np.isfinite(value):
                        raise ValueError('nonfinite validation loss')
                    val += value*len(batch.instance_ids)
                    number += len(batch.instance_ids)
            val /= number
            if val < best:
                best = val
                best_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history.append(dict(epoch=epoch+1, training_loss=total/count, validation_loss=val,
                                validation_relative_l2=l2_total/number,validation_relative_h1=h1_total/number,
                                learning_rate=lr))
            if progress is not None:
                progress(history[-1])
            state = dict(schema_version=2, config=config, epoch=epoch+1, history=history,
                         model=model.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
                         best_validation=best, best_model=best_model)
            temporary = dest/'checkpoint.tmp'
            torch.save(state, temporary)
            temporary.replace(dest/'checkpoint.pt')
            write_manifest(dest/'manifest.json', config | {'epochs': epochs},
                           {'status': 'running', 'history': history, 'test_evaluated': False}, perf_counter()-started)
    return write_manifest(dest/'manifest.json', config | {'epochs': epochs},
                          {'status': 'completed', 'history': history, 'test_evaluated': False,
                           'acceptance': 'not assessed; training completion is not scientific acceptance'}, perf_counter()-started)

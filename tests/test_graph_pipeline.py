from dataclasses import replace
import json
from pathlib import Path
import shutil
import numpy as np
import pytest

from physgraph.datasets.catalog import resolve_instance
from physgraph.datasets.generate import generate_dataset, file_sha256
from physgraph.datasets.instances import split_instances
from physgraph.datasets.loading import GraphDataset
from physgraph.datasets.normalization import fit_normalizer, load_normalizer, NODE_COLUMNS, EDGE_COLUMNS
from physgraph.experiments.graph_readiness import audit_dataset
from physgraph.graph.batching import batch_graphs, iter_batches


@pytest.fixture(scope='module')
def graph_dataset_path(tmp_path_factory):
    # Small deterministic fixture with two physical instances in each split.
    counts = dict(train=0, validation=0, test=0)
    items = []
    for i in range(200):
        item = {'physics': {'offset': 1+i/20, 'amplitude_2': -.1-i/1000,
                           'boundary_mode': 'right_neumann' if i % 2 else 'dirichlet'}}
        spec, _ = resolve_instance('parametric_square', item)
        split = split_instances([spec])[spec.instance_id]
        if counts[split] < 2:
            items.append(item)
            counts[split] += 1
        if min(counts.values()) == 2:
            break
    assert min(counts.values()) == 2
    path = tmp_path_factory.mktemp('graph_dataset')/'samples'
    generate_dataset(family='parametric_square', instances=items, resolutions=[2, 3], output=path)
    return path/'dataset_manifest.json'


@pytest.fixture
def mutable_dataset(graph_dataset_path, tmp_path):
    dest = tmp_path/'data'
    shutil.copytree(graph_dataset_path.parent, dest)
    return dest/'dataset_manifest.json'


def rewrite_archive(manifest_path, change):
    manifest = json.loads(manifest_path.read_text())
    row = manifest['metrics']['samples'][0]
    path = manifest_path.parent/row['file']
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    change(arrays)
    np.savez_compressed(path, **arrays)
    row['sha256'] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest))


def test_loader_preserves_raw_targets_and_readonly_contract(graph_dataset_path):
    dataset = GraphDataset(graph_dataset_path)
    assert len(dataset) == 12
    for row, sample in zip(dataset.rows, dataset.samples()):
        assert sample.instance_id == row['instance_id']
        assert sample.split == row['split']
        with np.load(graph_dataset_path.parent/row['file'], allow_pickle=False) as archive:
            np.testing.assert_array_equal(sample.fem_solution, archive['fem_solution'])
        assert not sample.fem_solution.flags.writeable
        assert not sample.mesh.triangles.flags.writeable
        assert not sample.inputs['coefficients'].flags.writeable
        assert 'fem_solution' not in sample.inputs


def test_checksum_precedes_archive_deserialization(mutable_dataset):
    data = GraphDataset(mutable_dataset)
    path = mutable_dataset.parent/data.rows[0]['file']
    path.write_bytes(b'not an NPZ file')
    with pytest.raises(ValueError, match='checksum'):
        data.load(0)


@pytest.mark.parametrize('kind', ['nodes', 'orientation', 'edge', 'normal', 'coefficients',
                                  'nonfinite', 'integer_mask', 'boundary_target', 'extra', 'metadata'])
def test_semantic_corruption_rejected_even_with_updated_checksum(mutable_dataset, kind):
    def change(a):
        if kind == 'nodes':
            a['nodes'][0, 0] += .01
        elif kind == 'orientation':
            a['triangles'][0] = a['triangles'][0, ::-1]
        elif kind == 'edge':
            a['edge_index'][1, 0] = a['edge_index'][0, 0]
        elif kind == 'normal':
            a['outward_normal'] *= -1
        elif kind == 'coefficients':
            a['coefficients'][:, 0] *= 2
        elif kind == 'nonfinite':
            a['fem_solution'][0] = np.nan
        elif kind == 'integer_mask':
            a['dirichlet_mask'] = a['dirichlet_mask'].astype(int)
        elif kind == 'boundary_target':
            a['fem_solution'][a['dirichlet_mask']] += .2
        elif kind == 'extra':
            a['neural_prediction'] = a['fem_solution'].copy()
        elif kind == 'metadata':
            meta = json.loads(a['metadata'].item())
            meta['split'] = 'unknown'
            a['metadata'] = np.array(json.dumps(meta))
    rewrite_archive(mutable_dataset, change)
    with pytest.raises(ValueError):
        GraphDataset(mutable_dataset).load(0)


@pytest.mark.parametrize('kind', ['incomplete', 'legacy', 'duplicate', 'split', 'missing', 'path', 'identity'])
def test_manifest_contract_rejections(mutable_dataset, kind):
    manifest = json.loads(mutable_dataset.read_text())
    rows = manifest['metrics']['samples']
    if kind == 'incomplete':
        manifest['metrics']['status'] = 'running'
    elif kind == 'legacy':
        del manifest['config']['dataset_schema_version']
    elif kind == 'duplicate':
        rows.append(rows[0])
    elif kind == 'split':
        rows[0]['split'] = 'test' if rows[0]['split'] != 'test' else 'train'
    elif kind == 'missing':
        rows.pop()
    elif kind == 'path':
        rows[0]['file'] = '../outside.npz'
    elif kind == 'identity':
        manifest['config']['instances'][0]['physics']['offset'] += 1
    mutable_dataset.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        GraphDataset(mutable_dataset)


def test_normalization_reads_training_features_only(graph_dataset_path, monkeypatch):
    dataset = GraphDataset(graph_dataset_path)
    normal = fit_normalizer(dataset)
    load = dataset.load

    def training_only(index):
        assert dataset.rows[index]['split'] == 'train', 'held-out data accessed during fitting'
        sample = load(index)
        # Supervised targets cannot affect feature statistics, even if replaced.
        return replace(sample, fem_solution=np.full_like(sample.fem_solution, 1e100))

    monkeypatch.setattr(dataset, 'load', training_only)
    changed = fit_normalizer(dataset)
    assert set(normal.provenance['fit_instance_ids']) == {k for k, v in dataset.instance_splits.items() if v == 'train'}
    for key in normal.statistics:
        for before, after in zip(normal.statistics[key], changed.statistics[key]):
            np.testing.assert_array_equal(before, after)
    # Constant c has unit scale, and k's area-weighted mean is exact for affine k.
    np.testing.assert_allclose(normal.statistics['coefficients'][0][:2], [1.3, .5])
    assert normal.statistics['coefficients'][1][1] == 1.


def test_normalizer_roundtrip_and_provenance(graph_dataset_path, tmp_path):
    dataset = GraphDataset(graph_dataset_path)
    normal = fit_normalizer(dataset)
    path = tmp_path/'normalization.json'
    normal.save(path)
    loaded = load_normalizer(path, dataset)
    for sample in dataset.samples():
        before, after = normal.transform(sample), loaded.transform(sample)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
        assert after[0].shape == (len(sample.graph.nodes), len(NODE_COLUMNS))
        assert after[1].shape == (len(sample.graph.length), len(EDGE_COLUMNS))
        assert np.all(after[0][~sample.inputs['dirichlet_mask'], 5] == 0.)
        assert np.all(after[1][~sample.inputs['neumann_mask'], 5] == 0.)
    with pytest.raises(FileExistsError):
        normal.save(path)
    raw = json.loads(path.read_text())
    raw['fit_instance_ids'].append('held-out-id')
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='provenance'):
        load_normalizer(path, dataset)


def test_batches_recover_meshes_fields_and_shuffle_deterministically(graph_dataset_path):
    dataset = GraphDataset(graph_dataset_path)
    normal = fit_normalizer(dataset)
    samples = list(dataset.samples('train'))
    batch = batch_graphs(samples, normal)
    for i, sample in enumerate(samples):
        lo, hi = batch.graph_ptr[i:i+2]
        elo, ehi = batch.edge_ptr[i:i+2]
        tlo, thi = batch.triangle_ptr[i:i+2]
        np.testing.assert_array_equal(batch.edge_index[:, elo:ehi]-lo, sample.graph.edge_index)
        np.testing.assert_array_equal(batch.triangles[tlo:thi]-lo, sample.mesh.triangles)
        np.testing.assert_array_equal(batch.fem_solution[lo:hi], sample.fem_solution)
        np.testing.assert_array_equal(batch.node_features[lo:hi], normal.transform(sample)[0])
    np.testing.assert_array_equal(batch.graph_index[batch.edge_index[0]], batch.graph_index[batch.edge_index[1]])
    assert not batch.edge_index.flags.writeable

    def ordering(epoch):
        return [pair for b in iter_batches(dataset, 'train', normal, batch_size=3, shuffle=True, seed=11, epoch=epoch)
                for pair in zip(b.instance_ids, b.resolutions)]
    assert ordering(0) == ordering(0)
    assert sorted(ordering(0)) == sorted(ordering(1))
    assert ordering(0) != ordering(1)
    with pytest.raises(ValueError, match='one split'):
        batch_graphs([samples[0], next(dataset.samples('test'))], normal)
    with pytest.raises(ValueError, match='duplicate'):
        batch_graphs([samples[0], samples[0]], normal)


def test_audit_records_success_and_gates_without_training(graph_dataset_path, tmp_path):
    result = audit_dataset(dataset=graph_dataset_path, output=tmp_path/'ready',
        minimum_instances=dict(train=2, validation=2, test=2), max_relative_l2=1., max_relative_h1=2.)
    assert result['metrics']['status'] == 'ready', result['metrics']
    assert result['metrics']['training_started'] is False
    assert result['metrics']['max_relative_residual'] < 1e-12
    assert result['config']['dataset_sha256'] == file_sha256(graph_dataset_path)
    failed = audit_dataset(dataset=graph_dataset_path, output=tmp_path/'failed')
    assert failed['metrics']['status'] == 'failed'
    assert 'insufficient' in failed['metrics']['reason']


def test_audit_reassembles_and_rejects_wrong_interior_labels(mutable_dataset, tmp_path):
    def change(arrays):
        arrays['fem_solution'][~arrays['dirichlet_mask']] += .1
    rewrite_archive(mutable_dataset, change)
    # Shape/BC checks alone are insufficient; the numerical audit catches this.
    GraphDataset(mutable_dataset).load(0)
    result = audit_dataset(dataset=mutable_dataset, output=tmp_path/'audit',
                          minimum_instances=dict(train=1, validation=1, test=1))
    assert result['metrics']['status'] == 'failed'
    assert 'residual' in result['metrics']['reason']


def test_reserved_resolution_never_used_for_normalization_or_training_batches(graph_dataset_path, monkeypatch):
    dataset = GraphDataset(graph_dataset_path)
    original = dataset.load

    def selected_training_only(index):
        assert dataset.rows[index]['split'] == 'train'
        assert dataset.rows[index]['resolution'] == 2
        return original(index)
    monkeypatch.setattr(dataset, 'load', selected_training_only)
    normalizer = fit_normalizer(dataset, resolutions=[2])
    batches = list(iter_batches(dataset, 'train', normalizer, shuffle=True))
    assert all(set(b.resolutions) == {2} for b in batches)
    assert normalizer.provenance['fit_resolutions'] == [2]
    with pytest.raises(ValueError, match='reserved'):
        list(iter_batches(dataset, 'train', normalizer, resolutions=[3], shuffle=True))
    monkeypatch.setattr(dataset, 'load', original)
    evaluation = list(iter_batches(dataset, 'test', normalizer))
    assert {n for b in evaluation for n in b.resolutions} == {2, 3}


def test_nodal_evaluation_separates_fem_truth_boundary_and_equation_errors(graph_dataset_path):
    from physgraph.evaluation.graph import prediction_metrics
    sample = GraphDataset(graph_dataset_path).load(0)
    same = prediction_metrics(sample, sample.fem_solution)
    assert same['absolute_fem_l2'] == 0.
    assert same['absolute_fem_h1_seminorm'] == 0.
    assert same['relative_equation_residual'] < 1e-12
    shifted = prediction_metrics(sample, sample.fem_solution+.25)
    assert shifted['absolute_fem_l2'] == pytest.approx(.25)
    assert shifted['absolute_fem_h1_seminorm'] < 1e-14
    assert shifted['max_dirichlet_error'] == pytest.approx(.25)
    changed = sample.fem_solution.copy()
    changed[sample.inputs['dirichlet_mask']] += .5
    violated = prediction_metrics(sample, changed)
    assert violated['relative_equation_residual'] > .01
    assert violated['max_dirichlet_error'] == pytest.approx(.5)

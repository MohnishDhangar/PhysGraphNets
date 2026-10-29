from dataclasses import asdict
import json
import numpy as np
import pytest
from physgraph.core.types import Mesh2D
from physgraph.mesh.structured import unit_square_mesh
from physgraph.mesh.annulus import annulus_mesh
from physgraph.graph.mesh import build_graph, physical_features
from physgraph.physics.annulus import radial_log_case
from physgraph.physics.diffusion import get_elliptic_case
from physgraph.datasets.instances import PDEInstance, split_instances
from physgraph.datasets.annulus import generate_dataset


@pytest.mark.parametrize('mesh', [unit_square_mesh(3), annulus_mesh(2,16)])
def test_graph_symmetry_areas_and_relabeling(mesh):
    graph=build_graph(mesh)
    arcs={tuple(e):i for i,e in enumerate(graph.edge_index.T)}
    assert len(arcs)==len(graph.length)
    for (a,b),i in arcs.items():
        assert a!=b and (b,a) in arcs
        j=arcs[b,a]
        np.testing.assert_allclose(graph.displacement[i],-graph.displacement[j])
        np.testing.assert_array_equal(graph.outward_normal[i],graph.outward_normal[j])
        assert graph.edge_tags[i]==graph.edge_tags[j]
    vertices=mesh.nodes[mesh.triangles]
    v,w=vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]
    area=.5*np.sum(v[:,0]*w[:,1]-v[:,1]*w[:,0])
    assert graph.node_area.sum()==pytest.approx(area)
    assert np.all(graph.node_area>0)
    perm=np.random.default_rng(3).permutation(len(mesh.nodes))
    inverse=np.argsort(perm)
    renamed=Mesh2D(mesh.nodes[perm],inverse[mesh.triangles],mesh.boundary[perm],
        None if mesh.boundary_edges is None else inverse[mesh.boundary_edges],mesh.boundary_edge_tags)
    other=build_graph(renamed)
    np.testing.assert_allclose(other.node_area,graph.node_area[perm])
    for j,(a,b) in enumerate(other.edge_index.T):
        i=arcs[perm[a],perm[b]]
        np.testing.assert_allclose(other.displacement[j],graph.displacement[i])
        np.testing.assert_allclose(other.outward_normal[j],graph.outward_normal[i])
    assert not graph.nodes.flags.writeable


def test_annulus_inner_normals_and_neumann_sign():
    mesh=annulus_mesh(3,24)
    graph=build_graph(mesh)
    features=physical_features(mesh,graph,radial_log_case(boundary_mode='inner_neumann').problem)
    inner=graph.edge_tags=='inner'
    midpoints=(graph.nodes[graph.edge_index[0]]+graph.nodes[graph.edge_index[1]])/2
    assert np.all(np.sum(midpoints[inner]*graph.outward_normal[inner],axis=1)<0)
    assert np.all(features['neumann_flux'][inner]<0)
    assert np.array_equal(features['neumann_mask'],inner)
    assert np.all(features['dirichlet_values'][~features['dirichlet_mask']]==0)


def test_graph_inputs_do_not_evaluate_dirichlet_data_in_interior():
    from dataclasses import replace
    mesh=unit_square_mesh(4)
    original=get_elliptic_case('affine_diffusion').problem
    def boundary_only(p):
        assert np.all(np.any((p==0)|(p==1),axis=1))
        return original.boundary.dirichlet(p)
    physical=replace(original,boundary=replace(original.boundary,dirichlet=boundary_only))
    features=physical_features(mesh,build_graph(mesh),physical)
    assert np.all(features['dirichlet_values'][~mesh.boundary]==0)
    assert set(features)=={'coefficients','dirichlet_mask','dirichlet_values','neumann_mask','neumann_flux'}


def test_graph_rejects_inverted_triangles():
    mesh=unit_square_mesh(2)
    broken=Mesh2D(mesh.nodes,mesh.triangles[:,::-1],mesh.boundary)
    with pytest.raises(ValueError,match='clockwise'):
        build_graph(broken)


def test_instance_canonical_identity_and_stable_whole_instance_splits():
    a=PDEInstance.create('example',{'radius':1,'center':[0,-0.]},{'k':2})
    b=PDEInstance.create('example',{'center':[0.,0.],'radius':1.},{'k':2.})
    assert a.instance_id==b.instance_id
    instances=[PDEInstance.create('example',{'radius':1},{'k':i+1}) for i in range(100)]
    splits=split_instances(instances,seed=9)
    assert set(splits.values())=={'train','validation','test'}
    assert splits==split_instances(instances[::-1],seed=9)
    assert all(split_instances(instances[:20],seed=9)[i.instance_id]==splits[i.instance_id] for i in instances[:20])
    assert split_instances([a,b],seed=9)==split_instances([a],seed=9)
    with pytest.raises(ValueError,match='finite'):
        PDEInstance.create('bad',{}, {'k':float('nan')})


def test_generated_dataset_uses_fem_labels_and_no_mesh_split_leakage(tmp_path):
    from physgraph.fem.elliptic import solve
    result=generate_dataset(instances=[{'physics':{'boundary_mode':'inner_neumann'}},
        {'physics':{'inner_value':2.,'outer_value':4.}}],resolutions=[2,3],output=tmp_path/'data')
    rows=result['metrics']['samples']
    assert len(rows)==4
    for identifier in result['metrics']['instance_splits']:
        group=[r for r in rows if r['instance_id']==identifier]
        assert len({r['split'] for r in group})==1
    for row in rows:
        with np.load(tmp_path/'data'/row['file'],allow_pickle=False) as archive:
            metadata=json.loads(archive['metadata'].item())
            geometry=metadata['instance']['geometry'];physics=metadata['instance']['physics']
            mesh=annulus_mesh(row['resolution'],8*row['resolution'],**geometry)
            truth=solve(mesh,radial_log_case(**geometry,**physics).problem,4)
            np.testing.assert_array_equal(archive['fem_solution'],truth.values)
            assert 'exact' not in archive and 'pinn_prediction' not in archive
            assert archive['edge_index'].shape[0]==2
    with pytest.raises(ValueError,match='duplicate'):
        generate_dataset(instances=[{},{}],resolutions=[2],output=tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()

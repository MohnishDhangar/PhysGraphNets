from dataclasses import replace
import numpy as np
import pytest
from physgraph.physics.pinn import get_pinn_case, MIXED_CASE_NAMES
from physgraph.physics.mixed_square import neumann_rule
from physgraph.models.pinn import TanhMLP, constrained_field
from physgraph.losses.pinn import loss, loss_gradient
from physgraph.mesh.structured import unit_square_mesh
from physgraph.fem.elliptic import solve
from physgraph.fem.error import field_errors


@pytest.mark.parametrize('name', MIXED_CASE_NAMES)
def test_manufactured_equation_flux_and_fem_convergence(name):
    case = get_pinn_case(name)
    p = np.random.default_rng(3).uniform(.1,.9,(15,2))
    h=1e-5
    div=np.zeros(len(p))
    for d in range(2):
        offset=np.eye(2)[d]*h
        plus=case.problem.diffusion(p+offset)*case.reference.gradient(p+offset)[:,d]
        minus=case.problem.diffusion(p-offset)*case.reference.gradient(p-offset)[:,d]
        div+=(plus-minus)/(2*h)
    np.testing.assert_allclose(-div+case.problem.reaction(p)*case.reference.exact(p), case.problem.forcing(p), atol=2e-8)
    edge,normals,_=neumann_rule(case.boundary_kind)
    expected=case.problem.diffusion(edge)*np.sum(case.reference.gradient(edge)*normals,axis=1)
    np.testing.assert_allclose(expected,case.problem.boundary.neumann(edge,normals),atol=2e-15)
    errors=[]
    for n in (4,8):
        mesh=unit_square_mesh(n)
        result=solve(mesh,case.problem,4)
        errors.append(field_errors(mesh,result.values,case.reference))
    assert errors[1]['relative_l2'] < errors[0]['relative_l2']*.35
    assert errors[1]['relative_h1_seminorm'] < errors[0]['relative_h1_seminorm']*.6


@pytest.mark.parametrize('name', MIXED_CASE_NAMES)
def test_mask_derivatives_and_dirichlet_partition(name):
    case=get_pinn_case(name);model=TanhMLP(3,2)
    params=model.initialize(np.random.default_rng(4))
    p=np.array([[.3,.6],[.8,.2]])
    u,g,lap=constrained_field(model,params,p,case.lifting,case.boundary_kind)
    fd_lap=0
    for d in range(2):
        step=np.eye(2)[d]*1e-4
        plus=constrained_field(model,params,p+step,case.lifting,case.boundary_kind)[0]
        minus=constrained_field(model,params,p-step,case.lifting,case.boundary_kind)[0]
        np.testing.assert_allclose(g[:,d],(plus-minus)/2e-4,atol=2e-7)
        fd_lap+=(plus-2*u+minus)/1e-8
    np.testing.assert_allclose(lap,fd_lap,atol=2e-6)
    t=np.linspace(0,1,51)
    edge=np.vstack([np.column_stack((t*0,t)),np.column_stack((t*0+1,t)),np.column_stack((t,t*0)),np.column_stack((t,t*0+1))])
    edge=edge[case.problem.boundary.dirichlet_selector(edge)]
    values=constrained_field(model,params,edge,case.lifting,case.boundary_kind)[0]
    np.testing.assert_allclose(values,case.problem.boundary.dirichlet(edge),atol=1e-15)


@pytest.mark.parametrize('name', MIXED_CASE_NAMES)
def test_torch_mixed_loss_gradient_parity_and_no_reference_leakage(name):
    torch=pytest.importorskip('torch')
    from physgraph.models.torch_pinn import TorchTanhMLP, field_derivatives
    from physgraph.physics.torch_pinn import sample_physics
    from physgraph.losses.torch_pinn import loss as torch_loss
    case=replace(get_pinn_case(name),reference=None)
    architecture=TanhMLP(3,2);params=architecture.initialize(np.random.default_rng(5))
    points=np.random.default_rng(6).random((24,2))
    model=TorchTanhMLP(3,2,params)
    data=sample_physics(points,case,neumann_points=12,neumann_weight=2.)
    expected=constrained_field(architecture,params,points,case.lifting,case.boundary_kind)
    for a,b in zip(field_derivatives(model,data.points,case.lifting,boundary_kind=case.boundary_kind),expected):
        np.testing.assert_allclose(a.detach().numpy(),b,atol=2e-12)
    objective=torch_loss(model,data);objective.backward()
    options={'neumann_points':12,'neumann_weight':2.}
    assert objective.item()==pytest.approx(loss(architecture,params,points,case,**options),rel=1e-12)
    np.testing.assert_allclose(model.coefficients.grad.numpy(),loss_gradient(architecture,params,points,case,**options),atol=2e-11,rtol=2e-10)


def test_mixed_training_flux_reduction_and_exact_resume(tmp_path):
    pytest.importorskip('torch')
    from physgraph.training.pinn import TrainingConfig
    from physgraph.training.torch_pinn import train
    from physgraph.evaluation.torch_pinn import evaluate_pinn
    cfg=TrainingConfig(case='mixed_sine_right',width=4,depth=1,collocation=48,learning_rate=.01)
    start=train(cfg,0);end=train(cfg,250)
    architecture=TanhMLP(4,1);case=get_pinn_case(cfg.case);mesh=unit_square_mesh(6)
    before=evaluate_pinn(mesh,architecture,start.parameters,case)[1]
    after=evaluate_pinn(mesh,architecture,end.parameters,case)[1]
    assert after['relative_l2'] < before['relative_l2']*.3
    assert after['neumann_flux_rms'] < before['neumann_flux_rms']*.4
    assert after['max_boundary_error'] < 1e-14
    path=tmp_path/'checkpoint.npz'
    train(cfg,100,checkpoint=path)
    resumed=train(cfg,250,resume=path)
    for key in ('parameters','first_moment','second_moment','losses'):
        np.testing.assert_array_equal(getattr(end,key),getattr(resumed,key))

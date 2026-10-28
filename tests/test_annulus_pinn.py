from dataclasses import replace
import json
import numpy as np
import pytest
from physgraph.core.domains import AnnulusDomain
from physgraph.models.constraints import AnnulusConstraint
from physgraph.models.pinn import TanhMLP, constrained_field
from physgraph.physics.pinn import get_pinn_case
from physgraph.physics.base import EllipticProblem
from physgraph.losses.pinn import loss, loss_gradient, strong_residual
from physgraph.training.pinn import TrainingConfig, train, load_checkpoint
from physgraph.evaluation.pinn import evaluate_pinn, evaluate_field


def test_annulus_sampling_area_distribution_and_membership():
    domain = AnnulusDomain(.3, 1.4, (2., -3.))
    points = domain.sample(np.random.default_rng(42), 20000)
    assert domain.contains(points, strict=True)
    assert not domain.contains(np.array([[2., -3.]]))
    assert not domain.contains(np.array([[4., -3.]]))
    delta = points-domain.center
    t = (np.sum(delta**2, axis=1)-.3**2)/(1.4**2-.3**2)
    assert abs(t.mean()-.5) < .01
    assert np.max(np.abs(np.mean(delta/np.linalg.norm(delta, axis=1)[:, None], axis=0))) < .02
    assert domain.contains(domain.boundary_points(100))
    assert not domain.contains(np.empty((0, 2)))


def test_annulus_quadrature_area_and_radial_moment():
    d = AnnulusDomain(.3, 1.4, (2., -3.))
    points, weights = d.quadrature(3)
    assert d.contains(points, strict=True)
    assert weights.sum() == pytest.approx(np.pi*(1.4**2-.3**2), rel=1e-14)
    r2 = np.sum((points-d.center)**2, axis=1)
    assert weights @ r2 == pytest.approx(np.pi/2*(1.4**4-.3**4), rel=1e-14)


@pytest.mark.parametrize('domain', [AnnulusDomain(), AnnulusDomain(.3, 1.4, (2., -3.))])
def test_constraint_derivatives_boundaries_and_nontrivial_lifting(domain):
    constraint = AnnulusConstraint(domain, 1., 2.)
    model = TanhMLP(3, 2)
    parameters = model.initialize(np.random.default_rng(9))
    points = domain.sample(np.random.default_rng(10), 7)
    u, gradient, laplacian = constrained_field(model, parameters, points, (0,0,0), constraint=constraint)
    lap_fd = 0
    for axis in range(2):
        offset = np.eye(2)[axis]*1e-4
        plus = constrained_field(model, parameters, points+offset, (0,0,0), constraint=constraint)[0]
        minus = constrained_field(model, parameters, points-offset, (0,0,0), constraint=constraint)[0]
        np.testing.assert_allclose(gradient[:, axis], (plus-minus)/2e-4, rtol=2e-6, atol=2e-6)
        lap_fd += (plus-2*u+minus)/1e-8
    np.testing.assert_allclose(laplacian, lap_fd, rtol=2e-6, atol=2e-6)
    boundary = domain.boundary_points(80)
    values = constrained_field(model, parameters, boundary, (0,0,0), constraint=constraint)[0]
    np.testing.assert_allclose(values, np.repeat([1.,2.], 80), atol=5e-15)
    # Polynomial lifting is not the exact interior logarithmic solution.
    radius = np.linalg.norm(points-domain.center, axis=1)
    exact = 1+np.log(radius/domain.inner_radius)/np.log(domain.outer_radius/domain.inner_radius)
    assert np.max(np.abs(constraint.values(points)[0]-exact)) > .01


def test_true_circle_and_polygon_evaluation_are_explicitly_separate():
    case = get_pinn_case('annulus_log')
    def field(points):
        return case.reference.exact(points)+1, case.reference.gradient(points)
    _, metrics = evaluate_field(case.domain.mesh(3), case, field, lambda p: np.zeros(len(p)))
    assert metrics['absolute_l2']**2 == pytest.approx(3*np.pi/4)
    assert metrics['polygon_absolute_l2']**2 == pytest.approx(metrics['polygon_domain_area'])
    assert metrics['polygon_domain_area'] < metrics['ideal_domain_area']
    assert metrics['absolute_h1_seminorm'] == 0
    assert metrics['max_boundary_error'] == pytest.approx(1.)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_annulus_resume_replays_state_and_rejects_hole_points(tmp_path, backend):
    trainer, loader = train, load_checkpoint
    if backend == 'torch':
        pytest.importorskip('torch')
        from physgraph.training.torch_pinn import train as trainer, load_checkpoint as loader
    cfg = TrainingConfig(case='annulus_log', width=3, depth=1, collocation=16)
    complete = trainer(cfg, 20)
    path = tmp_path/'checkpoint.npz'
    trainer(cfg, 7, checkpoint=path)
    resumed = trainer(cfg, 20, resume=path)
    for key in ('parameters', 'points', 'first_moment', 'second_moment', 'losses'):
        np.testing.assert_array_equal(getattr(complete, key), getattr(resumed, key))
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    arrays['points'][0] = [0,0]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='malformed'):
        loader(path, cfg)


def test_torch_autograd_annulus_parity_and_reference_isolation():
    pytest.importorskip('torch')
    from physgraph.models.torch_pinn import TorchTanhMLP, field_derivatives
    from physgraph.physics.torch_pinn import sample_physics
    from physgraph.losses.torch_pinn import loss as torch_loss
    case = replace(get_pinn_case('annulus_log'), reference=None)
    architecture = TanhMLP(3,2)
    parameters = architecture.initialize(np.random.default_rng(11))
    points = case.domain.sample(np.random.default_rng(12), 17)
    model = TorchTanhMLP(3,2,parameters)
    data = sample_physics(points,case)
    expected = constrained_field(architecture, parameters, points, case.lifting, constraint=case.constraint)
    actual = field_derivatives(model, data.points, case.lifting, constraint=case.constraint)
    for a,b in zip(actual,expected):
        np.testing.assert_allclose(a.detach().numpy(),b,rtol=1e-12,atol=1e-12)
    objective = torch_loss(model,data)
    objective.backward()
    assert objective.item() == pytest.approx(loss(architecture, parameters, points, case), rel=1e-12)
    np.testing.assert_allclose(model.coefficients.grad.numpy(),loss_gradient(architecture,parameters,points,case),rtol=1e-10,atol=1e-10)


def test_callable_coefficient_gradient_residual():
    case = get_pinn_case('affine_diffusion')
    physical = EllipticProblem('quadratic_k_affine', lambda p: 1+np.sum(p*p,axis=1),
        case.problem.reaction, lambda p: -4*p[:,0]+6*p[:,1]+2*(1+2*p[:,0]-3*p[:,1]), case.problem.boundary)
    case = replace(case, problem=physical, diffusion_gradient=lambda p: 2*p)
    points = np.random.default_rng(13).random((16,2))
    model = TanhMLP(3,1)
    params = np.zeros(model.parameter_count)
    np.testing.assert_allclose(strong_residual(model,params,points,case),0.,atol=2e-15)
    torch = pytest.importorskip('torch')
    from physgraph.models.torch_pinn import TorchTanhMLP
    from physgraph.physics.torch_pinn import sample_physics
    from physgraph.losses.torch_pinn import strong_residual as residual
    np.testing.assert_allclose(residual(TorchTanhMLP(3,1,params), sample_physics(points,case)).detach().numpy(),0.,atol=2e-15)


def test_annulus_training_reduces_independent_error():
    pytest.importorskip('torch')
    from physgraph.training.torch_pinn import train as trainer
    from physgraph.evaluation.torch_pinn import evaluate_pinn as evaluate
    cfg = TrainingConfig(case='annulus_log',width=6,depth=2,collocation=64)
    initial = trainer(cfg,0)
    trained = trainer(cfg,250)
    case = get_pinn_case(cfg.case);model = TanhMLP(6,2);mesh=case.domain.mesh(4)
    before = evaluate(mesh,model,initial.parameters,case)[1]
    after = evaluate(mesh,model,trained.parameters,case)[1]
    assert after['relative_l2'] < before['relative_l2']*.3
    assert after['holdout_residual_rms'] < before['holdout_residual_rms']*.4
    assert after['max_boundary_error'] < 1e-14


def test_annulus_cli_preserves_tagged_fem_and_neural_fields(tmp_path):
    from physgraph.experiments.pinn import run_experiment
    from physgraph.io.results import load_field
    result = run_experiment(case='annulus_log',backend='numpy',width=2,depth=1,collocation=8,steps=1,nx=3,output=tmp_path/'run')
    fem,_=load_field(tmp_path/'run/fem_reference.npz')
    with np.load(tmp_path/'run/comparison.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(data['fem_solution'],fem.solution)
        assert set(data['boundary_edge_tags']) == {'inner','outer'}
        assert not np.array_equal(data['fem_solution'],data['pinn_prediction'])
    assert result['config']['domain']['name']=='annulus'
    assert 'polygon_relative_l2' in result['metrics']['pinn']

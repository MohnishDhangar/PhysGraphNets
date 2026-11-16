"""DFG parameters, conormal outlet, cylinder topology, loads and comparison scope."""
from dataclasses import replace
from importlib.util import find_spec
import numpy as np
import pytest
from physgraph.physics.cylinder_flow import CylinderFlowCase
from physgraph.mesh.cylinder_flow import cylinder_flow_mesh
from physgraph.mesh.quality import mesh_quality
from physgraph.fem.navier_stokes import solve_navier_stokes
from physgraph.evaluation.cylinder_flow import cylinder_history, pressure_probe_weights, published_comparison


def solve(case=None, level=2, quadrature_order=4):
    case=CylinderFlowCase() if case is None else case
    return solve_navier_stokes(cylinder_flow_mesh(case,level),case.zero,case.boundary_velocity,case.zero,
        viscosity=case.viscosity,dt=.04,steps=3,dirichlet_tags=case.dirichlet_tags,
        tractions=case.tractions,viscous_form='laplacian',quadrature_order=quadrature_order)


def test_published_initial_inlet_and_fixed_normalization():
    case=CylinderFlowCase()
    points=np.array([[0,0],[0,.205],[0,.41]])
    assert np.max(np.abs(case.inlet(points,0)))==0
    assert case.inlet(points,4)[:,0]==pytest.approx([0,1.5,0])
    assert case.force_scale==pytest.approx(.05)
    assert case.matches_published_case
    assert not replace(case,length=2.8).matches_published_case


@pytest.mark.parametrize('level',[2,4,8])
def test_dedicated_mesh_domain_orientation_hole_and_probes(level):
    case=CylinderFlowCase()
    mesh=cylinder_flow_mesh(case,level)
    quality=mesh_quality(mesh)
    assert quality['holes']==1
    assert quality['min_area']>0
    assert set(mesh.boundary_edge_tags)=={'left','right','top','bottom','cylinder'}
    expected=case.length*case.height-.5*(16*level)*case.radius**2*np.sin(2*np.pi/(16*level))
    assert quality['area']==pytest.approx(expected,abs=1e-12)
    # Affine pressure probe interpolation must be exact, including circle tips.
    points=np.array([[.15,.2],[.25,.2]])
    pressure=1+2*mesh.nodes[:,0]-3*mesh.nodes[:,1]
    values=[w @ pressure[tri] for tri,w in pressure_probe_weights(mesh,points)]
    assert values==pytest.approx(1+2*points[:,0]-3*points[:,1],abs=1e-12)
    with pytest.raises(ValueError,match='outside'):
        pressure_probe_weights(mesh,np.array([[.2,.2]]))


def test_cylinder_no_slip_flux_pressure_policy_and_exact_quadrature():
    case=CylinderFlowCase()
    result=solve(case)
    fine_quadrature=solve(case,quadrature_order=6)
    assert np.max(np.abs(result.velocity-fine_quadrature.velocity))<1e-10
    assert np.max(np.abs(result.pressure-fine_quadrature.pressure))<1e-9
    history=cylinder_history(result,case)
    for row in history['rows']:
        assert row['max_obstacle_speed']==0
        assert row['pressure_constraint']=='traction'
        assert row['load_kind']=='laplacian_conormal'
        assert abs(row['net_flux'])<1e-11
        assert row['inlet_flux']==pytest.approx(-2*case.height*case.peak_speed*np.sin(np.pi*row['time']/8)/3,abs=1e-12)
        assert row['drag']>0
    assert published_comparison(history,case,.12)['status']=='not_assessed'


def test_benchmark_cannot_pass_short_or_modified_case():
    reference=published_comparison({},CylinderFlowCase(),.1)['reference']
    assert published_comparison(reference,CylinderFlowCase(),8)['status']=='passed'
    assert published_comparison(reference,CylinderFlowCase(),1)['status']=='not_assessed'
    assert published_comparison(reference,replace(CylinderFlowCase(),length=3.),8)['status']=='not_assessed'
    bad={**reference,'max_lift':.1}
    assert published_comparison(bad,CylinderFlowCase(),8)['status']=='failed_acceptance'


def test_laplacian_mixed_manufactured_transient_patch():
    from physgraph.physics.navier_stokes import NavierStokesCase
    from physgraph.mesh.structured import tagged_unit_square_mesh
    from physgraph.evaluation.stokes import stokes_errors
    case=NavierStokesCase('accelerating_shear',time_profile='linear',pressure_offset=2.5)
    def traction(x,n,t):
        _,grad,p,_=case.fields(x,t)
        return case.viscosity*np.einsum('qij,qj->qi',grad,n)-p[:,None]*n
    result=solve_navier_stokes(tagged_unit_square_mesh(3),case.forcing,case.velocity,
        lambda x:case.velocity(x,0),dt=.05,steps=3,dirichlet_tags=['left','top','bottom'],
        tractions={'right':traction},viscous_form='laplacian')
    errors=stokes_errors(result.frame(3),case.at(.15))
    assert max(errors[k] for k in ('velocity_l2','velocity_h1_seminorm','pressure_l2'))<1e-10


@pytest.mark.skipif(find_spec('dolfinx') is None,reason='optional FEniCSx environment')
def test_independent_cylinder_projection_fields_and_obstacle_reactions():
    from physgraph.verification.navier_stokes import compare_navier_stokes
    assert compare_navier_stokes(solve(),CylinderFlowCase())['passed']

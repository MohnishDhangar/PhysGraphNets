"""Independent UFL assembly and PETSc solution on the identical P1 mesh.

Targets DOLFINx 0.10/0.11 with real double PETSc, one process. No PhysGraph
assembly, gradient, quadrature or elimination code is used by the external solve.
"""
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as sparse_norm

from physgraph.fem.elliptic import solve
from physgraph.fem.error import field_errors
from physgraph.verification.cases import ComparisonCase


class FEniCSxUnavailable(RuntimeError):
    pass


def require_fenicsx() -> None:
    missing = [name for name in ('dolfinx', 'basix', 'ufl', 'mpi4py', 'petsc4py')
               if find_spec(name) is None]
    if missing:
        raise FEniCSxUnavailable('Missing '+', '.join(missing)+'. See docs/fenicsx.md for the isolated environment.')


def coordinate_permutation(original: np.ndarray, reordered: np.ndarray) -> np.ndarray:
    """Return original node indices in external order; require a true bijection."""
    original, reordered = np.asarray(original), np.asarray(reordered)
    if (original.ndim != 2 or original.shape[1] != 2 or reordered.shape != original.shape
            or len(original) == 0 or not np.isfinite(original).all() or not np.isfinite(reordered).all()):
        raise ValueError('coordinate arrays must have the same finite (N,2) shape')
    distance, indices = cKDTree(original).query(reordered)
    tolerance = 1e-12*max(1., float(np.max(np.abs(original))))
    if distance.max() > tolerance or len(np.unique(indices)) != len(original):
        raise ValueError('external P1 coordinates are not a bijective match to the input mesh')
    return indices


def _expressions(name, domain, ufl):
    """Independently derive forcing and flux symbolically from the stated field."""
    x, y = ufl.SpatialCoordinate(domain)
    k, c = 1., 1.
    if name == 'variable_sine':
        k, c = 1+x+y, 1+x
        exact = 1+x+2*y+ufl.sin(np.pi*x)*ufl.sin(np.pi*y)
    elif name == 'affine_diffusion':
        k, c = 1+x+y, 2.
        exact = 1+2*x-3*y
    elif name == 'mixed_polynomial':
        k = 1+x+y
        exact = 1+x*x+y*y
    elif name in ('mixed_sine_right', 'mixed_sine_left'):
        k = 1+x+y
        z = x if name == 'mixed_sine_right' else 1-x
        exact = 1+x+2*y+ufl.sin(np.pi*z/2)*ufl.sin(np.pi*y)
    elif name in ('parametric_square_dirichlet', 'parametric_square_right_neumann'):
        # Deliberately spell out the manufactured fixture independently of NumPy.
        k, c = 1+.4*x+.2*y, .5
        exact = 1+.5*x-.25*y+(.7*ufl.sin(np.pi*x)-.3*ufl.sin(2*np.pi*x))*ufl.sin(np.pi*y)
    elif name.startswith('annulus_'):
        c = 0.
        exact = 1+ufl.ln(ufl.sqrt(x*x+y*y)/.5)/np.log(2.)
    else:
        exact = 1+x*x+y*y
    forcing = -ufl.div(k*ufl.grad(exact))+c*exact
    flux = k*ufl.dot(ufl.grad(exact), ufl.FacetNormal(domain))
    return exact, k, c, forcing, flux


def compare(case: ComparisonCase, *, quadrature_degree: int = 12,
            assembly_order: int = 7, tolerance: float = 1e-8,
            cache_dir: str | Path = '/tmp/physgraph-fenicsx-cache') -> tuple[dict, dict]:
    require_fenicsx()
    from mpi4py import MPI  # Initialize MPI before importing DOLFINx.
    from petsc4py import PETSc
    import basix.ufl
    import dolfinx
    import ufl
    from dolfinx import fem, mesh as dxmesh
    from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector
    if MPI.COMM_WORLD.size != 1:
        raise ValueError('comparison runner supports one MPI process only')
    if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
        raise ValueError('comparison requires real float64 PETSc')
    if not dolfinx.__version__.startswith(('0.10.', '0.11.')):
        raise ValueError('adapter targets DOLFINx 0.10/0.11; this version has not been validated')
    if quadrature_degree < 4 or assembly_order < 3 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('positive tolerance and sufficiently high quadrature are required')
    source = case.mesh
    element = basix.ufl.element('Lagrange', 'triangle', 1, shape=(2,))
    domain = dxmesh.create_mesh(MPI.COMM_SELF, source.triangles, ufl.Mesh(element), source.nodes)
    V = fem.functionspace(domain, ('Lagrange', 1))
    permutation = coordinate_permutation(source.nodes, V.tabulate_dof_coordinates()[:, :2])
    inverse = np.argsort(permutation)
    geometry_map = coordinate_permutation(source.nodes, domain.geometry.x[:, :2])
    domain.topology.create_connectivity(1, 2)
    facets = dxmesh.exterior_facet_indices(domain.topology)
    facet_geometry = dxmesh.entities_to_geometry(domain, 1, facets, False)
    original_edges = geometry_map[facet_geometry]
    if case.dirichlet_tags is None:
        midpoints = source.nodes[original_edges].mean(axis=1)
        if case.name in ('mixed_polynomial', 'mixed_sine_right', 'mixed_sine_left', 'parametric_square_right_neumann'):
            neumann_x = 0. if case.name == 'mixed_sine_left' else 1.
            selected = ~np.isclose(midpoints[:, 0], neumann_x)
        else:
            selected = np.ones(len(facets), bool)
    else:
        tags = {tuple(sorted(edge)): tag for edge, tag in zip(source.boundary_edges, source.boundary_edge_tags)}
        selected = np.array([tags[tuple(sorted(edge))] in case.dirichlet_tags for edge in original_edges])
    # One tag for Dirichlet, another for Neumann; sorted entity indices required.
    order = np.argsort(facets)
    markers = dxmesh.meshtags(domain, 1, facets[order], np.where(selected[order], 1, 2).astype(np.int32))
    metadata = {'quadrature_degree': quadrature_degree}
    dx = ufl.Measure('dx', domain=domain, metadata=metadata)
    ds = ufl.Measure('ds', domain=domain, subdomain_data=markers, metadata=metadata)
    exact, k, c, forcing, flux = _expressions(case.name, domain, ufl)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (k*ufl.inner(ufl.grad(u), ufl.grad(v))+c*u*v)*dx
    mass = u*v*dx
    load = forcing*v*dx+flux*v*ds(2)
    jit = {'cache_dir': str(Path(cache_dir).resolve())}
    g = fem.Function(V)
    g.interpolate(fem.Expression(exact, V.element.interpolation_points, jit_options=jit))
    dofs = fem.locate_dofs_topological(V, 1, facets[selected])
    bc = fem.dirichletbc(g, dofs)
    problem = LinearProblem(a, load, bcs=[bc], petsc_options_prefix='physgraph_compare_',
        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu', 'ksp_error_if_not_converged': True}, jit_options=jit)
    uh = problem.solve()
    uh.x.scatter_forward()
    values = uh.x.array[inverse].copy()

    def matrix(form):
        assembled = assemble_matrix(fem.form(form, jit_options=jit))
        try:
            assembled.assemble()
            indptr, indices, data = assembled.getValuesCSR()
            copied = csr_matrix((data.copy(), indices.copy(), indptr.copy()), shape=assembled.getSize())
            return copied[inverse, :][:, inverse]
        finally:
            assembled.destroy()
    A, M = matrix(a), matrix(mass)
    vector = assemble_vector(fem.form(load, jit_options=jit))
    try:
        vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        F = vector.array[inverse].copy()
    finally:
        vector.destroy()
    reference = solve(source, case.manufactured.problem, assembly_order)
    if not np.array_equal(np.sort(permutation[dofs]), reference.reduced.boundary):
        raise ValueError('independent Dirichlet DOF sets disagree')
    analytical = case.manufactured.reference
    difference = values-reference.values
    mass_error = float(np.sqrt(max(float(difference @ (reference.assembly.mass @ difference)), 0.)))
    metrics = {
        'case': case.name, 'nodes': len(source.nodes), 'triangles': len(source.triangles),
        'dolfinx_version': dolfinx.__version__, 'ufl_version': ufl.__version__,
        'basix_version': basix.__version__, 'petsc_version': list(PETSc.Sys.getVersion()),
        'max_nodal_difference': float(np.max(np.abs(difference))),
        'relative_mass_difference': mass_error/max(float(np.sqrt(reference.values @ (reference.assembly.mass @ reference.values))), 1e-12),
        'relative_matrix_difference': float(sparse_norm(A-reference.assembly.stiffness)/max(sparse_norm(reference.assembly.stiffness), 1e-12)),
        'relative_mass_matrix_difference': float(sparse_norm(M-reference.assembly.mass)/max(sparse_norm(reference.assembly.mass), 1e-12)),
        'relative_load_difference': float(np.linalg.norm(F-reference.assembly.load)/max(np.linalg.norm(reference.assembly.load), 1.)),
        'physgraph_errors': field_errors(source, reference.values, analytical, 7),
        'fenicsx_nodal_field_errors': field_errors(source, values, analytical, 7),
    }
    delta = uh-exact
    metrics['fenicsx_independent_absolute_l2'] = float(np.sqrt(max(fem.assemble_scalar(fem.form(delta*delta*dx, jit_options=jit)), 0.)))
    metrics['fenicsx_independent_absolute_h1'] = float(np.sqrt(max(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(delta), ufl.grad(delta))*dx, jit_options=jit)), 0.)))
    boundary = reference.reduced.boundary
    metrics['max_boundary_error'] = float(np.max(np.abs(values[boundary]-analytical.exact(source.nodes[boundary]))))
    metrics['passed'] = bool(all(metrics[key] <= tolerance for key in (
        'max_nodal_difference', 'relative_mass_difference', 'relative_matrix_difference',
        'relative_mass_matrix_difference', 'relative_load_difference', 'max_boundary_error')))
    arrays = {'nodes': source.nodes, 'triangles': source.triangles, 'physgraph_fem': reference.values,
              'fenicsx_fem': values, 'exact': analytical.exact(source.nodes)}
    return metrics, arrays

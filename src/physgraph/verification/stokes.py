"""Independent UFL/PETSc Stokes solve, identical mesh, different pressure gauge."""
import numpy as np
from physgraph.verification.fenicsx import require_fenicsx, coordinate_permutation
from physgraph.physics.obstacle_stokes import ObstacleStokesCase


def compare_stokes(result, case, tolerance=1e-9):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('comparison tolerance must be positive and finite')
    if case.viscosity != result.viscosity:
        raise ValueError('comparison viscosity mismatch')
    require_fenicsx()
    from mpi4py import MPI
    from petsc4py import PETSc
    import basix.ufl
    import dolfinx
    import ufl
    from dolfinx import fem, mesh as dxmesh
    from dolfinx.fem.petsc import LinearProblem
    if MPI.COMM_WORLD.size != 1 or np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
        raise ValueError('Stokes comparison requires serial real float64 PETSc')
    source = result.space.mesh
    domain = dxmesh.create_mesh(MPI.COMM_SELF, source.triangles,
        ufl.Mesh(basix.ufl.element('Lagrange', 'triangle', 1, shape=(2,))), source.nodes)
    p2 = basix.ufl.element('Lagrange', 'triangle', 2)
    p1 = basix.ufl.element('Lagrange', 'triangle', 1)
    space = fem.functionspace(domain, basix.ufl.mixed_element([p2, p2, p1]))
    collapsed = [space.sub(i).collapse() for i in range(3)]
    x, y = ufl.SpatialCoordinate(domain)
    obstacle = isinstance(case, ObstacleStokesCase)
    if obstacle:
        exact, pressure = ufl.as_vector((0., 0.)), ufl.as_ufl(0.)
    elif case.name == 'affine':
        exact, pressure = ufl.as_vector((1+x, -.5-y)), x+y-1
    elif case.name == 'channel':
        exact, pressure = ufl.as_vector((y*(1-y), 0)), case.viscosity*(1-2*x)
    else:
        psi = x*x*(1-x)**2*y*y*(1-y)**2
        exact = ufl.as_vector((psi.dx(1), -psi.dx(0)))
        if case.name == 'translated_vortex':
            exact += ufl.as_vector((1., -.5))
        pressure = x*x+y*y-2/3
    if not obstacle:
        pressure += case.pressure_offset
    def stress(velocity, p):
        gradient = ufl.grad(velocity)
        if result.viscous_form == 'symmetric_gradient':
            gradient = gradient+ufl.transpose(gradient)
        return case.viscosity*gradient-p*ufl.Identity(2)
    force = (fem.Constant(domain, np.zeros(2, dtype=PETSc.ScalarType)) if obstacle
             else -ufl.div(stress(exact, pressure)))
    ux, uy, p = ufl.TrialFunctions(space)
    vx, vy, q = ufl.TestFunctions(space)
    u, v = ufl.as_vector((ux, uy)), ufl.as_vector((vx, vy))
    dx = ufl.Measure('dx', domain=domain, metadata={'quadrature_degree': 12})
    a = (ufl.inner(stress(u, p), ufl.grad(v))-q*ufl.div(u))*dx
    load = ufl.inner(force, v)*dx
    domain.topology.create_connectivity(1, 2)
    facets = dxmesh.exterior_facet_indices(domain.topology)
    geometry_map = coordinate_permutation(source.nodes, domain.geometry.x[:, :2])
    facet_vertices = geometry_map[dxmesh.entities_to_geometry(domain, 1, facets, False)]
    tag_lookup = ({} if source.boundary_edges is None else
                  {tuple(sorted(edge)): str(tag) for edge, tag in zip(source.boundary_edges, source.boundary_edge_tags)})
    names = [tag_lookup.get(tuple(sorted(edge)), 'boundary') for edge in facet_vertices]
    selected = np.array([result.dirichlet_tags is None or tag in result.dirichlet_tags for tag in names])
    tag_ids = {name: i+1 for i, name in enumerate(sorted(set(names)))}
    order = np.argsort(facets)
    markers = dxmesh.meshtags(domain, 1, facets[order], np.array([tag_ids[names[i]] for i in order], dtype=np.int32))
    ds = ufl.Measure('ds', domain=domain, subdomain_data=markers, metadata={'quadrature_degree': 12})
    normal = ufl.FacetNormal(domain)
    for tag in tag_ids:
        if result.dirichlet_tags is not None and tag not in result.dirichlet_tags:
            if not obstacle:  # obstacle outlet has explicitly zero physical traction
                load += ufl.inner(ufl.dot(stress(exact, pressure), normal), v)*ds(tag_ids[tag])
    jit = {'cache_dir': '/tmp/physgraph-stokes-fenicsx-cache'}
    bcs = []
    for i in range(2):
        vi, _ = collapsed[i]
        boundary_groups = [None] if not obstacle else sorted({tag for tag, chosen in zip(names, selected) if chosen})
        for tag in boundary_groups:
            expression = exact[i]
            active = selected.copy()
            if obstacle:
                active &= np.array([name == tag for name in names])
                if i == 0 and tag in ('outer', 'farfield'):
                    expression = ufl.as_ufl(case.speed)
                elif i == 0 and tag == 'left':
                    expression = case.speed*(1-y*y/(case.half_height**2))
                else:
                    expression = ufl.as_ufl(0.)
            g = fem.Function(vi)
            g.interpolate(fem.Expression(expression, vi.element.interpolation_points,
                                         comm=domain.comm, jit_options=jit))
            dofs = fem.locate_dofs_topological((space.sub(i), vi), 1, facets[active])
            bcs.append(fem.dirichletbc(g, dofs, space.sub(i)))
    # Pin an external pressure DOF; remove its mean after solving. This checks the
    # NumPy mean constraint independently instead of duplicating its gauge row.
    mean_gauge = result.diagnostics['pressure_constraint'] == 'mean_zero'
    if mean_gauge:
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0),
            np.array([collapsed[2][1][0]], dtype=np.int32), space.sub(2)))
    if not PETSc.Sys.hasExternalPackage('mumps'):
        raise RuntimeError('independent Stokes comparison requires PETSc MUMPS for pivoted saddle-point LU')
    problem = LinearProblem(a, load, bcs=bcs, petsc_options_prefix='physgraph_stokes_',
        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu',
                       'pc_factor_mat_solver_type': 'mumps', 'ksp_error_if_not_converged': True},
        jit_options=jit)
    solution = problem.solve()
    solution.x.scatter_forward()
    fields = [solution.sub(i).collapse() for i in range(3)]
    volume = fem.assemble_scalar(fem.form(fem.Constant(domain, PETSc.ScalarType(1))*dx, jit_options=jit))
    mean = fem.assemble_scalar(fem.form(fields[2]*dx, jit_options=jit))/volume
    if mean_gauge:
        fields[2].x.array[:] -= mean
    reordered = []
    for i, field in enumerate(fields):
        points = result.space.points if i < 2 else source.nodes
        perm = coordinate_permutation(points, field.function_space.tabulate_dof_coordinates()[:, :2])
        reordered.append(field.x.array[np.argsort(perm)].copy())
    uv = np.column_stack(reordered[:2])
    velocity_difference = float(np.max(np.abs(uv-result.velocity)))
    pressure_difference = float(np.max(np.abs(reordered[2]-result.pressure)))
    from physgraph.evaluation.stokes import stokes_boundary_integrals
    local_integrals = stokes_boundary_integrals(result)['by_tag']
    external_velocity = ufl.as_vector((fields[0], fields[1]))
    traction = ufl.dot(stress(external_velocity, fields[2]), normal)
    flux_difference, load_difference = 0., 0.
    for tag, marker in tag_ids.items():
        flux = fem.assemble_scalar(fem.form(ufl.dot(external_velocity, normal)*ds(marker), jit_options=jit))
        load_vector = np.array([fem.assemble_scalar(fem.form(traction[i]*ds(marker), jit_options=jit)) for i in range(2)])
        flux_difference = max(flux_difference, abs(float(flux)-local_integrals[tag]['flux']))
        load_difference = max(load_difference, float(np.max(np.abs(load_vector-local_integrals[tag]['force_on_fluid']))))
    reaction_difference = 0.
    if obstacle:
        from physgraph.evaluation.obstacle_stokes import obstacle_dofs
        original_body = obstacle_dofs(result, case.obstacle_tags)
        own_force = -result.velocity_reactions[original_body].sum(axis=0)
        external_force = []
        body_facets = facets[np.array([tag in case.obstacle_tags for tag in names])]
        for i in range(2):
            test = fem.Function(space)
            vi, mapping = collapsed[i]
            body_dofs = fem.locate_dofs_topological(vi, 1, body_facets)
            test.x.array[np.asarray(mapping)[body_dofs]] = 1.
            functional = ufl.action(ufl.action(a, solution), test)-ufl.action(load, test)
            external_force.append(-fem.assemble_scalar(fem.form(functional, jit_options=jit)))
        reaction_difference = float(np.max(np.abs(np.asarray(external_force)-own_force)))
    return {'dolfinx_version': dolfinx.__version__, 'petsc_version': list(PETSc.Sys.getVersion()),
            'max_velocity_difference': velocity_difference, 'max_pressure_difference': pressure_difference,
            'max_boundary_flux_difference': flux_difference, 'max_boundary_load_difference': load_difference,
            'max_obstacle_reaction_difference': reaction_difference,
            'tolerance': tolerance, 'passed': bool(max(velocity_difference, pressure_difference,
                                                       flux_difference, load_difference, reaction_difference) <= tolerance)}

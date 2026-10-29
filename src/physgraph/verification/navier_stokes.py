"""Independent UFL/SNES trajectory, including its own constrained projection."""
import numpy as np
from physgraph.verification.fenicsx import require_fenicsx, coordinate_permutation
from physgraph.evaluation.stokes import stokes_boundary_integrals


def compare_navier_stokes(trajectory, case, tolerance=1e-9):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('comparison tolerance must be positive and finite')
    if case.viscosity != trajectory.viscosity:
        raise ValueError('comparison viscosity mismatch')
    require_fenicsx()
    from mpi4py import MPI
    from petsc4py import PETSc
    import basix.ufl
    import dolfinx
    import ufl
    from dolfinx import fem, mesh as dxmesh
    from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
    if MPI.COMM_WORLD.size != 1 or np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
        raise ValueError('transient comparison requires serial real float64 PETSc')
    if not PETSc.Sys.hasExternalPackage('mumps'):
        raise RuntimeError('transient comparison requires PETSc MUMPS')
    source = trajectory.space.mesh
    domain = dxmesh.create_mesh(MPI.COMM_SELF, source.triangles,
        ufl.Mesh(basix.ufl.element('Lagrange', 'triangle', 1, shape=(2,))), source.nodes)
    p2 = basix.ufl.element('Lagrange', 'triangle', 2)
    p1 = basix.ufl.element('Lagrange', 'triangle', 1)
    space = fem.functionspace(domain, basix.ufl.mixed_element([p2, p2, p1]))
    collapsed = [space.sub(i).collapse() for i in range(3)]
    time = fem.Constant(domain, PETSc.ScalarType(trajectory.times[0]))
    symbolic_time = ufl.variable(time)
    amplitude = ufl.exp(symbolic_time) if case.time_profile == 'exponential' else 1+symbolic_time
    x, y = ufl.SpatialCoordinate(domain)
    if case.name == 'accelerating_shear':
        base, base_p = ufl.as_vector((1+y*(1-y), .5)), x+y-1
    else:
        psi = 32*x*x*(1-x)**2*y*y*(1-y)**2
        base = ufl.as_vector((psi.dx(1), -psi.dx(0)))
        if case.name == 'translated_vortex':
            base += ufl.as_vector((1., -.5))
        base_p = x*x+y*y-2/3
    exact, exact_p = amplitude*base, amplitude*base_p+case.pressure_offset

    def stress(u, p):
        return case.viscosity*(ufl.grad(u)+ufl.transpose(ufl.grad(u)))-p*ufl.Identity(2)

    forcing = ufl.diff(exact, symbolic_time)+ufl.dot(ufl.grad(exact), exact)-ufl.div(stress(exact, exact_p))
    dx = ufl.Measure('dx', domain=domain, metadata={'quadrature_degree': 16})
    domain.topology.create_connectivity(1, 2)
    facets = dxmesh.exterior_facet_indices(domain.topology)
    geometry_map = coordinate_permutation(source.nodes, domain.geometry.x[:, :2])
    vertices = geometry_map[dxmesh.entities_to_geometry(domain, 1, facets, False)]
    lookup = ({} if source.boundary_edges is None else
              {tuple(sorted(e)): str(tag) for e, tag in zip(source.boundary_edges, source.boundary_edge_tags)})
    names = [lookup.get(tuple(sorted(e)), 'boundary') for e in vertices]
    tags = {tag: i+1 for i, tag in enumerate(sorted(set(names)))}
    order = np.argsort(facets)
    markers = dxmesh.meshtags(domain, 1, facets[order], np.array([tags[names[i]] for i in order], dtype=np.int32))
    ds = ufl.Measure('ds', domain=domain, subdomain_data=markers, metadata={'quadrature_degree': 16})
    normal = ufl.FacetNormal(domain)
    selected = np.array([trajectory.dirichlet_tags is None or tag in trajectory.dirichlet_tags for tag in names])
    mean_gauge = bool(np.all(selected))
    jit = {'cache_dir': '/tmp/physgraph-transient-fenicsx-cache'}
    bcs, boundary_values = [], []
    for i in range(2):
        vi = collapsed[i][0]
        g = fem.Function(vi)
        expression = fem.Expression(exact[i], vi.element.interpolation_points, jit_options=jit)
        g.interpolate(expression)
        boundary_values.append((g, expression))
        dofs = fem.locate_dofs_topological((space.sub(i), vi), 1, facets[selected])
        bcs.append(fem.dirichletbc(g, dofs, space.sub(i)))
    if mean_gauge:
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0), np.array([collapsed[2][1][0]], dtype=np.int32), space.sub(2)))
    vx, vy, q = ufl.TestFunctions(space)
    v = ufl.as_vector((vx, vy))
    ux, uy, p = ufl.TrialFunctions(space)
    trial = ufl.as_vector((ux, uy))
    lu_options = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps',
                  'ksp_error_if_not_converged': True}
    projection = LinearProblem((ufl.inner(trial, v)-p*ufl.div(v)-q*ufl.div(trial))*dx,
        ufl.inner(exact, v)*dx, bcs=bcs, petsc_options_prefix='physgraph_ns_projection_',
        petsc_options=lu_options, jit_options=jit)
    initial = projection.solve()
    w, previous, older = fem.Function(space), fem.Function(space), fem.Function(space)
    previous.x.array[:] = initial.x.array
    older.x.array[:] = initial.x.array
    w.x.array[:] = initial.x.array
    w.x.array[np.asarray(collapsed[2][1])] = 0.
    ux, uy, p = ufl.split(w)
    u = ufl.as_vector((ux, uy))
    last = ufl.as_vector(ufl.split(previous)[:2])
    before = ufl.as_vector(ufl.split(older)[:2])
    dt = float(trajectory.times[1]-trajectory.times[0])
    if not np.allclose(np.diff(trajectory.times), dt, rtol=1e-12, atol=1e-14):
        raise ValueError('external trajectory comparison requires uniform steps')
    coefficients = fem.Constant(domain, np.array([1., -1., 0.], dtype=PETSc.ScalarType))
    rate = (coefficients[0]*u+coefficients[1]*last+coefficients[2]*before)/dt
    residual = (ufl.inner(rate+ufl.dot(ufl.grad(u), u)-forcing, v)
                +ufl.inner(stress(u, p), ufl.grad(v))-q*ufl.div(u))*dx
    for tag in tags:
        if trajectory.dirichlet_tags is not None and tag not in trajectory.dirichlet_tags:
            residual -= ufl.inner(ufl.dot(stress(exact, exact_p), normal), v)*ds(tags[tag])
    problem = NonlinearProblem(residual, w, bcs=bcs, petsc_options_prefix='physgraph_ns_',
        petsc_options={**lu_options, 'snes_type': 'newtonls', 'snes_linesearch_type': 'bt',
                       'snes_rtol': 1e-13, 'snes_atol': 1e-13, 'snes_max_it': 30,
                       'snes_error_if_not_converged': True}, jit_options=jit)
    permutations = []
    for i in range(3):
        target = trajectory.space.points if i < 2 else source.nodes
        permutations.append(np.argsort(coordinate_permutation(target, collapsed[i][0].tabulate_dof_coordinates()[:, :2])))

    def values(function):
        fields = [function.sub(i).collapse().x.array[permutations[i]].copy() for i in range(3)]
        return np.column_stack(fields[:2]), fields[2]

    errors = dict(max_velocity_difference=float(np.max(np.abs(values(initial)[0]-trajectory.velocity[0]))),
                  max_pressure_difference=0., max_boundary_flux_difference=0., max_boundary_load_difference=0.,
                  max_reaction_difference=0.)
    volume = fem.assemble_scalar(fem.form(fem.Constant(domain, PETSc.ScalarType(1))*dx, jit_options=jit))
    pressure_mean_form = fem.form(p*dx, jit_options=jit)
    shift = fem.Constant(domain, PETSc.ScalarType(0))
    flux_forms = {tag: fem.form(ufl.dot(u, normal)*ds(marker), jit_options=jit) for tag, marker in tags.items()}
    load_forms = {tag: [fem.form(ufl.dot(stress(u, p-shift), normal)[i]*ds(marker), jit_options=jit)
                       for i in range(2)] for tag, marker in tags.items()}
    reaction_forms = []
    for i in range(2):
        test = fem.Function(space)
        vi, mapping = collapsed[i]
        dofs = fem.locate_dofs_topological(vi, 1, facets[selected])
        test.x.array[np.asarray(mapping)[dofs]] = 1.
        reaction_forms.append(fem.form(ufl.action(residual, test), jit_options=jit))
    iterations = []
    for step, value in enumerate(trajectory.times[1:], 1):
        time.value = value
        if trajectory.scheme == 'bdf2' and step > 1:
            coefficients.value[:] = [1.5, -2., .5]
        for g, expression in boundary_values:
            g.interpolate(expression)
        problem.solve()
        if problem.solver.getConvergedReason() <= 0:
            raise RuntimeError('independent SNES failed to converge')
        iterations.append(problem.solver.getIterationNumber())
        w.x.scatter_forward()
        velocity, pressure = values(w)
        shift.value = fem.assemble_scalar(pressure_mean_form)/volume if mean_gauge else 0.
        pressure -= float(shift.value)
        errors['max_velocity_difference'] = max(errors['max_velocity_difference'], float(np.max(np.abs(velocity-trajectory.velocity[step]))))
        errors['max_pressure_difference'] = max(errors['max_pressure_difference'], float(np.max(np.abs(pressure-trajectory.pressure[step-1]))))
        integrals = stokes_boundary_integrals(trajectory.frame(step))['by_tag']
        for tag in tags:
            flux = fem.assemble_scalar(flux_forms[tag])
            load = np.array([fem.assemble_scalar(form) for form in load_forms[tag]])
            errors['max_boundary_flux_difference'] = max(errors['max_boundary_flux_difference'], abs(float(flux)-integrals[tag]['flux']))
            errors['max_boundary_load_difference'] = max(errors['max_boundary_load_difference'], float(np.max(np.abs(load-integrals[tag]['force_on_fluid']))))
        reaction = np.array([fem.assemble_scalar(form) for form in reaction_forms])
        errors['max_reaction_difference'] = max(errors['max_reaction_difference'], float(np.max(np.abs(reaction-trajectory.diagnostics[step-1]['dirichlet_reaction_on_fluid']))))
        older.x.array[:] = previous.x.array
        previous.x.array[:] = w.x.array
    return {**errors, 'passed': bool(max(errors.values()) <= tolerance), 'tolerance': tolerance,
            'dolfinx_version': dolfinx.__version__, 'petsc_version': list(PETSc.Sys.getVersion()),
            'steps_compared': len(trajectory.times)-1, 'initial_projection_compared': True,
            'snes_iterations': iterations}

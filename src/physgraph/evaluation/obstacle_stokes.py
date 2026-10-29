"""No-slip and independent surface/reaction force diagnostics for obstacles."""
import numpy as np
from physgraph.fem.stokes_boundary import stokes_facets
from physgraph.evaluation.stokes import stokes_boundary_integrals


def obstacle_dofs(result, tags):
    facets = stokes_facets(result.space.mesh)
    selected = set(tags)
    if not selected or not selected <= {facet.tag for facet in facets}:
        raise ValueError('unknown or empty obstacle tags')
    if result.dirichlet_tags is not None and not selected <= set(result.dirichlet_tags):
        raise ValueError('obstacle force requires a prescribed-velocity obstacle')
    body, other = set(), set()
    for facet in facets:
        target = body if facet.tag in selected else other
        target.update(facet.vertices)
        target.add(int(result.space.cells[facet.cell, 3+facet.local_edge]))
    if body & other:
        raise ValueError('reaction force requires a closed obstacle separated from other boundaries')
    return np.array(sorted(body), dtype=np.int64)


def obstacle_metrics(result, case):
    if result.viscous_form != 'symmetric_gradient' or result.velocity_reactions is None:
        raise ValueError('obstacle loads require physical stress and nodal reactions')
    boundary = stokes_boundary_integrals(result)
    dofs = obstacle_dofs(result, case.obstacle_tags)
    surface = -sum((np.asarray(boundary['by_tag'][tag]['force_on_fluid']) for tag in case.obstacle_tags), np.zeros(2))
    reaction = -result.velocity_reactions[dofs].sum(axis=0)
    scale = case.viscosity*case.speed
    return {'boundary_integrals': boundary,
            'max_obstacle_speed': float(np.max(np.linalg.norm(result.velocity[dofs], axis=1))),
            'surface_force_on_obstacle': surface.tolist(), 'reaction_force_on_obstacle': reaction.tolist(),
            'surface_resistance': (surface/scale).tolist(), 'reaction_resistance': (reaction/scale).tolist(),
            'relative_force_disagreement': float(np.linalg.norm(surface-reaction)/max(np.linalg.norm(reaction), 1e-30)),
            'relative_lift': float(abs(reaction[1])/max(abs(reaction[0]), 1e-30)),
            'relative_stress_balance': float(boundary['stress_force_balance']/max(np.linalg.norm(reaction), 1e-30))}

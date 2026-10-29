"""Resolve physical instances and their reproducible mesh recipes."""
import json

from physgraph.datasets.instances import PDEInstance
from physgraph.mesh.annulus import annulus_mesh
from physgraph.mesh.structured import unit_square_mesh
from physgraph.physics.annulus import radial_log_case
from physgraph.physics.parametric_square import DEFAULTS, square_case


def resolve_instance(family, item):
    if (not isinstance(item, dict) or set(item)-{'geometry', 'physics'}
            or not isinstance(item.get('geometry', {}), dict)
            or not isinstance(item.get('physics', {}), dict)):
        raise ValueError('each instance declares geometry and physics dictionaries only')
    if family == 'annulus_log':
        geometry_defaults = dict(inner_radius=.5, outer_radius=1., center=[0., 0.])
        physics_defaults = dict(inner_value=1., outer_value=2., diffusivity=1., boundary_mode='dirichlet')
        constructor = radial_log_case
    elif family == 'parametric_square':
        geometry_defaults, physics_defaults, constructor = {}, DEFAULTS, square_case
    else:
        raise ValueError(f'unsupported graph dataset family: {family}')
    if set(item.get('geometry', {}))-set(geometry_defaults) or set(item.get('physics', {}))-set(physics_defaults):
        raise ValueError(f'unknown {family} instance parameters')
    geometry = geometry_defaults | item.get('geometry', {})
    physics = physics_defaults | item.get('physics', {})
    case = constructor(**geometry, **physics)
    return PDEInstance.create(family, geometry, physics), case


def instance_mesh(instance, resolution):
    payload = json.loads(instance.canonical_json)
    if payload['family'] == 'annulus_log':
        return annulus_mesh(resolution, 8*resolution, **payload['geometry'])
    if payload['family'] == 'parametric_square':
        return unit_square_mesh(resolution)
    raise ValueError('unsupported graph dataset family')

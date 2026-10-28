import numpy as np
import pytest

from physgraph.mesh.structured import unit_square_mesh
from physgraph.fem.poisson import element_geometry


def test_mesh_counts_orientation_and_area():
    nx, ny = 4, 6
    mesh = unit_square_mesh(nx, ny)
    assert len(mesh.nodes) == (nx+1)*(ny+1)
    assert len(mesh.triangles) == 2*nx*ny
    assert mesh.boundary.sum() == 2*(nx+ny)
    areas = [element_geometry(mesh.nodes[tri])[0] for tri in mesh.triangles]
    assert all(area > 0 for area in areas)
    assert np.isclose(sum(areas), 1.0)


def test_bad_mesh_sizes():
    with pytest.raises(ValueError):
        unit_square_mesh(1)

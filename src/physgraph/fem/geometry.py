"""Geometry shared by P1 assembly and independent error integration."""
import numpy as np


def element_geometry(vertices: np.ndarray) -> tuple[float, np.ndarray]:
    """Return positive area and the three physical P1 basis gradients."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2) or not np.isfinite(vertices).all():
        raise ValueError("vertices must be a finite (3, 2) array")
    a, b, c = vertices
    twice_area = ((b[0] - a[0]) * (c[1] - a[1])
                  - (b[1] - a[1]) * (c[0] - a[0]))
    if not np.isfinite(twice_area) or twice_area <= 0:
        raise ValueError("Mesh has degenerate or clockwise-oriented triangles")
    gradients = np.array([[b[1] - c[1], c[0] - b[0]],
                          [c[1] - a[1], a[0] - c[0]],
                          [a[1] - b[1], b[0] - a[0]]]) / twice_area
    if not np.isfinite(gradients).all():
        raise ValueError("Triangle gradients are not finite")
    return float(0.5 * twice_area), gradients

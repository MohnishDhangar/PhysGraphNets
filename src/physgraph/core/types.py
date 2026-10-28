"""Stable data contracts shared by numerical, neural and visualization modules."""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


ScalarFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]
VectorFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class Mesh2D:
    """P1 triangular mesh; nodes (N,2), triangles (T,3), boundary (N,)."""

    nodes: NDArray[np.float64]
    triangles: NDArray[np.int64]
    boundary: NDArray[np.bool_]
    boundary_edges: NDArray[np.int64] | None = None
    boundary_edge_tags: NDArray[np.str_] | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(a, np.ndarray) for a in (self.nodes, self.triangles, self.boundary)):
            raise ValueError("mesh inputs must be NumPy arrays")
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must have shape (N, 2)")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (T, 3)")
        if self.boundary.shape != (len(self.nodes),):
            raise ValueError("boundary must have shape (N,)")
        if self.boundary.dtype != np.bool_:
            raise ValueError("boundary must have boolean dtype")
        if not np.issubdtype(self.triangles.dtype, np.integer):
            raise ValueError("triangle indices must have integer dtype")
        if not np.issubdtype(self.nodes.dtype, np.floating):
            raise ValueError("nodes must have floating-point dtype")
        if len(self.nodes) < 3 or len(self.triangles) == 0:
            raise ValueError("mesh must contain nodes and triangles")
        if not np.isfinite(self.nodes).all():
            raise ValueError("nodes must be finite")
        if np.any(self.triangles < 0) or np.any(self.triangles >= len(self.nodes)):
            raise ValueError("triangle indices out of range")
        if len(np.unique(self.triangles)) != len(self.nodes):
            raise ValueError("mesh contains unused nodes")
        canonical = np.sort(self.triangles, axis=1)
        if np.any(np.diff(canonical, axis=1) == 0):
            raise ValueError("triangle has repeated vertices")
        if len(np.unique(canonical, axis=0)) != len(canonical):
            raise ValueError("mesh contains duplicate triangles")
        if (self.boundary_edges is None) != (self.boundary_edge_tags is None):
            raise ValueError("boundary edges and tags must be supplied together")
        if self.boundary_edges is not None:
            edges, tags = self.boundary_edges, self.boundary_edge_tags
            if (not isinstance(edges, np.ndarray) or edges.ndim != 2 or edges.shape[1] != 2
                    or not np.issubdtype(edges.dtype, np.integer) or len(edges) == 0):
                raise ValueError("boundary_edges must be a nonempty integer (E, 2) array")
            if (not isinstance(tags, np.ndarray) or tags.shape != (len(edges),)
                    or tags.dtype.kind != "U" or np.any(np.char.strip(tags) == "")):
                raise ValueError("boundary_edge_tags must contain one nonempty Unicode string per edge")
            if (np.any(edges < 0) or np.any(edges >= len(self.nodes))
                    or np.any(edges[:, 0] == edges[:, 1])):
                raise ValueError("invalid boundary edge indices")
            if len(np.unique(np.sort(edges, axis=1), axis=0)) != len(edges):
                raise ValueError("duplicate tagged boundary edge")


@dataclass(frozen=True)
class ManufacturedCase:
    """A test problem for -Laplace(u) = forcing, with prescribed u on boundary."""

    name: str
    exact: ScalarFunction
    forcing: ScalarFunction
    gradient: VectorFunction


@dataclass(frozen=True)
class FieldResult:
    """Mesh-aligned scalar solution and independently computed reference."""

    mesh: Mesh2D
    solution: NDArray[np.float64]
    exact: NDArray[np.float64]
    case: str
    residual_relative: float
    l2_relative: float
    h1_seminorm_relative: float
    linf_nodes: float
    equation: str = "-laplacian(u) = f"
    boundary_type: str = "Dirichlet"

    def __post_init__(self) -> None:
        for values in (self.solution, self.exact):
            if values.shape != (len(self.mesh.nodes),) or not np.isfinite(values).all():
                raise ValueError("fields must contain one finite value per mesh node")
        metrics = (self.residual_relative, self.l2_relative,
                   self.h1_seminorm_relative, self.linf_nodes)
        if not all(np.isfinite(m) and m >= 0 for m in metrics):
            raise ValueError("field error metrics must be finite and nonnegative")

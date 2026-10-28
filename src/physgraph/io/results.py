"""Portable, non-pickle result files for simulations and viewer integration."""
import json
from pathlib import Path

import numpy as np

from physgraph.core.types import FieldResult, Mesh2D
from physgraph.fem.boundary import exterior_edges


SCHEMA_VERSION = 2


def save_field(path: str | Path, field: FieldResult) -> None:
    """Store mesh + scalar results in compressed .npz, no pickle necessary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if field.mesh.boundary_edges is not None:
        exterior_edges(field.mesh)  # Require a complete, valid boundary tagging.
        extra = {"boundary_edges": field.mesh.boundary_edges,
                 "boundary_edge_tags": field.mesh.boundary_edge_tags}
    metadata = {
        "schema_version": SCHEMA_VERSION if extra else 1,
        "case": field.case,
        "equation": field.equation,
        "boundary_type": field.boundary_type,
        "units": "dimensionless",
        "relative_residual": field.residual_relative,
        "relative_l2": field.l2_relative,
        "relative_h1_seminorm": field.h1_seminorm_relative,
        "max_nodal_error": field.linf_nodes,
    }
    np.savez_compressed(path,
                        nodes=field.mesh.nodes,
                        triangles=field.mesh.triangles,
                        boundary=field.mesh.boundary,
                        solution=field.solution,
                        exact=field.exact,
                        metadata=json.dumps(metadata), **extra)


def load_field(path: str | Path) -> tuple[FieldResult, dict]:
    """Load and validate a field archive without loading pickle objects."""
    with np.load(path, allow_pickle=False) as arr:
        metadata = json.loads(str(arr["metadata"].item()))
        schema = metadata.get("schema_version")
        if schema not in (1, SCHEMA_VERSION):
            raise ValueError("Unsupported field schema version")
        extra = {}
        if schema == 2:
            if not {"boundary_edges", "boundary_edge_tags"}.issubset(arr.files):
                raise ValueError("schema v2 requires boundary edges and tags")
            extra = {"boundary_edges": arr["boundary_edges"].copy(),
                     "boundary_edge_tags": arr["boundary_edge_tags"].copy()}
        mesh = Mesh2D(nodes=arr["nodes"].copy(),
                      triangles=arr["triangles"].copy(),
                      boundary=arr["boundary"].copy(), **extra)
        if schema == 2:
            exterior_edges(mesh)
        solution = np.asarray(arr["solution"], dtype=float).copy()
        exact = np.asarray(arr["exact"], dtype=float).copy()
    if solution.shape != (len(mesh.nodes),) or exact.shape != solution.shape:
        raise ValueError("Stored fields have incorrect shapes")
    field = FieldResult(mesh=mesh, solution=solution, exact=exact,
                        case=metadata["case"],
                        residual_relative=float(metadata["relative_residual"]),
                        l2_relative=float(metadata["relative_l2"]),
                        h1_seminorm_relative=float(metadata["relative_h1_seminorm"]),
                        linf_nodes=float(metadata["max_nodal_error"]),
                        equation=metadata.get("equation", "-laplacian(u) = f"),
                        boundary_type=metadata.get("boundary_type", "Dirichlet"))
    return field, metadata

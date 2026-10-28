"""2D scientific field visualization for triangular unstructured meshes."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless output works on Arch, Kaggle and Colab.
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
import numpy as np

from physgraph.core.types import FieldResult


def plot_comparison(field: FieldResult, output: str | Path,
                    show_edges: bool = False) -> Path:
    """Save side-by-side FEM, analytical, signed error and mesh views."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tri = mtri.Triangulation(field.mesh.nodes[:, 0], field.mesh.nodes[:, 1],
                            field.mesh.triangles)
    predicted, exact = field.solution, field.exact
    error = predicted - exact
    lo, hi = min(predicted.min(), exact.min()), max(predicted.max(), exact.max())
    err_cap = max(float(np.max(np.abs(error))), 1e-15)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, (title, values, cmap, vmin, vmax) in zip(
        axes.flat,
        [("FEM solution", predicted, "viridis", lo, hi),
         ("Analytical solution", exact, "viridis", lo, hi),
         ("Nodal signed error (FEM - exact)", error, "coolwarm", -err_cap, err_cap)]
        + [("Mesh", None, None, None, None)],
    ):
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if values is None:
            ax.triplot(tri, linewidth=0.3, color="slategray")
            if field.mesh.boundary_edges is not None:
                for index, tag in enumerate(np.unique(field.mesh.boundary_edge_tags)):
                    edges = field.mesh.boundary_edges[field.mesh.boundary_edge_tags == tag]
                    ax.add_collection(LineCollection(field.mesh.nodes[edges],
                                                      colors=f"C{index % 10}",
                                                      linewidths=1.4, label=str(tag)))
                ax.legend(loc="upper right", fontsize=8)
        else:
            pcm = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap,
                               vmin=vmin, vmax=vmax)
            fig.colorbar(pcm, ax=ax, shrink=0.78)
            if show_edges:
                ax.triplot(tri, linewidth=0.17, color="white", alpha=.6)
    fig.suptitle(f"PhysGraph | {field.case} | {len(field.mesh.nodes)} nodes",
                 fontsize=13)
    fig.savefig(output, dpi=155)
    plt.close(fig)
    return output


def plot_convergence(rows: list[dict], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    h = np.array([r["h_max"] if "h_max" in r else 1/r["nx"] for r in rows], dtype=float)
    l2 = np.array([r["relative_l2"] for r in rows])
    h1 = np.array([r["relative_h1_seminorm"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(h, l2, "o-", label="Relative L2")
    ax.loglog(h, h1, "s-", label="Relative H1 seminorm")
    ax.invert_xaxis()
    ax.set_xlabel("Maximum element diameter h" if "h_max" in rows[0] else "Mesh spacing h = 1/n")
    ax.set_ylabel("Relative error")
    ax.set_title("P1 FEM convergence against manufactured solution")
    ax.grid(True, which="both", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=155)
    plt.close(fig)
    return output

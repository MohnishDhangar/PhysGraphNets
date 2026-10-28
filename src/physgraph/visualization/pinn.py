"""Exact/FEM/PINN comparisons with separately named fields and errors."""
from pathlib import Path
import numpy as np
from physgraph.visualization.plotting import plt, mtri


def plot_pinn_comparison(mesh, exact, fem, prediction, output):
    tri = mtri.Triangulation(*mesh.nodes.T, triangles=mesh.triangles)
    lo = min(float(a.min()) for a in (exact, fem, prediction))
    hi = max(float(a.max()) for a in (exact, fem, prediction))
    errors = (fem-exact, prediction-exact, prediction-fem)
    cap = max(float(np.max(np.abs(e))) for e in errors)
    cap = max(cap, 1e-15)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    panels = [('Analytical', exact), ('CPU FEM', fem), ('PINN', prediction),
              ('FEM − analytical', errors[0]), ('PINN − analytical', errors[1]), ('PINN − FEM', errors[2])]
    for i, (ax, (title, values)) in enumerate(zip(axes.flat, panels)):
        artist = ax.tripcolor(tri, values, shading='gouraud', cmap='viridis' if i < 3 else 'coolwarm',
                             vmin=lo if i < 3 else -cap, vmax=hi if i < 3 else cap)
        ax.set(title=title, xlabel='x', ylabel='y', aspect='equal')
        fig.colorbar(artist, ax=ax, shrink=.8)
    fig.suptitle('Independent nodal fields; continuous PINN errors use direct quadrature evaluation')
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return Path(output)


def plot_training(losses, output):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(len(losses)), np.maximum(losses, 1e-30))
    ax.set(xlabel='Adam step', ylabel='Normalized strong-form residual MSE', title='Fixed-collocation PINN training')
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

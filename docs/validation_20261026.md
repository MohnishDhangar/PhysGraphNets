# Local validation record — 26 October 2026

The user executed the tests and experiments in `.fenicsx-env`. Test counts and
elapsed times below are from the supplied terminal output. Comparison and
convergence metrics were inspected directly in the saved manifests. This status
update did not rerun numerical work.

## Executed checks

- Focused FEniCSx suite: **25 passed, 1 skipped** (12.68 seconds in the latest reported run).
- Complete suite: **161 passed, 1 skipped** (39.24 seconds).
- The skipped test intentionally exercises missing dependencies; the 12 external integration tests ran.
- Comparison runner: **24/24 passed**, covering 12 fixtures at resolutions 4 and 8.
- Original Poisson refinement demo: passed for 8×8, 16×16 and 32×32 meshes.
- User-reported PyTorch check: **2.10.0+cpu**, CUDA build `None`, second derivative of x³ at x=2 equal to 12.

## Provenance

Python: **3.12.13**; CPU float64; DOLFINx **0.10.0**;
Basix **0.10.0**; UFL **2025.2.1**; PETSc **3.25.6**.
NumPy **2.5.4**; SciPy **1.18.1**.

Git revision is unavailable in this workspace. Both saved experiment manifests
record the same source SHA-256:
`25cb288177f3df31dc8d63a81217bb66aeec948071eb94ae54413fa3c623ff39`.

[Comparison manifest](../outputs/fenicsx/20261026T181836Z_c2b6cdc5/manifest.json)
and [convergence manifest](../outputs/fenicsx_environment_convergence/convergence_manifest.json)
contain full settings, case names, node/triangle counts and errors. These
generated artifacts remain outside version control; this record retains their
provenance and numerical summary.

## External comparison results

| Metric | Maximum over 24 comparisons |
|---|---:|
| Absolute nodal solution difference | 1.850742e-13 |
| Relative stiffness-matrix difference | 5.260155e-16 |
| Relative mass-matrix difference | 2.220839e-16 |
| Load difference with documented normalization | 5.182626e-15 |
| Absolute prescribed-boundary error | 2.220446e-16 |

The worst nodal difference occurred for `airfoil_airfoil_neumann` at n=8.
The configured parity threshold was 1e-8. The comparisons cover graded squares,
annuli, rectangle/cylinder domains and symmetric airfoil exteriors, including
nonzero Dirichlet data and mixed boundaries. Independent UFL/PETSc assembly and
solving agree with the CPU reference on these shared meshes.

## Poisson mesh convergence

| Subdivisions | Relative L2 | Relative H1 seminorm | L2 rate | H1 rate |
|---|---:|---:|---:|---:|
| 8×8 | 0.04221224 | 0.19437753 | — | — |
| 16×16 | 0.01075142 | 0.09792576 | 1.9731 | 0.9891 |
| 32×32 | 0.00270066 | 0.04905618 | 1.9931 | 0.9973 |

The observed rates approach 2 in L2 and 1 in H1, consistent with the existing
smooth manufactured P1 reference. This closes the independent FEM verification
gate for the implemented fixture set and tested resolutions; it does not validate
unimplemented PDE families or arbitrary future inputs.

## Next work

Historical status below is superseded for the PyTorch backend by the
[27 October implementation and validation](validation_20261027.md).

Implement and verify the PyTorch/autodiff PINN backend against the existing
NumPy model, residuals and boundary enforcement. Installation and a second-
derivative smoke check are complete; the production PINN runner still uses NumPy.
Actual cloud execution and broader neural studies remain pending.

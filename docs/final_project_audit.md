# PhysGraphNets project audit — 17 November 2026

**This is an interim audit of an incomplete project, not a completion or release
certificate.** The implementation assignment remains open. The original objective
is a mathematically verified, reproducible PIGNN framework with mesh/geometry
generalization, fair baseline comparisons, qualified transient fluids, measured
acceleration, deployment and a supported publication contribution.

## Implementation inventory and acceptance gates

| Requested deliverable | Current status | Evidence or outstanding requirement |
|---|---|---|
| 1. Scalar weak-form PIGNN | IMPLEMENTED / TESTED | Independent P1 residual parity, gradcheck, finite differences, FEM recovery |
| 2. Three training modes | IMPLEMENTED / TESTED | Physics never accesses labels; supervised MSE/Sobolev; explicit hybrid weights; exact resume |
| 3. Mesh-general architecture | IMPLEMENTED / TESTED; scoped n=3 gates PASS, resolution transfer FAILED | FEM-weighted local/two-level correction improves all five seeds; broad qualification remains open |
| 4. External mesh ingestion | IMPLEMENTED / TESTED within scope | JSON / linear ASCII Gmsh 2.2; overlap certification PENDING |
| 5. Multi-geometry datasets | IMPLEMENTED instance factory; benchmark PENDING | Square/nonuniform, annulus, cylinder, airfoil scalar cases; no fresh final benchmark |
| 6. Multi-seed qualification | IMPLEMENTED / TESTED; broad holdout FAILED | Five-seed validation and locked n=3/5 protocol executed; every n=3 seed passes, every n=5 seed fails |
| 7. Six-way baseline comparisons | IMPLEMENTED matched scalar objectives; full six-way comparison PENDING | Physics, MSE, Sobolev and two hybrids compared; existing PINN/qualified-square GNN contracts differ |
| 8. Numerical metrics and ablations | IMPLEMENTED / TESTED; full performance qualification PENDING | Flux/conservation, analytical errors, cached-call timing, fixed/adaptive gradients, field/gradient/residual maps added |
| 9. Qualified transient reference and fluid PIGNN | FAILED reference gate / PENDING surrogate | Historical cylinder refinement failure unchanged |
| 10. Optional GPU acceleration | PENDING | CPU-only Torch; no new GPU backend or speed claim |
| 11. Interactive viewer | PENDING | Deferred until primary scientific dependencies qualify |
| 12. Installation and CLI | IMPLEMENTED new orchestration; DEPLOYABLE status PENDING | New train/study/import commands; clean-clone release verification outstanding |
| 13. CI/deployment | IMPLEMENTED additional smoke job; hosted TESTED status PENDING | Local suite passed; hosted CI/container/release checks open |
| 14. Math/developer documentation | IMPLEMENTED for new scalar increment | Plan, status, decisions, mesh contract and measured validation report |
| 15. Bibliography and licensing | IMPLEMENTED verified bibliography; licensing PENDING | Seven primary records; bundled PDF/video permissions not established |
| 16. Publication figures and tables | PENDING | Development tables measured; full scientific comparison figures not generated |
| 17. Manuscript/reproducibility archive | PENDING manuscript; IMPLEMENTED per-run source archives | No unsupported manuscript claims |
| 18. Final qualification report | IMPLEMENTED interim audit; final completion PENDING | This file explicitly records unmet gates |

Gate A has scoped scalar correctness/training evidence. Gates B/C remain open:
the original multigeometry validation FAILED; the numerical recovery passes its
unchanged n=3 validation/holdout gates but fails finer-resolution transfer.
Gate D remains FAILED
at the transient reference prerequisite. Gate E has no new measured acceleration.
Gates F/G remain PENDING. The project is neither generally QUALIFIED nor
DEPLOYABLE nor publication-ready.

## Mathematical formulation and architecture

For `-div(k grad u)+c u=f`, local P1 integrals form `R_I=(Au-F)_I`, including
source and conormal flux with correct weak signs. `where(mask,g,prediction)`
enforces essential nodal values exactly before residual evaluation. The energy
dual squared residual `R_I^T A_II^-1 R_I` equals discrete energy error squared;
the alternative lumped-mass residual has different high-frequency weighting.
Both use explicitly documented physical scaling. Coefficients and boundary data
are sampled without reference solution labels. The SciPy FEM solver remains
independent. Dense energy preconditioning is bounded and is not a speed benefit.

The old local and fixed-square global architectures remain baselines. The new
candidate adds normalized residual message blocks, optional per-graph global mean
exchange and coordinate-conditioned decoding without fixed anchors. General
boundary data are prescribed directly. Vector outputs are architecture support
only; scalar residuals do not implement Navier–Stokes.

See [technical decisions](technical_decisions.md) and
[external-mesh support limits](pignn_mesh_contract.md).

## Data, measurements and reproducibility

Physical identities resolve parameter defaults and exclude mesh resolution and
grading. Training and validation identities must be disjoint across refinements.
Normalization fits training inputs only. Physics checkpoint selection uses only
validation physics. No final test instances, historical holdout predictions or
test-based model selection were used.

New experiment outputs are exclusively created under ignored
`outputs/pignn_development_20261116/`. Configs, input/normalization checksums,
optimizer/RNG/checkpoints, source revision/fingerprint and predictions are saved.
The final driver additionally archives its source and rejects source mutation.
Early development runs lacked source snapshots; that limitation is documented.
Failed accuracy studies remain intact. No generated output or third-party media
was committed, deleted or redistributed in this work.

Hardware: AMD Ryzen 5 5600H, 12 logical CPUs, 7,611,854,848 bytes installed RAM.
PCI enumeration reports AMD Navi 24 and Cezanne graphics; no NVIDIA device was
assumed. Installed Torch 2.10.0 is CPU-only; Python 3.12.13. No GPU VRAM or GPU
performance result was measured. Training used CPU FP64 and one Torch thread.

Final full regression: **545 passed, 10 skipped in 113.85 seconds**. Skips:
one intentional missing-FEniCSx path and nine explicit real-GPU checks. Original
n=8/16/32 convergence retained approximately second-order L2 and first-order H1.
Physics-only and hybrid smoke validation reached L2/H1 .00802/.0570 and
.00709/.0562 at n=4, seed 0, with exact boundaries. These are tiny development
experiments. Five-seed n=3 multigeometry accuracy failed: global mean L2/H1
.0723/.581 versus declared .05/.2 mean limits. All seeds are reported.

The complete numerical rows, timing/RSS methodology, source identities and exact
reproduction commands are in [the validation report](validation_pignn_20261117.md).

## Outstanding engineering and research

Continue training/validation-controlled architecture, conditioning and dataset
improvements before declaring a fresh final holdout. Establish cross-resolution,
coefficient, boundary and geometry extrapolation experiments; report all five
seeds, full baseline costs and physical flux/conservation metrics. A random
parameter split is not arbitrary-geometry generalization. The failed global
candidate is not adopted as a scientifically qualified replacement.

Complete the existing three-level cylinder study without changing its gates
before learned fluid qualification. Then profile real PIGNN workloads before
GPU optimization. Viewer, clean installation/wheel/container/release checks,
redistribution permissions and manuscript preparation remain subsequent work.
No factual evidence currently supports software licensing permissions for all
bundled papers/videos. See [publication readiness](publication_readiness.md)
for the verified literature and unresolved contribution requirements.

## Executed numerical recovery update

The [recovery investigation](pignn_recovery_20261117.md) and
[all-seed numerical tables](pignn_recovery_results_20261117.md) supersede the
current-status interpretation of the earlier development numbers above; their
historical reports and artifacts remain unchanged.

The original local/global scores were reproduced exactly. A same-budget,
same-data, five-seed ablation reduced physics-only mean/max FEM H1 from
.581289/1.044439 to .070796/.207861, using four local FEM messages plus a small
coarse Galerkin correction. No learnable parameter was added by that correction.
It is a neural-plus-numerical method, and numerical-only controls show that some
of the square benefit already comes from the solver. All five seeds passed
the original n=3 validation gates with zero Dirichlet error.

Five matched objective variants all passed validation. Physics-only and the
validation-selected gradient-balanced hybrid were frozen before a checksummed
90-row-per-candidate holdout. Both passed every seed at n=3 and failed every
seed at n=5. Physics-only n=5 mean/max L2 were .053434/.149765 and H1
.362177/.838674; hybrid n=5 H1 was .362106/.841028. These failures remain FAILED.
Boundary errors were zero and tested permutation differences below 3.2e-15.
The heldout airfoil family, changed radii and mixed boundaries are reported
separately; no universal geometry generalization is claimed.

Read-only stage replay exactly reproduced frozen predictions. It showed larger
learned-initialization errors and weaker fixed-correction contraction on finer
meshes: cylinder free DOFs grow 192→640 while coarse DOFs remain eight, and
the Jacobi minimum eigenvalue drops .01155→.00406. A mesh-independent convergence
or learned-operator claim is not justified. The inspected holdout is now spent
and must not be reused to select new models.

The recovery regression completed with 554 passed / 10 skipped; the mesh demo
retained its original n=8/16/32 errors. The graded affine mixed-flux diagnostic
passed at approximately 1e-15. New artifacts include immutable configurations,
source snapshots, checkpoints, gradient disagreement histories, physical flux
and conservation diagnostics, analytical versus FEM errors, timing/memory and
worst-case field/gradient/residual maps. No GPU parity, GPU speedup or hosted CI
result was measured. Source work remains local and uncommitted.

The saved transient reference audit also corrects an earlier possible inference:
the complete three-level spatial/time study exists, and its finest trajectory
passes the published benchmark including lift timing. Summary/history refinement
gates still fail. No reference solver change or four-hour rerun was made here.

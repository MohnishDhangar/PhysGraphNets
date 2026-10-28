# PhysGraph: Research, Architecture, and Development Roadmap

**Project:** Physics-Informed Graph Neural PDE Solver  
**Status:** CPU elliptic reference independently checked with FEniCSx; initial NumPy PINN implemented; PyTorch backend, GNN and CUDA remain pending  
**Document version:** 1.6 — FEniCSx comparison passed; CPU PyTorch installed, 26 October 2026  
**Intended audience:** Developer/researcher, future collaborators, and Codex sessions

> **Core principle:** First make the numerical solver *exist and be demonstrably correct*. Then train and verify the PINN/GNN components. Measure memory and speed using reference implementations. **Only after identifying real bottlenecks should custom CUDA kernels be introduced.**
>
> This document consolidates our architectural decisions and an initial literature survey. It is **not** an exhaustive systematic literature review, and proposed research benefits must be demonstrated experimentally rather than presumed.

## Contents

1. [Research objective and scope](#1-research-objective-and-scope)
2. [Literature survey and reading order](#2-literature-survey-and-reading-order)
3. [Mathematical formulation](#3-mathematical-formulation)
4. [Proposed numerical/neural architecture](#4-proposed-numericalneural-architecture)
5. [Data interfaces and storage](#5-data-interfaces-and-storage)
6. [Hardware and development environments](#6-hardware-and-development-environments)
7. [Repository and script maps](#7-repository-and-script-maps)
8. [Experimental design and metrics](#8-experimental-design-and-metrics)
9. [Two-dimensional visualization design](#9-two-dimensional-visualization-design)
10. [Deferred CUDA strategy](#10-deferred-cuda-strategy)
11. [Phased roadmap and acceptance gates](#11-phased-roadmap-and-acceptance-gates)
12. [Current implementation and how to run it](#12-current-implementation-and-how-to-run-it)
13. [Risks, research hypotheses, and open decisions](#13-risks-research-hypotheses-and-open-decisions)
14. [Reference links](#14-reference-links)

---

## 1. Research objective and scope

### 1.1 What we are building

PhysGraph is a modular framework for approximating families of PDE solution operators on two-dimensional finite-element meshes. It integrates:

- **Analytical solutions** for manufactured test cases and exact verification.
- **Finite Element Method (FEM)** as a trusted numerical baseline and provider of discrete weak-form residuals.
- **Physics-Informed Neural Networks (PINNs)** as a conventional pointwise neural baseline.
- **Graph Neural Networks (GNNs)** to learn solution fields on irregular meshes.
- **Physics-informed GNN training**, ideally with weak-form/variational objectives rather than only nodewise strong-form derivatives.
- **Interactive 2D visualizations** inspired by the immediacy and diagnostic value of Sebastian Lague-style simulators.
- **Custom CUDA kernels**, *after* profiling indicates a worthwhile memory-traffic bottleneck.

The goal is to learn a **solution operator across many problem instances**, not train one network on one forcing function and call it a general solver. The first experiments will be with second-order elliptic problems. General-purpose means extensible across a well-defined set of PDE families, **not** a universal black-box replacement for all FEM solvers.

### 1.2 Research questions

1. Can a mesh GNN map geometry, coefficients, forcing, and boundary conditions to a solution on *unseen PDE instances*?
2. Does adding physics-informed weak-form loss improve label efficiency or out-of-distribution generalization?
3. Can the model transfer across mesh resolutions and, eventually, mesh geometries?
4. Which operations dominate GPU memory and wall-clock time during training?
5. Does a **measured** custom fused GPU implementation improve the accuracy–memory–latency trade-off over optimized existing graph operators?

### 1.3 Scope freeze

| Topic | First working target | Later extensions |
|---|---|---|
| Spatial dimension | 2D | 3D |
| PDE | Poisson | Variable diffusion/reaction; heat equation; elasticity/Burgers |
| FE elements | Linear P1 triangles | Higher-order or mixed elements |
| Boundaries | Dirichlet | Neumann, Robin, mixed |
| Domain | Unit square | L-shape, holes, nonuniform/irregular meshes |
| Baselines | Exact + FEM | Standard MLP PINN, supervised GNN |
| Physics loss | Reduced-system residual | Energy, combined objectives, better preconditioning |
| Visualization | Field plots, errors, meshes | Interactive WebGL viewer, probes, timeline |
| Performance | Reference CPU/GPU profiling | Kernel fusion and optimization |

---

## 2. Literature survey and reading order

This survey is grounded in accessible paper abstracts, article descriptions, and available methods pages; selected papers have accessible full-text HTML. It identifies relevant established work, **not** a proven novelty gap.

### 2.1 Primary reading sequence

| Order | Paper | Insight to borrow | What we should reproduce or check |
|---:|---|---|---|
| 1 | **Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks* (ICLR 2021)** [R1] | Encoder–processor–decoder, geometric edge features, message passing over mesh connectivity | Basic MeshGraphNet components; distinction between static solution field and time-step prediction |
| 2 | **Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks* (JCP 2019)** [R2] | Strong-form residuals and automatic differentiation | Independent coordinate-based PINN baseline and its boundary-condition handling |
| 3 | **Gao, Zahr & Wang, *Physics-informed graph neural Galerkin networks* (CMAME 2022)** [R3] | Discrete graph learning married to Galerkin/finite-element physics; exact essential constraints | Variational loss, reduced finite-element degrees of freedom, behavior on irregular meshes |
| 4 | **Würth et al., *Physics-informed MeshGraphNets (PI-MGNs)* (CMAME 2024)** [R4] | Physics-based MeshGraphNet training on arbitrary meshes, including more complex time-dependent problems | Label-free training, small-to-large mesh transfer, fair FEM reference evaluation |
| 5 | **Chenaud, Magoulès & Alves, *Physics-Informed Graph-Mesh Networks for PDEs* (2024)** [R5] | Combining graph learning with numerical kernels and physical invariances | Numerical gradient vs autodiff choices, handling complex geometries |
| 6 | **Li et al., *Multipole Graph Neural Operator for Parametric PDEs* (NeurIPS 2020)** [R6] | Long-range effects and multilevel graph information | Does a local message-passing GNN struggle on elliptic global dependencies? |
| 7 | **Kovachki et al., *Neural Operator: Learning Maps Between Function Spaces* (JMLR 2023)** [R7] | Learning mappings between function spaces; discretization/mesh questions | Define genuine operator generalization rather than only fitting a fixed-resolution vector map |

**Additional implementation and systems reading:**

- **Zhang et al., *Combining physics-informed graph neural network and finite difference for solving forward and inverse spatiotemporal PDEs*** [R8]: alternative discrete-physics formulation; use as a comparator, not evidence that finite differences and FEM are interchangeable.
- **PyTorch Geometric, *Memory-Efficient Aggregations*** [R9]: gather/scatter memory behavior and already-available sparse/fused operators. This is essential before claiming a novel kernel optimization.
- **GNNAdvisor** [R10]: performance differences across graph degrees and GPU workload scheduling.
- **FEniCSx documentation** [R11]: eventual independent FEM cross-check, especially beyond the initial hand-coded P1 elements.
- **Finite-element physics-informed operator learning** [R12]: relevant evidence that FEM variational learning is already an established approach; revisit for novelty assessment.

### 2.2 Main design lessons

1. **Mesh-aware, not image-grid-only.** A triangular mesh and its element list must remain first-class objects. Graph edges do not by themselves retain all element and quadrature information.
2. **Physics evaluated on the FEM discretization.** When a GNN outputs nodal degrees of freedom, use the assembled FE algebraic/variational problem, rather than assuming automatic differentiation of arbitrary nodal outputs provides a continuum gradient.
3. **Predict across cases.** Vary source terms, coefficient fields and boundaries; split by problem instance, not by individual nodes from the same solution.
4. **Boundary conditions matter.** Exact Dirichlet enforcement reduces an unnecessary optimization burden.
5. **Local graph propagation has limitations.** Poisson/elliptic solutions have long-range dependence; depth, coarse-scale pathways and conditioning need experiments.
6. **Numerical correctness precedes ML.** Test matrix assembly, load quadrature, element orientation, boundary elimination and convergence independently.
7. **Existing optimized aggregations set the baseline.** PyG supports fused message-and-aggregate patterns for applicable operators. A new kernel must surpass the best suitable existing implementation, not an unnecessarily memory-hungry strawman.

### 2.3 Literature matrix template

For each new paper, record: exact citation and DOI, code availability, PDE family, geometry, boundary conditions, weak/strong physics formulation, graph representation, number of training examples, model inputs/outputs, accuracy metrics, memory/hardware, observed speedups, generalization tests, reproducibility gaps, and direct relevance to PhysGraph.

**Publication caution:** The combination **PINN + GNN + FEM is not, by itself, a new research contribution**. A defensible contribution will require new evidence: better memory/throughput, mesh generalization, useful operator design, or rigorous accuracy–efficiency comparisons. Recheck relevant 2025–2026 work before writing a paper claim.

---

## 3. Mathematical formulation

### 3.1 Abstract operator learning formulation

Let a PDE instance be specified by domain `Ω`, operator parameters `μ`, forcing `f`, and boundary data `g`:

$$
\mathcal{N}(u;\mu)=f \quad \text{in }\Omega,\qquad
\mathcal{B}(u;\mu)=g \quad \text{on }\partial\Omega.
$$

The target learned map is

$$
\mathcal{G}_{\theta}:(\Omega,\mu,f,g)\mapsto\hat u.
$$

For irregular meshes, the output is a set of approximated finite-element degrees of freedom `U_hat` on the supplied mesh. This yields a practical **mesh-conditioned surrogate**. Whether it deserves stronger discretization-invariance claims must be established experimentally.

### 3.2 First generalizable elliptic family

Once the Poisson baseline is verified, extend to

$$
-\nabla\cdot\big(k(\mathbf{x})\nabla u\big)
+c(\mathbf{x})u=f(\mathbf{x}) \quad\text{in }\Omega,
$$

with `k(x) >= k_min > 0`, `c(x) >= 0`, and Dirichlet boundary values `u=g` (initially). The scalar `k` can later be replaced by a symmetric positive-definite diffusion tensor.

### 3.3 Weak form and finite-element discretization

For zero Dirichlet values and test functions `v` vanishing on that boundary:

$$
\int_\Omega k\nabla u\cdot\nabla v\,d\Omega
+\int_\Omega c u v\,d\Omega
=\int_\Omega f v\,d\Omega.
$$

Using linear triangular shape functions `φ_j`:

$$
u_h(\mathbf{x})=\sum_{j=1}^{N}U_j\phi_j(\mathbf{x}),
$$

$$
K_{ij}=\int_\Omega
\big(k\nabla\phi_i\cdot\nabla\phi_j+c\phi_i\phi_j\big)\,d\Omega,
\quad F_i=\int_\Omega f\phi_i\,d\Omega.
$$

Thus `K U = F` before boundary elimination. Partition free degrees `I` and prescribed degrees `B`:

$$
K_{II}U_I = F_I-K_{IB}g_B.
$$

**Implementation requirement:** the FEM reference and graph-learning loss consume the **same well-verified discrete operator**, but their construction and solver code should remain independently testable.

### 3.4 Manufactured analytical reference

For the unit square `Ω=[0,1]^2`, start with

$$
u(x,y)=\sin(\pi x)\sin(\pi y),\quad u\vert_{\partial\Omega}=0,
$$

$$
-\Delta u=2\pi^2\sin(\pi x)\sin(\pi y).
$$

Also use an affine exact solution with **nonzero** Dirichlet boundaries to catch incorrect constraint elimination. Manufactured cases should expand as new operators are introduced.

### 3.5 Physics and data losses

The interior algebraic residual is

$$
r_I=K_{II}\hat U_I+K_{IB}g_B-F_I.
$$

A dimensionless-like normalized residual for a nonzero effective right-hand side is

$$
\mathcal L_R=\frac{\|r_I\|_2^2}
{\|F_I-K_{IB}g_B\|_2^2+\varepsilon}.
$$

Its interpretation depends on scaling and mesh conditioning; a tiny algebraic residual **does not** guarantee a small continuum solution error. Study energy norms and preconditioning, too.

For symmetric positive-definite `K_II`, a variational objective is

$$
\mathcal L_E=\frac12\hat U_I^{\mathsf T}K_{II}\hat U_I
-\hat U_I^{\mathsf T}(F_I-K_{IB}g_B).
$$

This energy may be negative. For comparison or combined training, define a consistent shift/normalization; do **not** simply assume differently scaled loss terms can be added without adjustment.

With labeled FEM solutions, use a mass-matrix-weighted relative error:

$$
\mathcal L_D=
\frac{(\hat U-U_{\mathrm{FEM}})^{\mathsf T}M
(\hat U-U_{\mathrm{FEM}})}
{U_{\mathrm{FEM}}^{\mathsf T}M U_{\mathrm{FEM}}+\varepsilon}.
$$

Investigate **physics-only**, **supervised-only**, and **hybrid** training as distinct experiments. Hard-impose `U_hat[B]=g_B` in the first elliptic tasks.

### 3.6 Time-dependent extension (later)

A next PDE family is heat transfer:

$$
\rho c_p\frac{\partial u}{\partial t}
-\nabla\cdot(k\nabla u)=f.
$$

Define an explicit time-stepping or implicit stepping residual, e.g. backward Euler for the assembled mass `M` and stiffness `K`:

$$
\left(\frac{M}{\Delta t}+K\right)U^{n+1}
=\frac{M}{\Delta t}U^n+F^{n+1}.
$$

Do not introduce it before the stationary solver generalizes across a family of cases.

---

## 4. Proposed numerical/neural architecture

```text
     PDE family + coefficients + forcing + boundary data
                             |
                     validated Mesh2D
                             |
                    +--------+---------+
                    |                  |
                FEM assembly       graph construction
                (K, F, M)          (nodes, edges, features)
                    |                  |
          +---------+-----+       encoder + processor
          |               |           + decoder
       FEM solver     physics losses        |
          |               +---------- predicted nodal field
          |                               |
          +----> numerical verification <-+
                       |
                  SolutionField
                       |
            reproducible storage + 2D viewer
```

### 4.1 Modular model

- **Encoder:** transform physical node and edge features into hidden vectors.
- **Processor:** repeated message-passing blocks, initially 3 layers at hidden width 32.
- **Decoder:** output a nodal scalar `U_hat` for stationary Poisson/diffusion.
- **Physics module:** use FEM operators to evaluate residual/energy; do not entangle matrix assembly with learned inference.

Possible node features:

$$
h_i^{(0)}=[x_i,y_i,k_i,c_i,f_i,b_i,g_i].
$$

Possible directed edge features:

$$
e_{ij}=[x_j-x_i,y_j-y_i,\|\mathbf{x}_j-\mathbf{x}_i\|,w_{ij}].
$$

`w_ij` must be unambiguously defined per experiment (for example, a normalized stiffness-related weight), never an unexplained mixed geometry/material number.

Graph updates:

$$
m_{ij}^{(\ell)}=\phi_\theta(h_i^{(\ell)},h_j^{(\ell)},e_{ij}),
\qquad a_i^{(\ell)}=\sum_{j\in\mathcal N(i)}m_{ij}^{(\ell)},
$$

$$
h_i^{(\ell+1)}=h_i^{(\ell)}+\psi_\theta(h_i^{(\ell)},a_i^{(\ell)}).
$$

**Key unresolved design problem:** the depth necessary to model globally coupled elliptic physics. Treat multiscale graph links, coarse-grid operators, or neural multigrid ideas as **experimental alternatives**, not requirements for the first GNN.

### 4.2 Abstraction boundaries

- PDE description must not assume a rectangular image grid.
- Mesh holds triangles, orientation, boundary markers and coordinates.
- GraphData can be regenerated from Mesh2D and physical inputs.
- Numerical assembly may run on CPU even when GNN training runs on GPU; transfer policy should be deliberate and benchmarked.
- FEM ground truth and neural predictions remain separate fields in artifacts.
- Viewer never has to import PyTorch or execute a differentiable FEM solve.
- Future CUDA/Triton operators must fit a reference operator contract without changing the science APIs.

---

## 5. Data interfaces and storage

| Concept | Required information | Likely shapes |
|---|---|---|
| `PDEProblem` | PDE type, coefficient descriptions, forcing, boundary data, parameters and units | Configuration + samples |
| `Mesh2D` | `xy`, triangle connectivity, boundary nodes/tags, geometry metadata | Coordinates `[N,2]`; triangles `[T,3]` |
| `GraphData` | Node features, edges, edge features, link to mesh | Nodes `[N,F]`; directed edges `[2,E]` |
| `SolutionField` | Mesh, values, field kind/source, time, units | Scalar `[N]` or frames `[Nt,N]` |
| `ExperimentManifest` | Git revision, config, random seeds, environment, wall time, metrics | JSON/YAML metadata |

**Storage recommendation:**

- Use compressed `.npz` for initial local solutions and plots; never require pickle to load basic data.
- Use appendable/chunked HDF5 or another documented layout for large cloud datasets if necessary.
- Save mesh once where possible; avoid duplicating connectivity in every record.
- Include schema versions, mesh identifiers, normalization parameters, and floating-point precision.
- Checkpoints need model, optimizer, scheduler (if used), epoch/step and RNG states.
- Use VTK/VTU export when broad scientific visualization interoperability becomes valuable.
- Keep separate folders for development test cases, training instances and a small local visualization gallery.

**Dataset split policy:** Split by PDE instance (coefficient field/forcing/boundary combination), preserve independent seeds, then evaluate held-out resolutions and geometries separately. Never make claims about generalization from random node splits of a single solved mesh.

---

## 6. Hardware and development environments

### 6.1 Confirmed resources and constraints

- **Permanent local workstation:** Arch Linux, 8 GB system RAM, Ryzen 5 CPU (exact physical core/thread count to check via `lscpu`), AMD Radeon RX 6500M with 4 GB VRAM.
- **Intermittent access:** NVIDIA RTX 3050 with 6 GB VRAM; do not rely on it for the main research schedule.
- **Available free GPU services:** Kaggle and Google Colab, subject to account quotas, GPU availability, session restrictions and changes in their policies.
- **Editors:** VS Code and LazyVim; **shell:** Fish; **coding assistant:** Codex CLI.

AMD's official ROCm support matrices do not list the RX 6500M as a supported PyTorch ROCm GPU in the checked documentation [R15]. Use local **CPU PyTorch** for debugging and keep Radeon use focused on graphics/visualization, unless a compatible setup is independently proven.

### 6.2 Final division of labor

| Location | Preferred work |
|---|---|
| Local Arch CPU | Architecture, coding, tests, manufactured solutions, small FEM runs, small model shape checks, plot generation |
| Local Radeon | Desktop/WebGL viewer rendering; no assumption of supported ROCm training |
| **Kaggle free GPU — primary cloud** | Main PINN/GNN training, large mesh datasets, first GPU profiling, eventual custom CUDA experiments |
| **Colab free GPU — backup** | Portability tests, additional experiments when available; not a guaranteed accelerator |
| RTX 3050 when accessible | Optional performance portability and CUDA parity measurements; never the required default |

Kaggle has documented T4×2 and P100 options [R13]; **each T4 has its own VRAM**. Two 16 GB GPUs do *not* automatically provide a single 32 GB address space for one graph. Accelerator choices and quota are account/session dependent, so verify the actual assignment in each run. Colab states that free GPU availability and limits vary [R14].

### 6.3 Portable execution design

The repository is the **single source of truth**. Cloud notebooks should only bootstrap dependencies, mount/copy data, invoke the same scripts, save checkpoints and export artifacts. Do not create separate Kaggle-only training algorithms.

Proposed profiles:

```text
configs/environments/
  local_cpu.yaml           # CPU reference, small graphs
  kaggle_t4.yaml           # NVIDIA GPU, larger batches/graphs
  colab_cuda.yaml          # GPU autodetection and safe defaults
```

These YAML files are **planned**, not in the currently shipped Phase 0–1 ZIP.

Cloud runs must log selected accelerator model, compute capability, total GPU memory, installed driver/runtime/PyTorch versions, CPU RAM and dataset revision.

### 6.4 Initial conservative ML budget

| Parameter | Start | Only after profiling |
|---|---:|---:|
| Nodes per graph | 1,000–3,000 | 5,000–20,000+ |
| Hidden width | 32 | 64+ |
| Message passing blocks | 3 | 4–8 / multiscale |
| Batch size | 1 graph | Increase if measured safe |
| Numeric precision | FP32 | Selective mixed precision after physics accuracy checks |
| Gradient checkpointing | Off initially | Benchmark memory/time trade-off |
| CUDA custom kernels | Not used | Phase 6 only |

These numbers are **starting configurations, not promises of memory fit**. Do not store hundreds of large graph instances and sparse matrices in the laptop's 8 GB RAM; generate on demand, stream data or use cloud preprocessing.

---

## 7. Repository and script maps

### 7.1 Actual shipped Phase 0–1 repository

The first starter archive is `physgraph-phase01.zip`, and includes **46 entries** (including package markers and documentation). Its important contents are:

```text
physgraph/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── .gitignore
├── configs/experiments/
│   ├── poisson_local.json
│   └── poisson_cloud.json
├── src/physgraph/
│   ├── core/types.py
│   ├── physics/manufactured.py
│   ├── mesh/structured.py
│   ├── fem/poisson.py
│   ├── fem/error.py
│   ├── experiments/pipeline.py
│   ├── io/results.py
│   ├── visualization/plotting.py
│   ├── graph/README.md             # reserved
│   ├── models/README.md            # reserved
│   ├── losses/README.md            # reserved
│   ├── training/README.md          # reserved
│   ├── profiling/README.md         # reserved
│   └── cuda/README.md              # reserved
├── scripts/
│   ├── run_phase1.py
│   ├── mesh_convergence.py
│   └── cloud_smoke.py
├── tests/
│   ├── test_mesh.py
│   ├── test_poisson.py
│   ├── test_io.py
│   └── test_cli.py
├── notebooks/cloud/phase1_cloud.ipynb
├── docs/
│   ├── architecture.md
│   ├── data_contract.md
│   ├── roadmap.md
│   └── verification.md
├── research/reading_order.md
├── data/
├── outputs/
└── viewer/README.md                # reserved
```

### 7.2 Planned phase extensions (do not confuse with existing files)

**Implementation update:** The numerical Phase 2A subset now exists, with
independent PDE/reference contracts, scalar diffusion/reaction, mixed boundaries,
graded meshes, quadrature, mass/reduced operators, a CLI and manifests. The exact
current file map and limits are in `docs/phase2.md`. The tree below remains the
broader plan; neural, generalized meshing and GPU entries are still future work.

```text
src/physgraph/
├── physics/
│   ├── base.py             # interface for future PDE families
│   ├── diffusion.py        # variable k(x), reaction c(x)
│   └── boundary.py         # Dirichlet / Neumann specifications
├── mesh/
│   ├── unstructured.py     # Gmsh triangles and boundary tags
│   └── quality.py          # aspect ratio / Jacobian checks
├── fem/
│   ├── assembly.py         # generalized stiffness/load/mass
│   └── operators.py        # sparse residual/energy contracts
├── graph/
│   ├── builder.py          # Mesh2D -> GraphData
│   ├── features.py         # geometric and physical features
│   └── adjacency.py        # COO/CSR conversion
├── models/
│   ├── pinn.py             # coordinate-based MLP baseline
│   ├── encoder.py
│   ├── processor.py
│   ├── decoder.py
│   └── meshgnn.py          # assembled model
├── losses/
│   ├── physics.py
│   ├── energy.py
│   ├── supervised.py
│   └── combined.py
├── training/
│   ├── dataset.py
│   ├── trainer.py
│   └── checkpoint.py
├── evaluation/
│   ├── metrics.py
│   ├── convergence.py
│   └── generalization.py
├── profiling/
│   ├── latency.py
│   ├── memory.py
│   └── operators.py
├── visualization/
│   ├── comparisons.py
│   ├── animation.py
│   └── export.py
└── cuda/                   # Phase 6, only after profiling
```

Scripts will remain **entry points** rather than duplicating package logic:

| Planned script | Role | Earliest phase |
|---|---|---|
| `scripts/data/generate_cases.py` | Sample forcing, coefficients, boundary instances | 2 |
| `scripts/solve/run_fem.py` | General PDE-family FEM baseline | 2 |
| `scripts/train/train_pinn.py` | Strong-form PINN baseline | 2 |
| `scripts/train/train_gnn.py` | Supervised/physics-informed training | 3 |
| `scripts/evaluate/evaluate_model.py` | Metrics over test problem families | 3 |
| `scripts/evaluate/mesh_convergence.py` | Refinement/cross-resolution evaluation | 3–4 |
| `scripts/profile/profile_solver.py` | End-to-end runtime/memory | 4 |
| `scripts/profile/profile_graph.py` | Graph-layer bottleneck inspection | 4 |
| `scripts/visualize/export_animation.py` | Frame sequence and playback export | 5 |

The viewer will eventually use its own `viewer/` frontend, **after** validating stable exported datasets and initial Matplotlib triangulated fields.

---

## 8. Experimental design and metrics

### 8.1 Main experiment ladder

| Experiment | Comparison / objective | Proof required |
|---|---|---|
| **E0** — FEM exact verification | Analytical solution vs P1 FEM | Boundary correctness; L2/H1 convergence and sparse-system residual |
| **E1** — conventional PINN | MLP with coordinate-based PDE residual | Reproducible optimization, error, training cost |
| **E2** — supervised MeshGraphNet | Learn nodal solutions from FEM-generated labels | Held-out forcing, coefficients and boundary cases |
| **E3** — physics-only MeshGraphNet | Physics residual or variational energy with no solution labels | Accuracy and stability across unseen instances |
| **E4** — mixed supervision | Physics + limited FEM labels | Label efficiency and ablation versus E2/E3 |
| **E5** — mesh generalization | Train small, evaluate refined/irregular | Controlled accuracy vs mesh size and geometry |
| **E6** — profile and optimize | Reference PyG/sparse ops vs chosen custom kernel | Same outputs/gradients, GPU peak memory and end-to-end speed |

### 8.2 Dataset policy

Start with the unit square. Generate smooth forcing as low-frequency randomized Fourier sums, then add heterogeneous positive diffusivity fields and varying Dirichlet values. Add L-shaped and perforated domains only when the geometry/mesh tooling and FEM checks are solid.

**Initial planning size:** ~200 cases for smoke evaluation, ~1,000+ cases for broader studies (scale according to cost). Train/validation/test split of **70/15/15 by whole PDE instance**, plus designated holdouts for resolution and geometry. Record seeds and avoid overlap in parameter draws that would trivialize generalization.

### 8.3 Accuracy and physics metrics

- Relative continuous `L2` error estimated by element quadrature, preferably against analytic truth where available.
- `H1` seminorm/energy-norm error when reliable gradients are available.
- Maximum nodal error, and boundary error (should be zero up to numerical precision for hard-imposed Dirichlet values).
- Algebraic discrete residual with documented normalization; include a note that it is not continuum truncation error.
- Absolute error and signed error maps on identical meshes.
- Cross-resolution and out-of-distribution performance, clearly identified as such.

### 8.4 Performance metrics

- Dataset generation time, baseline FEM assembly time, FEM solve time, GNN training time and GNN inference time **reported separately**.
- Cold and warm inference; include data-transfer cost where applicable.
- Peak allocated and reserved GPU memory; CPU RAM and graph storage.
- Forward and backward operator timings with warmup/synchronization and repeated trials.
- Device hardware, driver/toolkit, precision, batch size, graph nodes/edges/degree distribution.
- Amortized break-even discussion: training may be expensive even when neural inference is fast.

### 8.5 Baseline fairness

Use the same held-out PDE instances, boundary data and meshes where possible. A neural model should not be compared to a deliberately unoptimized FEM implementation if claiming practical speed superiority. Do not subtract offline preprocessing costs silently from one method but not another.

---

## 9. Two-dimensional visualization design

### 9.1 Aesthetic and purpose

Create an exploratory 2D simulator with live-feeling feedback similar to Sebastian Lague's visual demonstrations: show the quantity being solved, geometry, derived physical quantities and numerical errors. It is fundamentally a **scientific instrument**, not only a presentation layer.

### 9.2 Required first-viewer modes

1. **Scalar field:** colored solution over triangular elements; consistent color limits across reference and prediction.
2. **Mesh inspection:** wireframe, boundary tags, interior/boundary nodes, bad element highlighting.
3. **Comparison:** exact / FEM / PINN / GNN (as available), absolute and signed error panels.
4. **Derived fields:** gradients and flux vectors (`-k∇u`); later velocities and streamlines for appropriate PDEs.
5. **Geometry inspection:** square, nonuniform triangles, L-shape, hole; no interpolation through empty/removed regions.
6. **Simulation controls (later):** play/pause, time slider, source/parameter changes, mouse probe, pan/zoom, image/video export.

### 9.3 Development path

- **Phase 1:** Matplotlib triangulation output as independent truth-checking view. This is already in the starter repository.
- **Phase 2–4:** comparison plotting, error heatmaps and saved frames from model and FEM output.
- **Phase 5:** React + TypeScript and Canvas/WebGL, reading exported mesh and field data. The browser does not need to run FEniCSx or PyTorch. Avoid requiring Three.js for strictly 2D images unless it materially helps render performance.
- **Time-varying PDEs:** export time-stamped arrays and metadata; support deterministic playback before attempting live learned simulation.

### 9.4 Export contract

The viewer consumes `xy`, `triangles`, `fields` (with names and units), optional time values, solver metadata, color-scale suggestions and domain masks. For fair comparisons it must share geometry, coordinate conventions, colormap, and range across panels.

---

## 10. Deferred CUDA strategy

**No custom CUDA kernels until Phase 4 has objectively identified the bottleneck and the reference implementation is correct.** The optimization must never redefine the physical model or become a precondition for CPU tests.

### 10.1 Likely candidates, not commitments

- Graph gather–message–scatter intermediates.
- Sparse weighted aggregation using CSR.
- Redundant graph feature gathering and memory writes.
- Physics residual sparse multiplication if it emerges as a larger measured bottleneck.
- Memory footprint of saved activations during backpropagation.

A candidate reference operator is

$$
y_i = \sum_{j\in\mathcal N(i)}w_{ij}h_j.
$$

Start with **forward and backward correctness** against torch/PyG. A linear weighted aggregation is considerably easier to fuse than a nonlinear edge-MLP message function, so don't promise the latter in the first kernel.

### 10.2 Memory-traffic motivation

For `E` directed edges and hidden width `H`, an FP32 materialized edge-message tensor costs roughly

$$
4EH\text{ bytes}.
$$

For `E=600,000` and `H=64`, this is about **146.5 MiB** *per such tensor*, excluding graph indices, multiple intermediate copies, gradients, optimizer states, temporary buffers and runtime overhead. Actual peak memory must be **measured**, not inferred from this expression.

### 10.3 Baseline comparisons

1. Naive-but-correct gather/scatter implementation.
2. PyTorch sparse matrix multiplication where mathematical form permits.
3. PyG's existing `message_and_aggregate` / optimized sparse path where applicable [R9].
4. The new custom implementation (Triton prototype and/or CUDA C++).

Pass/fail conditions:

- Numerically equivalent outputs and gradients across many degree distributions and irregular meshes, with documented tolerances.
- No loss of physical accuracy or convergence from floating-point reordering.
- Repeated measured memory and end-to-end timing improvements against the **best appropriate baseline**, not merely one microbenchmark.
- CUDA GPU model, compute capability and profiling restrictions reported.

Cloud runtimes may prohibit low-level performance-counter profiling; if so, use available PyTorch/CUDA timers and memory statistics, while explicitly documenting the profiling limitation.

---

## 11. Phased roadmap and acceptance gates

These are **evidence gates**, not promised calendar deadlines.

### Phase 0 — Environment, repository and contracts

**Implemented scaffold:** Python source package, JSON configs, test harness, cloud notebook, documentation, Codex `AGENTS.md`.

**Acceptance:** package installs in a virtual environment; reproducible scripts and clean module boundaries; consistent mesh/field data formats.

### Phase 1 — Verified Poisson FEM baseline

**Initial implementation exists:** unit-square P1 FEM, manufactured exact solutions, nonzero Dirichlet tests, sparse solve, plots, export, mesh convergence.

**Acceptance:** tests pass locally, error decreases with refinement, an affine solution is reproduced at near-machine precision, and result artifacts/plots can be reloaded. Re-run and independently verify before freezing the baseline.

### Phase 2 — General elliptic physics and conventional PINN

**Phase 2A implemented:** scalar variable diffusion/reaction, full Dirichlet or
mixed Dirichlet/Neumann data, nonuniform triangular meshes, independently
specified analytical references, reusable reduced/mass operators and manifests.
Manufactured cases and numerical tests are described in `docs/phase2.md`.
Pure Neumann/nullspace handling and Gmsh imports are outside this increment.

**Geometry extension implemented:** a radial/angular annulus with a periodic seam,
tagged inner/outer edges, analytical logarithmic diffusion, both mixed-boundary
orientations, and geometry/solution convergence diagnostics. Tagged field archives
use schema v2, retaining v1 compatibility. See `docs/annulus.md`. The rectangle containing a cylinder is now implemented with tagged boundaries,
affine/quadratic manufactured fields and geometry/solution convergence (see
`docs/cylinder.md`). A closed symmetric NACA-style airfoil exterior is also
implemented, with a circular far field, upper/lower surface tags, rotation,
affine/quadratic references, and separate profile/far-field geometry errors
(see `docs/airfoil.md`). These are scalar elliptic verification tasks before any
separate flow-PDE extension. External FEniCSx checks passed for the fixture set;
actual cloud execution remains open. See `docs/validation_20261026.md`.

**Implementation update:** all 24 FEniCSx comparisons and the integration suite
passed in the installed `.fenicsx-env`. CPU PyTorch 2.10.0 and second derivatives
are verified. The user authorized beginning Phase 2B alongside this work. A small
NumPy tanh PINN now supports square Dirichlet manufactured cases, analytical
spatial derivatives, complex-step parameter gradients, Adam, deterministic resume
and separate exact/FEM/PINN evaluation. PyTorch/autodiff and broader neural scope
remain pending as implementation work. CPU PyTorch is installed in the optional
environment; no accelerator dependencies were introduced. See
`docs/fenicsx.md` and `docs/pinn.md`.

**Acceptance:** independent manufactured solutions and numerical convergence; PINN training and metrics reproducible; source-vs-FEM/PINN plots; boundary constraints documented.

### Phase 3 — Physics-informed MeshGraphNet

**Not implemented.** Build graph conversion, encoder–processor–decoder, supervised and physics-only training, optional hybrid loss.

**Acceptance:** graph topology checks; forward/backward gradient tests; train/resume checkpoints; held-out PDE-instance evaluation; meaningful ablations between supervised and physics-informed approaches.

### Phase 4 — Research-grade verification and profiling

**Not implemented.** Add geometry/resolution transfer, performance tables, GPU profiles and reproducible experiment manifests.

**Acceptance:** independently repeatable accuracy and performance results; clear bottleneck identification; no cherry-picked best-case comparison; separate reporting of solver time, training and inference.

### Phase 5 — Interactive 2D viewer

**Not implemented.** Promote verified plotting outputs into a standalone interactive field explorer.

**Acceptance:** correct rendering on irregular domains, consistent color scales, correct probes, comparison panels, playback from saved frames, export and basic usability.

### Phase 6 — Custom CUDA/Triton kernel

**Deferred.** Only implement operators shown to matter by Phase 4 profiling.

**Acceptance:** reference parity forward/backward, improvements in useful workloads and peak memory, multi-run benchmarking and documented architecture restrictions.

### Phase 7 — Optional publication package

After successful ablations: polished experimental tables, reproducibility manifest, precise novelty comparison with recent literature, and stable figure-generation scripts.

---

## 12. Current implementation and how to run it

### 12.1 What the starter ZIP actually contains

**Current checkout update:** The following paragraphs describe the historical
Phase 0–1 archive. The checkout now also includes Phase 2A; run
`python scripts/run_phase2.py --config configs/experiments/elliptic_local.json`.
Independent higher-order error integration now replaces the original error
quadrature in both phases, so the historical error table below is not the
current metric baseline. Original Poisson assembly remains unchanged.

The supplied Phase 0–1 archive is a **numerical FEM starter**, **not a functioning PINN/GNN model**. It can solve the 2D Dirichlet Poisson problem on structured triangulated unit squares, evaluate manufactured cases, compute errors, save portable fields and make PNG comparisons. It includes a Kaggle/Colab smoke notebook but does not require a GPU.

In the prior development run, the starter was reported as passing 11 automated tests, with a mesh-refinement study of the sine solution. The following are the previously recorded results; **reproduce them in your own environment before treating them as authoritative**:

| Mesh subdivisions | Relative L2 error | Relative H1-seminorm error |
|---:|---:|---:|
| 8 × 8 | 0.040819 | 0.194448 |
| 16 × 16 | 0.010402 | 0.097935 |
| 32 × 32 | 0.002613 | 0.049057 |

The near-second-order L2 and near-first-order H1 convergence are consistent with a smooth solution and P1 FEM, under standard regularity and quadrature assumptions. The exact continuum error estimates are numerical quadrature approximations.

### 12.2 Local setup (Arch + Fish)

From the extracted `physgraph/` repository root:

```fish
python -m venv .venv
source .venv/bin/activate.fish
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/run_phase1.py --config configs/experiments/poisson_local.json
python scripts/mesh_convergence.py --sizes 8 16 32
```

Expected generated outputs include field NPZ files, JSON metrics, `comparison.png`, and convergence CSV/PNG in `outputs/`.

### 12.3 Cloud smoke run

Use the packaged `notebooks/cloud/phase1_cloud.ipynb` or unpack the repository and invoke the same tests/scripts in Kaggle/Colab. The first cloud smoke test should run **without a GPU**. Later notebooks will install GPU PyTorch and call the identical package APIs.

### 12.4 First development sequence after this document

1. Run Phase 1 locally and capture exact dependency versions and test results.
2. Inspect the Poisson numerical modules and add an independent check against known solutions.
3. Freeze `Mesh2D`, `PDEProblem` and `SolutionField` APIs before expanding the equation family.
4. Phase 2A generalized elliptic FEM and manufactured tests are implemented;
   the FEniCSx gate is now closed for the tested fixtures (see `docs/validation_20261026.md`).
5. Train a conventional strong-form PINN baseline (Phase 2B).
6. Add mesh→graph conversion, GNN and physics-informed training (Phase 3).
7. Profile and verify **before** writing any custom GPU kernel.

---

## 13. Risks, research hypotheses, and open decisions

### 13.1 Primary risks

| Risk | Mitigation |
|---|---|
| Learned model overfits one PDE instance | Train across a family of forcing/coefficient/boundary inputs; hold out complete instances |
| Poor mesh-refinement generalization | Test at explicitly unseen resolutions; consider multiscale/normalization alternatives |
| Poor conditioning of physics residual | Compare residual vs energy objectives; normalization, operator scaling and preconditioning ablations |
| FEM is mathematically wrong but neural training still decreases loss | Manufactured solutions, element tests, positive-definiteness and independent FEM cross-check |
| GPU memory exhaustion | Begin with small graphs; stream cases; measure activations; use checkpointing only if supported by data |
| Free cloud session termination | Persist checkpoint+config+RNG state to durable storage; resume cleanly |
| GPU benchmark cannot be reproduced | Record hardware and environment, multiple runs and warmups; report variance |
| Custom kernel duplicates optimized existing operator | Benchmark PyG's best sparse/fused path first; quantify incremental benefit |
| Data-to-viewer geometry mismatch | One versioned mesh schema, triangle-aware interpolation and exact domain masks |
| False novelty claim | Perform updated prior-art search and report actual ablation evidence |

### 13.2 Hypotheses to test, not assert

- Weak-form physics improves learning under limited FEM labels or mesh shifts.
- A mesh GNN provides more transferable inductive bias than a coordinate-only MLP for irregular geometries.
- Sparse/fused execution can meaningfully reduce training memory enough to fit larger problems on constrained GPU hardware.
- Neural inference becomes competitive with repeated FEM solves **only after enough repeated related queries** to amortize training and dataset generation.

### 13.3 Decisions still open

1. **Phase 2 PDE generalization:** stay with hand-coded SciPy P1 assembly as primary and use FEniCSx as independent cross-check (recommended), or switch to FEniCSx as the main engine?
2. **Unstructured meshing:** when to add Gmsh, and whether to support holes/L-shaped domains before or after supervised GNN convergence.
3. **First graph operator:** a simple weighted linear sparse aggregation for baseline/kernel clarity, or a richer nonlinear edge-MLP MeshGraphNet from the beginning.
4. **Physics objective:** residual-only, energy-only, or a controlled staged curriculum; the answer should come from ablations.
5. **Artifact persistence:** versioned Kaggle datasets, another approved storage service, or disciplined manual exports; test the chosen path early.
6. **Viewer stack:** native Matplotlib/PyVista for all initial verification, then React + Canvas/WebGL once schemas stabilize (recommended).
7. **Accelerator availability:** verify Kaggle/Colab account quotas and GPU permissions at experiment launch rather than pinning plans to a permanently available T4.

---

## 14. Reference links

The bibliography below provides **stable primary-source URLs** where possible, plus relevant implementation/vendor docs. Paper-specific findings above are summaries; consult the originals before citing experimental performance in a publication.

- **[R1]** Pfaff, T. et al. (2021). *Learning Mesh-Based Simulation with Graph Networks*. ICLR. <https://arxiv.org/abs/2010.03409>. Full-text HTML: <https://arxiv.org/html/2010.03409v4>. Code: <https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets>.
- **[R2]** Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. *Journal of Computational Physics*. <https://doi.org/10.1016/j.jcp.2018.10.045>.
- **[R3]** Gao, H., Zahr, M. J., Wang, J.-X. (2022). *Physics-informed graph neural Galerkin networks: A unified framework for solving PDE-governed forward and inverse problems*. *Computer Methods in Applied Mechanics and Engineering*. <https://doi.org/10.1016/j.cma.2021.114502>. Preprint: <https://arxiv.org/abs/2107.12146>.
- **[R4]** Würth, T. et al. (2024). *Physics-informed MeshGraphNets (PI-MGNs): Neural finite element solvers for non-stationary and nonlinear simulations on arbitrary meshes*. *CMAME*. <https://doi.org/10.1016/j.cma.2024.117102>. Preprint: <https://arxiv.org/abs/2402.10681>.
- **[R5]** Chenaud, M., Magoulès, F., Alves, J. (2024). *Physics-Informed Graph-Mesh Networks for PDEs: A hybrid approach for complex problems*. *Advances in Engineering Software*. <https://doi.org/10.1016/j.advengsoft.2024.103758>.
- **[R6]** Li, Z. et al. (2020). *Multipole Graph Neural Operator for Parametric Partial Differential Equations*. NeurIPS. <https://arxiv.org/abs/2006.09535>.
- **[R7]** Kovachki, N. et al. (2023). *Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs*. *Journal of Machine Learning Research*. <https://arxiv.org/abs/2108.08481>.
- **[R8]** *Combining physics-informed graph neural network and finite difference for solving forward and inverse spatiotemporal PDEs*. <https://arxiv.org/abs/2405.20000>.
- **[R9]** PyTorch Geometric. *Memory-Efficient Aggregations* and `MessagePassing` documentation. <https://pytorch-geometric.readthedocs.io/en/latest/advanced/sparse_tensor.html> and <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html>.
- **[R10]** Wang, Y. et al. (2020). *GNNAdvisor: An Adaptive and Efficient Runtime System for GNN Acceleration on GPUs*. <https://arxiv.org/abs/2006.06608>.
- **[R11]** FEniCSx official documentation. <https://docs.fenicsproject.org/>.
- **[R12]** *A finite element-based physics-informed operator learning framework for spatiotemporal partial differential equations on arbitrary domains*. <https://link.springer.com/article/10.1007/s00366-024-02033-8>.
- **[R13]** Kaggle Notebooks official technical documentation. <https://www.kaggle.com/docs/notebooks>. NVIDIA hosted Kaggle GPU setup: <https://docs.nvidia.com/datascience/deployment/latest/platforms/kaggle/>.
- **[R14]** Google Colab official FAQ, GPU access and resource limits. <https://research.google.com/colaboratory/faq.html>.
- **[R15]** AMD ROCm supported Radeon/Ryzen GPUs. <https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/>.

---

### Project operating rules (one-page summary)

- **Scientific validity > raw model complexity > kernel speed.**
- **Local CPU development and reproducible cloud GPU training share the exact same solver package.**
- **Every numerical equation must have a manufactured or independent reference test.**
- **Train across PDE instances; evaluate on unseen cases, meshes and eventually geometries.**
- **Maintain distinct exact, FEM and model output fields in every plot and dataset.**
- **Do not implement CUDA until the ordinary solver is designed, verified and profiled.**
- **Custom acceleration must be judged by forward/backward parity and end-to-end useful speed/memory improvements.**
- **Update this document whenever an architecture decision or acceptance gate changes.**

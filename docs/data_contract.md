# Data contracts v1 and v2

`Mesh2D`: `nodes` float64 [N,2], `triangles` int64 [T,3], `boundary` bool [N].
The first phase uses a positively oriented unit-square mesh in dimensionless
coordinates. Triangle order is significant for basis gradients.
Phase 2A retains this contract for nonuniform meshes. `boundary` marks all
geometric exterior nodes; it does not identify the Dirichlet subset. Integer
boundary masks and noninteger connectivity are rejected.
An optional pair `boundary_edges` int64 [E,2] and `boundary_edge_tags` Unicode [E]
names every exterior edge exactly once. Both arrays must be supplied together.
Tags describe geometric groups, independently of the physical boundary condition.

`ManufacturedCase`: vectorized `exact(points)->values`, `forcing(points)->values`,
`gradient(points)->vectors`; `points` shape [M,2]. PDE: `-Δu=f`.

`FieldResult`: mesh-aligned `solution` and `exact` float64 [N]; scalar metrics.
Both fields and all stored error metrics must be finite. `equation` and
`boundary_type` identify Poisson versus generalized elliptic results; existing
v1 files default to the original Poisson/Dirichlet interpretation when absent.

`field.npz`: includes `nodes`, `triangles`, `boundary`, `solution`, `exact`
and JSON `metadata` with `schema_version: 1`. Loading uses
`numpy.load(..., allow_pickle=False)`. The viewer may consume this through a
future server/converter, but must never assume a regular grid layout.

Tagged meshes use `schema_version: 2` and also store `boundary_edges` and
`boundary_edge_tags`. v2 loading validates shapes, names and complete exterior
coverage. v1 archives load unchanged, with both optional arrays set to None;
untagged meshes continue to be written as v1. A loaded v1 mesh can be migrated
by assigning a complete edge/tag pair and saving it, producing a v2 archive.

`EllipticProblem`: scalar diffusion, reaction and forcing callables, plus
`BoundaryConditions` with Dirichlet data and optional edge selector/Neumann
function. `AnalyticalReference`: exact scalar field and vector gradient; used
only for evaluation. `ReducedSystem`: CSR matrix, effective RHS, free/prescribed
indices, prescribed values, and full-field lifting. See `phase2.md` for signs
and boundary partition semantics. Field archives do not yet serialize callable
PDE definitions; manufactured cases are identified by case name and source digest.
`dirichlet_tags` optionally selects boundary groups instead of a midpoint
selector; supplying both is rejected. Annulus geometry and physical parameters
are recorded in the accompanying experiment manifest. See `annulus.md`.

Later schemas will add `geometry_id`, `pde_family`, parameter fields, edge
attributes, time index, units, and labels. Breaking schema changes require a
version bump and migration support; do not silently overwrite v1 definitions.

The initial PINN uses a separate `comparison.npz` artifact, identified by
`artifact: pinn_comparison`, schema version 1. It stores nodes, triangles,
boundary, exact, fem_solution and pinn_prediction under distinct keys. It is
not a `FieldResult` archive. `fem_reference.npz` continues to use the existing
field schema and always contains the CPU FEM solution. Neural L2/H1 metrics
refer to direct network/gradient evaluation at quadrature points, not nodal
interpolation. Checkpoint schema 1 is also distinct and contains parameter,
optimizer, collocation, RNG and training configuration state; see `pinn.md`.

Independent comparison NPZ files use `physgraph_fem`, `fenicsx_fem` and `exact`
on shared nodes/triangles. Their accompanying manifests identify cases, external
versions and parity results. A blocked comparison produces no external solution
arrays; see `fenicsx.md`.

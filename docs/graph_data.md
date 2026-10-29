# Graph-data foundations

[Executed generation and validation](validation_annulus_graph.md).

The graph pipeline includes CPU NumPy graph construction, FEM-labelled annulus
and parameterized-square datasets, validated loading, normalization and batching.
It introduces no graph-learning dependency and no trained GNN. Its purpose is to
establish data contracts and numerical entry gates before model development.

## Mesh graph

`graph/mesh.py` builds a directed graph from triangle connectivity. Each unique
mesh edge appears twice, once in each direction. There are no self loops and no
distance-based edges crossing holes. Arc order is deterministic for a fixed node
numbering; physical features transform consistently under node relabeling.

Arrays include nodes, edge_index [2,E], target-minus-source displacement [E,2],
length, lumped nodal area, geometric boundary mask, edge boundary flags/tags and
outward normals. Interior-edge normals are zero. Both graph directions on an
exterior facet have the **same physical outward normal**, while displacement
reverses sign. Inner-hole normals point into the hole. Mesh topology, orientation,
complete boundary coverage and positive area/length are checked.

`physical_features` samples diffusion/reaction/forcing at nodes, with columns
in that order. Dirichlet masks and values come from the physical edge partition;
values are zero outside prescribed nodes. Neumann flags and outward derivative
q=k grad(u).n are sampled at actual polygon-facet midpoints, using actual normals.
Those samples are physical inputs, not the integrated FEM load vector.

No analytical interior values or solved fields enter these input features.
Generated archives contain a separate `fem_solution` target. They contain no
neural predictions and do not use the analytical reference as a supervised label.

## Instance identity and splitting

`datasets/instances.py` hashes canonical family, geometry and physical parameters.
Numeric aliases such as 1 and 1.0 or signed zero have the same identity. Key order
does not matter. Parameters must be finite JSON-compatible values. Resolution,
node numbering, labels, filenames and split names are excluded from identity.
The annulus generator resolves defaults before computing identity and rejects
duplicate physical instances.

Split membership hashes the identity and split seed into fixed train/validation/
test intervals (0.70/0.15/0.15). Reordering or appending instances does not alter
existing assignments. Every resolution of an instance remains in the same split.
Small datasets can have an empty split; manifests explicitly record counts and
`splits_nonempty`. Model training must require all needed splits to be nonempty.

This split controls exact physical-identity overlap. It does not establish
out-of-distribution separation or eliminate near-duplicate parameter settings;
geometry/coefficient extrapolation needs separately declared split policies.

## Generator and archive

```bash
python scripts/generate_graph_dataset.py \
  --config configs/experiments/annulus_graph_dataset.json \
  --output outputs/my_annulus_graph_dataset
```

The supplied example varies radii, boundary values, diffusivity and boundary
mode across 16 physical instances, each meshed at two resolutions (32 samples).
Schema-1 `fem_graph_sample` NPZ files load with `allow_pickle=False`. Each holds
graph arrays, physical input arrays, triangle connectivity, FEM labels and JSON
identity/split metadata. The dataset manifest records every resolution, mesh size,
FEM residual/error, instance parameters, split assignment and source provenance.
Existing output directories must be empty; partial generation is marked failed.

These radial harmonic cases are a small contract-validation dataset. Constant
diffusivity does not change the Dirichlet harmonic solution, so changing it alone
does not create a new solution shape. Broad GNN learning claims require richer
forcing/coefficient families, dedicated held-out-instance evaluation and tests
for geometry/resolution shifts. A supervised GNN remains next. Loading, normalization,
batching and the richer square family are now implemented as described below.

## Validated loading and preprocessing

The generator now writes **dataset manifest schema 2**, adding a SHA256 checksum
for each archive. The existing sample-array schema remains version 1. Legacy
manifests without checksums are rejected for training use with a regeneration
message; existing experiment directories are preserved. Regenerate from the
original config into a new directory rather than silently relabeling old files.

`datasets/loading.GraphDataset` requires completed generation, canonical physical
identities, complete instance/resolution coverage, correct deterministic splits,
safe local paths and matching archive checksums. On access it reconstructs the
declared mesh and physical inputs and compares every array, including geometry,
triangle orientation, edge symmetry, outward normals and boundary data. Targets
must be finite float64 values and satisfy prescribed Dirichlet values. Unexpected
arrays, object/pickle payloads and inconsistent metadata are rejected. This
loader supports the canonical annulus/square generator recipes; it is not a
general importer for arbitrary meshes or arbitrary node permutations.

Checksums detect file changes relative to a manifest, not authenticity. The
numerical readiness audit separately reassembles equations and recomputes errors
to catch incorrect interior labels even if their checksums have been updated.

`datasets/normalization.py` fits coefficient, prescribed-value and Neumann-flux
means/scales using **training instances at declared training resolutions only**.
Instances have equal weight, resolutions within an instance have equal weight,
and coefficients use lumped nodal area weights. Dirichlet statistics use only
prescribed nodes; flux statistics use only Neumann arcs with length weights.
Boundary statistics average only over instances with that boundary support.
Absent flux support uses mean zero/scale one; constant channels use scale one.
Missing BC values remain zero after transformation and carry explicit masks.
Geometry, areas, normals, masks and FEM targets retain their physical values.
No solved field is used to fit input statistics. Saved fits record dataset and
training-archive checksums, training identities, resolutions and feature columns.

Node feature columns: x, y, normalized k/c/f, normalized prescribed value,
Dirichlet mask, geometric boundary mask, lumped area. Edge columns: dx, dy,
length, outward normal x/y, normalized q, Neumann mask, geometric boundary mask.
Geometric tag strings remain available in samples; model inputs use the physical
boundary masks and normals, not a fitted categorical vocabulary.

`graph/batching.py` concatenates disjoint meshes, offsets edge/triangle indices,
and retains graph/edge/triangle pointers, raw boundary values and separate FEM
targets. A batch cannot mix splits or duplicate a sample. Epoch shuffling uses a
private seeded NumPy RNG, preserves the last partial batch, and defaults to the
normalizer's training resolutions. Shuffling reserved evaluation resolutions is
rejected. No graph-learning framework or new runtime dependency is required.

`evaluation/graph.py` evaluates nodal predictions with exact P1 mass-integrated
L2 error and element-gradient H1 error against FEM labels. It also reports
prescribed-boundary error and the free-node residual of the **unmodified full
prediction**. It does not repair boundary violations during evaluation. These
are differences between P1 fields, distinct from continuous PINN errors.

## Richer manufactured square family

On the unit square, define

\[
u=b_0+b_xx+b_yy+A\sin(\pi x)\sin(\pi y)
  +B\sin(2\pi x)\sin(\pi y),\qquad
k=k_0+k_xx+k_yy,\quad c=c_0.
\]

Require k0>0 and kx, ky, c0>=0. The PDE is -div(k grad u)+c u=f, where

\[
f=k\pi^2[2A\sin(\pi x)+5B\sin(2\pi x)]\sin(\pi y)
  -(k_x,k_y)\cdot\nabla u+c_0u.
\]

All-Dirichlet cases prescribe the affine boundary trace; mixed cases prescribe
that trace on x=0, y=0 and y=1, and q=k grad(u).n on x=1. Dirichlet edge endpoints
remain prescribed at mixed corners. The physical callbacks and analytical field
are separate from FEM assembly. Complex-step tests check gradient/forcing,
an affine patch checks nonzero boundary elimination, and independent UFL fixtures
derive forcing and flux symbolically for both boundary modes.

`configs/experiments/parametric_square_graph_dataset.json` declares 64 instances
with two modes, affine boundary variation, spatially varying diffusion, reaction
and two Fourier amplitudes. Each has resolutions 8 and 16: 128 FEM-labelled graphs.
This is a controlled manufactured family, not evidence of arbitrary-PDE or
out-of-distribution generalization. The constant coefficient changes in the
annulus dataset remain a separate, simpler geometry test.

```bash
python scripts/generate_graph_dataset.py \
  --config configs/experiments/parametric_square_graph_dataset.json \
  --output outputs/square_graph
python scripts/verify_graph_family.py --output outputs/square_family_verification
python scripts/check_graph_readiness.py \
  --dataset outputs/square_graph/dataset_manifest.json \
  --config configs/experiments/graph_readiness.json \
  --output outputs/square_readiness
```

Use fresh output directories. Readiness exits nonzero on incomplete data,
inadequate split counts, label errors, residual failures or invalid preprocessing.
It also checks label residuals after increasing assembly quadrature order by two.
The audit saves normalization and a manifest without constructing or training a
model. See the [training/evaluation protocol](gnn_protocol.md) and
[executed evidence](validation_pre_gnn.md).

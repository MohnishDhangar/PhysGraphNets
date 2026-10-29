# Local video and fluid repository review

Reviewed 29 October 2026. This is a source and timestamped-frame review, not a
qualification of the external solvers. No external application or its test suite
was run. No numerical implementation or dependency was changed.

## Evidence and scope

Six supplied MP4s were inspected using 12 broad samples and nine closely spaced
samples each: 126 frames, presented in 12 contact sheets. This establishes visual
features and short motion sequences; it is not continuous playback or a measurement
of the original application's runtime performance.

Local evidence is under `outputs/reference_review_20261029/`:

- `video_manifest.json`: filenames, SHA-256 hashes, codec dimensions, encoded frame
  rates, durations, requested sample times, Python/FFmpeg versions and local revision.
- `video_0/` through `video_5/`: extracted frames, `overview.jpg` and
  `detail_sequence.jpg`. Seek times are requested times, not exact presentation timestamps.
- `repository_manifest.json`: complete repository tree inventories, pinned revisions,
  selected source paths and local hashes. Source copies are in `repos/`.

All three repository trees were inventoried. The MATLAB script was read in full;
Kutta's fluid, particle, rendering, timing, geometry and relevant test paths were
reviewed; Spectrometry's wind-tunnel engine was reviewed, including its solver,
rendering, scene configuration and diagnostic paths. Its unrelated chemistry and
other engines were outside this fluid-viewer audit. The source collection contains
46 files; it is not a claim that every file in every repository was audited.

| Repository | Pinned revision | Most useful contribution |
| --- | --- | --- |
| [MATLAB 2D Fluid Simulation](https://github.com/robert-trirop/MATLAB-2D-Fluid-Simulation/tree/7ae9a2079579f6df1139035145026da3d2288a28) | `7ae9a2079579` | Continuous tracer injection and fine jet visualization |
| [Kutta](https://github.com/crgimenes/kutta/tree/5586d7dacbfd42783df5b228f2444dcb26b700da) | `5586d7dacbfd` | Interactive wind-tunnel controls, batched particles, geometry and animation |
| [Spectrometry](https://github.com/ec175/spectrometry_public/tree/f17b5a6ade723cddf5314d77dbf67b1f98229f45) | `f17b5a6ade72` | Python rendering, dense flow marks and streamed video export |

The reviewed code carries MIT licenses: MATLAB's notice is inside its script,
and the other two repositories have root license files. Preserve applicable
copyright/license notices and author attribution when adapting source. No source
has been incorporated into PhysGraph's runtime by this review.

## What the videos actually show

IDs below correspond to the local evidence directories, not YouTube IDs.

| ID / local clip | Observed configuration | Backend verdict | Viewer requirement |
| --- | --- | --- | --- |
| 0: Aero-Shifted, “ANSYS Fluent” | Three stacked airfoil views, colored fields and dense short trails; force arrows | Filename claims Fluent; source provenance remains unverified. Similar colors/arrows to Kutta are insufficient to identify a solver. | Independent field/particle layers; comparison panels |
| 1: Cesar Gimenes, Macarena wing | Main element and rotating flap, colored speed field, bright trails and force vectors; changing wake during rotation | Strong association with the supplied Kutta repository and its author; Kutta implements its own LBM solver | Geometry timeline, dense trails, field controls; physically consistent moving boundaries later |
| 2: Energy2D, laminar/turbulent flows | Four channels labelled with different velocities, heated squares, temperature plumes, viscosity controls and a streamline toggle | Energy2D interface is visible; its own fluid/thermal implementation | Temperature transport, optional streamlines and synchronized comparisons |
| 3: Robert S., MATLAB | Fine blue particles injected into a jet; coherent roll-up and increasingly dense filaments | Supplied MATLAB repository links the original video | Point/line emitters and small tracer marks; a field background is optional |
| 4: Sim.Lab.MP4, NACA 4412 | Three stacked flow-speed comparisons with purple/orange fields, white marks and wakes evolving at different rates | Supplied Spectrometry engine is relevant, but the exact NACA 4412 / 20–30–40 m/s preset was not found in the reviewed source | Independent runs, common physical clock, explicit units and shared color normalization |
| 5: Luke Morgan, PERRINN DRS | Brief 3D introduction, then a 2D rear-wing section; moving flap and velocity contours, a numerical legend and an inset time plot | OpenFOAM confirmed by the author | Moving-boundary reference data, contour playback and synchronized numerical probes |

Luke Morgan identifies OpenFOAM's `overPimpleDyMFoam` and an overset moving mesh.
His article explicitly describes a **2D isolated rear-wing study**, not a full-car
3D simulation. This corrects the earlier title-based assumption in our viewer plan.
The isolated-wing setup also limits comparisons to the complete car.
[Author's methodology and linked animation](https://luke.cfd/modelling-drs-activation-in-cfd/).

Energy2D documents its own semi-Lagrangian fluid solver and educational purpose;
the visible temperature plumes require thermal transport, rather than just colored
velocity or particle trails. The title alone does not establish a validated
turbulence model. [Energy2D documentation](https://energy.concord.org/energy2d/about.html).

**Verdict:** OpenFOAM explains the PERRINN reference. It is not the common engine
behind the three supplied repositories. Those implement their own numerical
methods. The remaining provenance uncertainties should stay explicit. Six local
clips do not establish that all eight original YouTube references were analyzed;
the earlier smoke-video reference, for example, is absent from these six files.

## Repository findings and reuse decisions

### MATLAB: useful emitter example, limited reference solver

The [complete script](https://github.com/robert-trirop/MATLAB-2D-Fluid-Simulation/blob/7ae9a2079579f6df1139035145026da3d2288a28/MATLAB_2D_Fluid_Simulation.m)
uses a 400 × 200 grid, pressure correction with 101 fixed Jacobi iterations,
backtraced velocity advection and RK4 tracer advection. It adds 66 particles per
iteration and renders their positions with small scatter marks.

Use the continuous emitter and fine-mark appearance. The filaments arise from
dense injected particles; they do not require drawing a long history for every
particle. Our existing particle integrator provides a better starting point than
porting the entire script.

Changes required before adopting its numerical approach:

- Replace the unbounded particle append with a bounded pool and explicit lifetime,
  outflow and invalid-position policies.
- Specify physical units, timestep, viscosity and boundary conditions. The compact
  example does not provide a calibrated viscous-flow configuration.
- Use a pressure residual/divergence acceptance criterion and verify discrete
  gradient/divergence compatibility; a fixed iteration count is insufficient evidence.
- Handle interpolation outside the grid and establish manufactured/refinement tests.

The encoded 60 fps of this video is not evidence that its solver ran at 60 fps.

### Kutta: strongest interaction reference

Kutta is a Go/Ebitengine application with browser WebAssembly support. Its
[`viz/particles.go`](https://github.com/crgimenes/kutta/blob/5586d7dacbfd42783df5b228f2444dcb26b700da/viz/particles.go)
exposes a small velocity-sampling/solid-query interface and a fixed seeded particle
pool. [`game.go`](https://github.com/crgimenes/kutta/blob/5586d7dacbfd42783df5b228f2444dcb26b700da/game.go)
batches 3,000 tracer marks, accumulates them in a fading image, and optionally adds
bloom. It also provides separate instantaneous streamlines. Scene, outline and
timing modules support custom geometry, multiple elements and prescribed animation.

Adapt the sampler interface, batched rendering, layer controls, geometry presets
and timeline concepts. Keep the Python numerical backend; adopting the whole Go
application would add a second application architecture without resolving our
mesh-based solver requirements.

Scientific and integration limitations found in the source:

- The [README](https://github.com/crgimenes/kutta/blob/5586d7dacbfd42783df5b228f2444dcb26b700da/README.md)
  explicitly describes qualitative aerodynamics. The D2Q9 BGK solver uses lattice
  units; displayed loads are not calibrated aerodynamic coefficients.
- Collision stability safeguards clamp velocity/density. Finite-state tests and
  force-sign tests are useful, but do not replace grid/time convergence or load
  benchmarks. Any adopted solver must record safeguard activation.
- Force integration uses pressure and excludes viscous drag.
- Animated geometry updates a solid mask and refills uncovered cells. The inspected
  update interface does not supply a moving-wall velocity; this is insufficient
  for a qualified moving-wing reference.
- Tracers use forward Euler and recycle at very low speed. That refreshes the
  display, but would bias residence-time or trapping measurements.
- Trail fading is applied per draw. Ours should use physical age or exponential
  decay based on elapsed time, so persistence does not change with refresh rate.
- SVG support has geometry restrictions, including ignored transforms and filled
  holes. Our arbitrary supported shapes must preserve topology and reject unsupported input.

### Spectrometry: strongest Python video reference

Focus on [`engines/wind-tunnel`](https://github.com/ec175/spectrometry_public/tree/f17b5a6ade723cddf5314d77dbf67b1f98229f45/engines/wind-tunnel).
The rendering pipeline separates scalar color mapping, tracer drawing, body
outlines and FFmpeg output. It supports a NumPy/SciPy/Pillow CPU path and optional
GPU paths. It contains both a D2Q9 BGK/Smagorinsky solver and a separate compressible
finite-volume solver using MUSCL/HLLC and SSP-RK2.

The immediately reusable ideas are:

- Subpixel particle rasterization, staggered lifetimes and smooth birth/death fading.
- Particle density scaled with output area; clean body outlines and layered fields.
- Interpolate scalar values before applying the color map; keep legends consistent.
- Stream RGB frames to FFmpeg instead of accumulating a whole movie in memory.
- Separate solver time from output frame rate; use one sampling/rendering contract
  across different numerical backends.

Important corrections and boundaries:

1. [`streaks.py`](https://github.com/ec175/spectrometry_public/blob/f17b5a6ade723cddf5314d77dbf67b1f98229f45/engines/wind-tunnel/src/wt/streaks.py)
   moves particle heads with midpoint substeps, but constructs tails by backward
   integration through the current frozen field. Those tails are short instantaneous
   streamline glyphs, not recorded unsteady pathlines or true emitter streaklines.
   Offer that inexpensive display mode under an accurate label; retain real
   timestamped histories for scientific particle tracking.
2. The triple-foil scenes place multiple objects in one coupled domain, sometimes
   with different inlet bands. They are not three independent wind-tunnel tests.
   Our parameter-comparison panels should load independent runs unless interaction
   is explicitly the experiment.
3. Some faster presets increase solver steps per video frame and change labels.
   This is playback scaling unless dimensional length/time/viscosity mapping is
   also established. Never infer 20/30/40 m/s from visual speed alone.
4. The `u0` parameter has different meanings between LBM and compressible backends.
   Our artifact must store units and solver identity explicitly.
5. The moving-wall implementation is more elaborate than Kutta's, but includes wall
   velocity clamps and distribution floors. Its health metric excludes the fluid
   cell layer adjoining bodies. Monitor near-wall errors, mass and momentum directly;
   a finite animation is not a validation result.
6. Dye populations are clipped nonnegative. If clipping activates it can change
   mass, so a conservative transport reference requires a measured mass budget.
7. Optional CuPy/NVENC selection and environment mutation should not be inherited
   into our strict CPU baseline. Rendering features can be adapted independently.
8. Diagnostic scripts include Sod and oblique-shock checks. Their presence is useful
   evidence of verification intent, but they were not executed in this review and
   do not qualify incompressible airfoil loads or arbitrary moving geometry.

## Rendering dependencies retained for the final phase

Subsequent user direction defers viewer/export development. The order below records
rendering dependencies, not the active development queue. The active numerical
sequence is in [the Stokes milestone](stokes.md) and [checklist](checklist.md).

| Step | Concrete deliverable | Acceptance evidence |
| --- | --- | --- |
| 1: artifact and sampler | Timestamped fields, mesh/boundaries, units, solver identity, velocity sampling and solid queries; periodic adapter first | Archive/sample agreement, coordinate checks, explicit error outside the domain |
| 2: smooth dense replay | V1/V2 viewer with 2,000 then 10,000 particles, true history trails, optional frozen-field glyphs, point/line emitters, layer toggles, scrub/speed/zoom | Trajectory reconstruction refinement, seam and boundary handling, deterministic seeds, bounded memory; measured frame times |
| 3: reproducible export | Shared rendering semantics for browser replay and CPU offline export; landscape/portrait layouts and independent comparison panels | Correct physical duration at 30/60 fps, consistent legends and timestamps, streamed 1080p MP4; encoding FPS reported separately from computation time |
| 4: fixed obstacle physics | Manufactured Stokes reference, then transient incompressible flow; channel, cylinder and airfoil presets | Boundary/pressure-gauge checks, divergence and flux balance, mesh/time refinement and independent force benchmarks |
| 5: moving wing physics | Prescribed flap motion with wall velocity and verified mesh-motion or immersed-boundary treatment | Moving-boundary manufactured/benchmark cases; conservation, force and timestep checks |
| 6: transported scalars and learning | Validated dye/temperature as separate fields; fluid datasets and surrogate rollout evaluation after reference qualification | Scalar mass/heat balances; held-out geometry/regime/rollout accuracy and reference fallback |

Scalar work can proceed independently on the periodic domain after smooth replay;
fluid physics and model qualification now take priority over presentation. The detailed
V1–V6 dependencies remain in [the viewer plan](viewer_plan.md).

An OpenFOAM artifact importer is a useful later option for externally computed
moving-wing references. Preserve mesh, field location, pressure convention, physical
times, units and turbulence-model metadata, and verify interpolation into the
viewer. Compare solvers only on matching equations, geometry, BCs and regimes.
Installing OpenFOAM or replacing our FEM foundation is not necessary to improve
the current viewer's particle density, smoothness or exports.

High-resolution rendering cannot recover unresolved wakes from a coarse solution.
Likewise, our qualified steady scalar GNN does not predict transient fluid velocity
and pressure. Build final viewer integration on qualified reference data and qualify a fluid
surrogate separately before offering neural playback as a predictive tool.

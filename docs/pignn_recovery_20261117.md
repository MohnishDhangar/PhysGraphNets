# PIGNN numerical recovery investigation — 17 November 2026

Historical outputs and the initial uncommitted source were preserved under
`outputs/pignn_recovery_20261117/initial_source.zip`. This work does not replace
the 16–17 November validation report. All accuracy gates remain unchanged.

## Hypothesis matrix, before new training

| Hypothesis | Discriminating experiment | Decision rule |
|---|---|---|
| Weak residual/Jacobian/flux/index defect | Independent assembly, Q4/Q7 comparison, physical-gradient P1 integration, gradcheck on graded and hole meshes | Fix a demonstrated discrepancy before architecture work |
| H1 evaluator defect | Independent element-loop errors versus saved matrix-form metrics | Require roundoff agreement, preserve original results |
| Geometry loss imbalance | Per-geometry normalized loss and parameter-gradient norms at saved checkpoints | Distinguish small objective from small relative H1 |
| Insufficient optimization | Same model, instances and seed, longer validation-only training | Budget ablation is not an architecture improvement |
| Missing useful local physical coupling | FEM-weighted residual messages with a fixed damped Jacobi update, compared at matched training budget | Include assembly and iteration costs; no label access |
| Underfitting versus parameter/mesh shift | Training versus validation metrics by geometry; resolution/quality diagnostics after selection | Do not call parameter shift geometry holdout |
| Hybrid imbalance | Separate component gradient norms and cosine disagreement; fixed versus training-gradient-balanced weights | No test-based adaptation or checkpoint selection |

Baseline aggregation pools seed-instance rows equally; standard deviation uses
population `ddof=0`. A pooled pass need not mean every seed passes, so new reports
will retain both pooled and per-seed decisions. The original study had exactly
one training and one validation instance per geometry and only 100 Adam epochs.
The validation square also changes grading from 1 to 1.3; this confounds a pure
parameter-shift interpretation. No final holdout will be used for tuning.

## Forensic evidence established before candidate selection

Baseline regression: 545 passed, 10 skipped, 143.35 seconds in Python 3.12.13 /
CPU Torch 2.10.0. The original uncommitted files and branch were preserved.
Checkpoint audit (`forensics/manifest.json`) covered 60 model/instance rows.
Independent element-loop L2/H1 values agree with saved metrics within 4.0e-15.
Weak residuals agree with SciPy assembly within 1.1e-14. Q7 reassembly of the
Q4 square reference gives residual norm 3.55e-5; hole-mesh discrepancies are
near roundoff. This is quadrature sensitivity, not an explanation for H1 near 1.

Mean training parameter-gradient norms at selected historical checkpoints were
.119 square, 2.903 annulus and .997 cylinder. The corresponding H1 means were
.809 / .174 / .639 and normalized physics losses .0147 / .174 / .0466. Thus
objective/gradient magnitude is not proportional to relative H1 error across
families. This confirms scale imbalance, without proving it is the only cause.

An explicit source-alias test constructs zero forcing and
`sin(3*pi*x)^2*sin(3*pi*y)^2` on the n=3 square. Their nodal graph inputs agree
to floating precision while their FEM solutions differ. Nodal samples alone
cannot represent every permitted sub-element forcing field. The original smooth
benchmark does not establish that this alias is its own failure cause.

The original evaluator also lacked denominator protection for zero reference
L2/H1 norms. This defect was fixed with the repository's existing norm floors
and absolute metrics; it does not change the nonzero original benchmark scores.

## Controlled numerical-message candidates

B1 applies four fixed damped Jacobi steps to the predicted field:
`u <- constrain(u - .5 D^-1 (A u-F))`, D=diag(A). These are physical FEM-weighted
local graph messages, with no new learnable parameters, labels or reference
solution in inference. For PSD three-node element matrices, `A <= 3 D`, making
this damping energy-stable on free DOFs. It does not guarantee uniform mesh
contraction. Local correction seed 0 improved square H1 from 1.027 to .0682
but worsened cylinder H1 from .509 to .773 at the matched 100-epoch budget.

B2 keeps four fine steps (two before/two after) and adds a coarse Galerkin
correction. Deterministic farthest-point coordinate aggregates supply a tentative
prolongation, smoothed by two homogeneous Jacobi steps with zero constrained
rows. The coarse operator is `P^T A P`; the correction is
`u <- u - P (P^T A P)^-1 P^T R`. At most eight coarse DOFs and at most roughly
one per eight free nodes are used. Coarse matrices are physical, fixed and cached.
The method is a neural prediction plus an iterative FEM solver, not pure neural
inference. Its numerical-only counterpart is reported explicitly.

The three-geometry B2 seed-0 smoke gave L2/H1 .00633/.0510, .000787/.00715 and
.01886/.2079 for square/annulus/cylinder. It retained zero boundary error. These
numbers motivated the five-seed ablation; they were not used as final tests.

Gradient diagnostics now include component norms, cosine disagreement, actual
effective weights, combined norm, clipping factor and nonfinite-event status.
Optional hybrid balancing equalizes detached weighted component gradient norms
using their geometric mean, with explicit bounds [.001,1000]. Validation
checkpoint selection keeps the fixed declared objective, independent of the
changing optimizer weights. Resume equivalence and gradcheck cover these paths.

One short test run triggered the source-change guard because a new configuration
was written while its experiment snapshot was active. The guard correctly failed
that run; the stable rerun passed 27 tests. No successful experiment with a
changed source identity is counted as valid evidence.

The locked holdout protocol was written before new five-seed selection to
`outputs/pignn_recovery_20261117/locked_protocol.json`, checksum
`543242149a90c0321a7114d7a8bde696fa3d214cf906d3c07ec3aedbff068685`.
It declares n=3/5 parameter, grading, radius, unseen-airfoil-family and mixed-BC
cases. It is separate from architecture/optimizer selection, and its results
must not be used for another tuning cycle.

## Five-seed architecture results

All four variants used seeds 0–4, the original three training and three validation
instances, n=3, width 16, depth 2, 100 Adam epochs at .003, FP64 and one CPU
thread. Only the specified communication/correction component changed. Neither
old baseline nor acceptance configurations were overwritten. The fresh baseline
reruns reproduced the original scores exactly. The study took 183.39 seconds
including evaluation/figures/I/O; all 20 runs completed without numerical failure.

| Candidate | Mean / max L2 | Mean / max H1 | Original pooled gates | Every seed passes |
|---|---|---|---|---|
| Original local | .082213 / .207702 | .633644 / 1.323404 | FAIL | No |
| Original global | .072275 / .149776 | .581289 / 1.044439 | FAIL | No |
| Global + four local FEM steps | .030737 / .084201 | .296286 / .782306 | FAIL | No |
| Global + four fine steps + coarse correction | .007023 / .018860 | .070796 / .207861 | PASS | Yes, 5/5 |

For the two-level physics-only candidate, per-geometry mean L2/H1 was
.006191/.049980 square, .001081/.010649 annulus and .013798/.151758 cylinder.
All boundary errors were exactly zero. Its full median/std/worst cases and
per-seed rows are in `architecture/manifest.json`. This is a successful restricted
validation result, not a statement that unseen geometry or continuum accuracy
has qualified. The original n=3 square FEM analytical H1 error is about .578.

Numerical-only controls matter. For seed 0, the two-level zero-initialized
numerical corrector alone obtained square/annulus/cylinder H1
.050826/.072226/.639932; the learned-initialization counterpart obtained
.051037/.007152/.207861. Thus the square benefit at this budget is already
explained by numerical correction, while the learned field helps substantially
for annulus and cylinder. No pure-neural contribution is claimed for the square
row. Full controls are retained in each seed's diagnostic manifest.

Mean training+export durations per seed were 2.696 / 2.713 / 4.810 / 3.851
seconds in table order. Mean cached-input forward durations across validation
graphs were 1.47 / 1.11 / 1.50 / 1.71 milliseconds. These are tiny workloads in
a warm process and include no mesh preparation. Mean preparation times were
.467 / .084 / .099 / .084 seconds; first-use overhead affects the first row.
Process RSS high-water reached 445624320 bytes, inherited across sequential
runs. The differences do not establish an end-to-end neural speed advantage.

The independent sparse FEM reference was timed separately during evaluation;
it is also included in artifacts. Dense energy-norm factorization during
training, graph preparation, coarse-space construction, label solves for
supervision, and plotting/checkpoint overhead must remain explicit in any future
amortized comparison. Neither a cached forward call nor these coarse fixtures
constitutes a fair overall cost benchmark against the conventional PINN or the
historically qualified square GNN, which have different training/scope contracts.

## Separate transient-reference evidence audit

The saved `outputs/cylinder_full_qualification_20261112/manifest.json` is more
complete than an interpretation of "missing three-level evidence" suggests.
It contains all three spatial levels 4/8/16 at dt=.005 and all three time steps
.02/.01/.005 on level 16, five trajectories total. Its recorded runtime was
15145.26 seconds (about 4.2 hours). The finest level-16 dt=.005 trajectory passed
the published-quantity candidate check and its lift peak-time error was .0121875
seconds. Thus it is incorrect to attribute that finest run's failure to a
lift peak time above .05 seconds.

The study itself remains FAILED because the refinement gates are not met.
For level 8→16, summary lift/pressure differences were .05503/.06327, normalized
lift/pressure history RMS .18973/.14635, and lift peak-time difference .045 s.
For dt .01→.005 on level 16, lift/pressure summary differences were .04875/.03471,
lift history RMS .14917 and lift peak-time difference .035 s. Peak timing passed
for these fine pairs; summary/history gates did not. Discrete, independent-solver,
surface/reaction and finest benchmark checks passed in the saved manifest.
Thresholds remain summary .03, peak times .05 s, history RMS .05. No reference
solver rewrite or four-hour rerun was performed in this scalar debugging work.
Further spatial/temporal phase-history convergence is needed, independently of
the PIGNN changes. This evidence does not qualify a learned fluid surrogate.

## Matched objective ablation and frozen selection

`pignn_recovery_objectives.json` holds architecture, fine/coarse steps, instances,
seeds and 100-epoch budget fixed. All 25 runs completed in 219.01 seconds,
including diagnostics and artifact I/O. Every variant passed every seed on the
restricted validation set. Statistics below pool 15 seed-instance rows with
population standard deviation, exactly as the original script did.

| Objective | Mean / median / std / max L2 | Mean / median / std / max H1 |
|---|---|---|
| Physics only | .007023 / .005316 / .005701 / .018860 | .070796 / .043128 / .063354 / .207861 |
| Supervised MSE | .005257 / .003454 / .004988 / .016070 | .056910 / .026896 / .058038 / .190731 |
| Supervised Sobolev | .005964 / .004180 / .005449 / .017804 | .061639 / .033353 / .061140 / .207340 |
| Fixed hybrid | .005957 / .003949 / .005953 / .018442 | .061622 / .031623 / .063934 / .209719 |
| Gradient-balanced hybrid | .006355 / .003125 / .005934 / .016673 | .066405 / .025930 / .066419 / .189378 |

All boundary errors remain exactly zero. MSE had the lowest mean H1 in this small
experiment; Sobolev was not automatically superior. Gradient balancing improved
hybrid worst H1 but worsened its mean. Hybrid component gradient cosine values
had min/median/max -.3448/.5889/.9795 for fixed weights and
-.2806/.7631/.9865 for balanced weights. Across 500 training epochs pooled over
seeds, respectively 27 and 11 had negative cosine. Clipping was disabled and
no nonfinite event occurred. These observations expose disagreement but do not
prove a universal optimizer recommendation.

Before any locked predictions, `selection_rule.json` and `frozen_selection.json`
declared physics-only as the primary label-free candidate, and selected one
hybrid comparator by smallest validation worst H1 among five-seed-passing
hybrids, breaking ties on mean H1. This selected `hybrid_balanced`.
No final-test selection or subsequent tuning is permitted. Matched supervised
variants remain baselines, not evidence that labels are required for qualification.

Baseline installation remains the existing `.fenicsx-env`; no dependency was
added. New code is in `models/fem_messages.py`, `evaluation/pignn.py`,
`experiments/pignn_forensics.py`, `experiments/pignn_holdout.py`, and
`visualization/pignn_diagnostics.py`, with focused changes to the existing
PIGNN trainer and experiment/study orchestration. Added tests cover energy
contraction, gradcheck, permutation, source aliasing, zero-reference metrics,
adaptive-hybrid resume, locked checksums and identity leakage rejection.

Diagnostic NPZ files retain the actual nodal fields and raw assembled residuals.
Figures show shared-range FEM/PIGNN fields, absolute field error, physical P1
gradient error and free-node residual maps; constrained reaction residuals are
excluded from the plotted equation residual. Each study retains all seeds and
worst-case figures. No learned field replaces a reference or analytical metric.

## Locked qualification outcome and limitations

Both frozen candidates completed 90 rows (five seeds, nine cases, n=3/5), using
only the recorded training normalization. Every seed passes the original gates
at n=3, but every seed fails at n=5. Overall broad qualification remains FAILED.

| Candidate / resolution | Mean / max L2 | Mean / max H1 | Boundary max | Result |
|---|---|---|---:|---|
| Physics / n=3 | .016908 / .067412 | .110960 / .336488 | 0 | PASS |
| Physics / n=5 | .053434 / .149765 | .362177 / .838674 | 0 | FAIL |
| Balanced hybrid / n=3 | .015763 / .067395 | .100958 / .319455 | 0 | PASS |
| Balanced hybrid / n=5 | .051937 / .154223 | .362106 / .841028 | 0 | FAIL |

Permutation checks on seed 0 across all declared cases/resolutions had maximum
differences 2.89e-15 / 3.11e-15. The physics/hybrid full locked evaluations took
13.66 / 14.18 seconds, with peak process RSS 420413440 / 413757440 bytes.
Airfoil-family, radius-change, mixed-BC, parameter and grading results remain
separate in [the generated numerical tables](pignn_recovery_results_20261117.md).
The now-inspected final set is spent; no model, step count or loss weight was
changed in response to these results.

Read-only stage diagnosis of seed 0 exactly reproduces the locked predictions.
For the parameter-shift cylinder case, initial raw H1 errors were 5.573 at n=3
and 8.476 at n=5. The same numerical correction reduced them to .233 and .790.
Squared-energy contraction weakened from .001706 to .008556 while the Jacobi
minimum eigenvalue fell .01155→.00406 and free/coarse DOFs changed 192/8→640/8.
For the square, contraction weakened .00288→.05539. The correction remains
energy-stable, but neither the learned initialization nor this capped coarse
space is resolution-independent. This supplies reproducible contributing failure
mechanisms without claiming that all feature and optimizer effects are resolved.

Final regression after adding the read-only diagnosis: **554 passed, 10 skipped
in 117.64 seconds**. The earlier post-change run also passed 554/10 in 163.15 s.
The ten skips are one deliberate missing-dependency path and nine explicit
real-GPU tests. The original mesh demo executed at n=8/16/32 with the unchanged
L2 errors .04221224/.01075142/.00270066 and H1
.19437753/.09792576/.04905618. A graded n=5 affine mixed-boundary diagnostic gave
Neumann flux L2 8.18e-16 and conservation defect 2.22e-16.

Only three pre-existing source files changed relative to the initial snapshot:
`training/pignn.py`, `experiments/pignn.py`, `experiments/pignn_study.py`.
The stable FEM, weak assembler, existing graph architectures and previous reports
remain intact. The separate `.github` modification predates this task and was
preserved. No dependency was added and no commit, push or output deletion occurred.
The final source archive is `outputs/pignn_recovery_20261117/final_source.zip`,
SHA-256 `569b8bbd93e2abbb3a392c0c3e9e6e8074a325cf9e686f31dbd5ee48f2e82834`;
source fingerprint `937a7168f5e960cb5eb62bf6f1811034d98e291383a8b2ac58dad7069981538a`.

## Next cycle, without reusing this holdout for selection

1. Grow training physical diversity and include multiple training resolutions,
   keeping every refinement of a physical identity in one split. Reserve fresh
   identities and mesh levels for the next locked test.
2. Use training/validation diagnostics to investigate a genuine multilevel
   hierarchy or residual-controlled iterative solve with explicitly reported
   iteration/assembly costs. A fixed eight-dimensional coarse space and four
   fine steps are not enough for uniform transfer.
3. Preserve the quadrature/operator input path to avoid the demonstrated nodal
   forcing alias. Study coefficient/source variability and arbitrary partitions
   before extending supported-domain claims.
4. Keep fixed/adaptive hybrid comparisons controlled; balancing did not improve
   every metric. Investigate per-geometry scaling without fitting test losses.
5. Benchmark cold setup, warm inference, repeat solves and all baseline scopes
   in isolated processes at comparable accuracy before any speed claim. GPU
   parity remains unexecuted on this CPU Torch installation.
6. Treat the independent cylinder refinement failure separately. Learned fluids,
   kernel fusion, viewer, deployment and publication benchmarks remain subsequent
   milestones; this scalar result does not satisfy those prerequisites.

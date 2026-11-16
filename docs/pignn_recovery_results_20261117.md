# PIGNN recovery numerical tables — 17 November 2026

Generated from saved manifests. All five seeds are retained. Standard deviations are population values over equally weighted seed-instance rows. No threshold changed.

Gates: relative FEM L2 mean/max <= .05/.10; H1 seminorm mean/max <= .20/.40; boundary max <= 1e-12.

## Architecture ablation

| Variant | L2 mean / median / std / max | H1 mean / median / std / max | All seeds |
|---|---|---|---|
| original_local | 0.082213 / 0.060539 / 0.062511 / 0.207702 | 0.633644 / 0.643843 / 0.398202 / 1.323404 | FAIL |
| original_global | 0.072275 / 0.054058 / 0.050745 / 0.149776 | 0.581289 / 0.525835 / 0.343693 / 1.044439 | FAIL |
| local_fem | 0.030737 / 0.011755 / 0.032023 / 0.084201 | 0.296286 / 0.074979 / 0.323191 / 0.782306 | FAIL |
| two_level_fem | 0.007023 / 0.005316 / 0.005701 / 0.018860 | 0.070796 / 0.043128 / 0.063354 / 0.207861 | PASS |

Elapsed including plots/I/O: 183.389 s. Source SHA-256 `9fc975548b3e35ba89bc3f9ed52609d0df062dbc372d002145fcceb35a2adbf1`.

| Variant / geometry | Mean L2 | Mean H1 | Max H1 |
|---|---:|---:|---:|
| original_local / square | 0.122579 | 0.905170 | 1.130709 |
| original_local / annulus | 0.019154 | 0.183107 | 0.383089 |
| original_local / cylinder | 0.104906 | 0.812656 | 1.323404 |
| original_global / square | 0.138688 | 0.981694 | 1.044439 |
| original_global / annulus | 0.022200 | 0.210795 | 0.595008 |
| original_global / cylinder | 0.055935 | 0.551376 | 0.783429 |
| local_fem / square | 0.010569 | 0.061519 | 0.074979 |
| local_fem / annulus | 0.006005 | 0.074714 | 0.078580 |
| local_fem / cylinder | 0.075636 | 0.752625 | 0.782306 |
| two_level_fem / square | 0.006191 | 0.049980 | 0.077970 |
| two_level_fem / annulus | 0.001081 | 0.010649 | 0.016839 |
| two_level_fem / cylinder | 0.013798 | 0.151758 | 0.207861 |

## Matched objective ablation

| Variant | L2 mean / median / std / max | H1 mean / median / std / max | All seeds |
|---|---|---|---|
| physics | 0.007023 / 0.005316 / 0.005701 / 0.018860 | 0.070796 / 0.043128 / 0.063354 / 0.207861 | PASS |
| supervised_mse | 0.005257 / 0.003454 / 0.004988 / 0.016070 | 0.056910 / 0.026896 / 0.058038 / 0.190731 | PASS |
| supervised_sobolev | 0.005964 / 0.004180 / 0.005449 / 0.017804 | 0.061639 / 0.033353 / 0.061140 / 0.207340 | PASS |
| hybrid_fixed | 0.005957 / 0.003949 / 0.005953 / 0.018442 | 0.061622 / 0.031623 / 0.063934 / 0.209719 | PASS |
| hybrid_balanced | 0.006355 / 0.003125 / 0.005934 / 0.016673 | 0.066405 / 0.025930 / 0.066419 / 0.189378 | PASS |

Elapsed including plots/I/O: 219.014 s. Source SHA-256 `d3e0ce2a9f2a515be8085c034a4abc57f665d8a0dd48921fe51b9f94612b1acc`.

| Variant / geometry | Mean L2 | Mean H1 | Max H1 |
|---|---:|---:|---:|
| physics / square | 0.006191 | 0.049980 | 0.077970 |
| physics / annulus | 0.001081 | 0.010649 | 0.016839 |
| physics / cylinder | 0.013798 | 0.151758 | 0.207861 |
| supervised_mse / square | 0.002446 | 0.019448 | 0.043431 |
| supervised_mse / annulus | 0.002149 | 0.025521 | 0.075768 |
| supervised_mse / cylinder | 0.011176 | 0.125760 | 0.190731 |
| supervised_sobolev / square | 0.003060 | 0.024331 | 0.044741 |
| supervised_sobolev / annulus | 0.002241 | 0.025777 | 0.074680 |
| supervised_sobolev / cylinder | 0.012591 | 0.134810 | 0.207340 |
| hybrid_fixed / square | 0.002669 | 0.021516 | 0.046083 |
| hybrid_fixed / annulus | 0.001636 | 0.020431 | 0.063983 |
| hybrid_fixed / cylinder | 0.013565 | 0.142917 | 0.209719 |
| hybrid_balanced / square | 0.003626 | 0.029332 | 0.046236 |
| hybrid_balanced / annulus | 0.001203 | 0.013425 | 0.021070 |
| hybrid_balanced / cylinder | 0.014236 | 0.156457 | 0.189378 |

## Locked physics-only evaluation

Overall: **FAILED**. Protocol SHA-256 `543242149a90c0321a7114d7a8bde696fa3d214cf906d3c07ec3aedbff068685`. No tuning followed this evaluation.

| Group | L2 mean / median / std / max | H1 mean / median / std / max | Result |
|---|---|---|---|
| all | 0.035171 / 0.025828 / 0.031594 / 0.149765 | 0.236568 / 0.198400 / 0.191065 / 0.838674 | FAIL |
| n=3 | 0.016908 / 0.010296 / 0.018708 / 0.067412 | 0.110960 / 0.073522 / 0.098037 / 0.336488 | PASS |
| n=5 | 0.053434 / 0.044019 / 0.031294 / 0.149765 | 0.362177 / 0.324379 / 0.178452 / 0.838674 | FAIL |
| geometry:airfoil | 0.080044 / 0.073000 / 0.018252 / 0.108724 | 0.351452 / 0.317399 / 0.063200 / 0.455121 | FAIL |
| geometry:annulus | 0.030314 / 0.017719 / 0.036274 / 0.149765 | 0.199034 / 0.151628 / 0.173046 / 0.602018 | FAIL |
| geometry:cylinder | 0.038207 / 0.035079 / 0.024949 / 0.081935 | 0.387367 / 0.390753 / 0.241011 / 0.838674 | FAIL |
| geometry:square | 0.023046 / 0.017263 / 0.017458 / 0.066631 | 0.135275 / 0.095294 / 0.091038 / 0.300454 | PASS |
| category:boundary_and_parameter_shift | 0.046988 / 0.032313 / 0.037612 / 0.149765 | 0.217574 / 0.154984 / 0.157900 / 0.602018 | FAIL |
| category:geometry_parameter_shift | 0.023225 / 0.015574 / 0.021118 / 0.067358 | 0.243655 / 0.155040 / 0.199513 / 0.616175 | FAIL |
| category:heldout_geometry_family | 0.080044 / 0.073000 / 0.018252 / 0.108724 | 0.351452 / 0.317399 / 0.063200 / 0.455121 | FAIL |
| category:mesh_quality_and_parameter | 0.017386 / 0.014685 / 0.012275 / 0.038356 | 0.135255 / 0.104856 / 0.092325 / 0.285738 | PASS |
| category:unseen_parameter | 0.026228 / 0.019947 / 0.023410 / 0.081935 | 0.239983 / 0.204709 / 0.232550 / 0.838674 | FAIL |

| Seed | n | Mean L2 | Max L2 | Mean H1 | Max H1 | Result |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 3 | 0.016290 | 0.067412 | 0.105753 | 0.319538 | PASS |
| 0 | 5 | 0.056728 | 0.106655 | 0.383276 | 0.790270 | FAIL |
| 1 | 3 | 0.018407 | 0.065662 | 0.131667 | 0.336488 | PASS |
| 1 | 5 | 0.070545 | 0.149765 | 0.468371 | 0.838674 | FAIL |
| 2 | 3 | 0.014649 | 0.061497 | 0.088052 | 0.291805 | PASS |
| 2 | 5 | 0.042921 | 0.082237 | 0.298187 | 0.485544 | FAIL |
| 3 | 3 | 0.016505 | 0.060436 | 0.106481 | 0.286640 | PASS |
| 3 | 5 | 0.046334 | 0.096602 | 0.307452 | 0.616175 | FAIL |
| 4 | 3 | 0.018690 | 0.066442 | 0.122846 | 0.314828 | PASS |
| 4 | 5 | 0.050641 | 0.105194 | 0.353597 | 0.742430 | FAIL |

All boundary errors: zero. Maximum tested permutation discrepancy: 2.887e-15. Elapsed 13.660 s; process peak RSS 420413440 bytes. Source SHA-256 `1ab16c45a68273c7fc1ade29b6a05b0950a44c01209b880c9c3bd6fde1bb238e`.

## Locked balanced-hybrid evaluation

Overall: **FAILED**. Protocol SHA-256 `543242149a90c0321a7114d7a8bde696fa3d214cf906d3c07ec3aedbff068685`. No tuning followed this evaluation.

| Group | L2 mean / median / std / max | H1 mean / median / std / max | Result |
|---|---|---|---|
| all | 0.033850 / 0.025393 / 0.030951 / 0.154223 | 0.231532 / 0.201249 / 0.188818 / 0.841028 | FAIL |
| n=3 | 0.015763 / 0.007102 / 0.018972 / 0.067395 | 0.100958 / 0.055632 / 0.095596 / 0.319455 | PASS |
| n=5 | 0.051937 / 0.040125 / 0.030028 / 0.154223 | 0.362106 / 0.337368 / 0.167532 / 0.841028 | FAIL |
| geometry:airfoil | 0.076642 / 0.069084 / 0.016251 / 0.103323 | 0.338425 / 0.315415 / 0.058039 / 0.436106 | FAIL |
| geometry:annulus | 0.030119 / 0.020356 / 0.035659 / 0.154223 | 0.197946 / 0.169840 / 0.170939 / 0.647589 | FAIL |
| geometry:cylinder | 0.035701 / 0.034432 / 0.024603 / 0.081759 | 0.362828 / 0.340808 / 0.239237 / 0.841028 | FAIL |
| geometry:square | 0.022083 / 0.018207 / 0.018594 / 0.063408 | 0.141957 / 0.117668 / 0.118343 / 0.454397 | FAIL |
| category:boundary_and_parameter_shift | 0.045069 / 0.032337 / 0.037291 / 0.154223 | 0.223845 / 0.194589 / 0.163430 / 0.647589 | FAIL |
| category:geometry_parameter_shift | 0.022008 / 0.012788 / 0.021195 / 0.077795 | 0.231132 / 0.135395 / 0.202899 / 0.716052 | FAIL |
| category:heldout_geometry_family | 0.076642 / 0.069084 / 0.016251 / 0.103323 | 0.338425 / 0.315415 / 0.058039 / 0.436106 | FAIL |
| category:mesh_quality_and_parameter | 0.020062 / 0.013108 / 0.018528 / 0.060235 | 0.153846 / 0.108338 / 0.138640 / 0.454397 | FAIL |
| category:unseen_parameter | 0.024598 / 0.020356 / 0.022542 / 0.081759 | 0.227188 / 0.184862 / 0.219765 / 0.841028 | FAIL |

| Seed | n | Mean L2 | Max L2 | Mean H1 | Max H1 | Result |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 3 | 0.017443 | 0.067395 | 0.114024 | 0.319455 | PASS |
| 0 | 5 | 0.054517 | 0.103323 | 0.378511 | 0.841028 | FAIL |
| 1 | 3 | 0.015864 | 0.064771 | 0.102934 | 0.306878 | PASS |
| 1 | 5 | 0.062662 | 0.154223 | 0.436083 | 0.647589 | FAIL |
| 2 | 3 | 0.013303 | 0.060336 | 0.080025 | 0.286368 | PASS |
| 2 | 5 | 0.048343 | 0.081172 | 0.334188 | 0.480548 | FAIL |
| 3 | 3 | 0.015016 | 0.058240 | 0.093617 | 0.276265 | PASS |
| 3 | 5 | 0.044108 | 0.082655 | 0.308398 | 0.716052 | FAIL |
| 4 | 3 | 0.017189 | 0.065725 | 0.114189 | 0.311374 | PASS |
| 4 | 5 | 0.050057 | 0.099938 | 0.353352 | 0.713985 | FAIL |

All boundary errors: zero. Maximum tested permutation discrepancy: 3.109e-15. Elapsed 14.177 s; process peak RSS 413757440 bytes. Source SHA-256 `1ab16c45a68273c7fc1ade29b6a05b0950a44c01209b880c9c3bd6fde1bb238e`.

## Frozen-stage transfer postmortem

Seed 0, no new iteration count, parameter update, or model selection. Stage replay agrees exactly with the locked predictions.

| Case | n | Free/coarse DOFs | Jacobi minimum eigenvalue | Raw/final H1 | Energy error squared contraction |
|---|---:|---|---:|---|---:|
| parameter_square | 3 | 4/1 | 0.504474 | 0.778170 / 0.042758 | 0.002878 |
| quality_square | 3 | 4/1 | 0.567918 | 0.908394 / 0.054738 | 0.003508 |
| parameter_annulus | 3 | 48/6 | 0.282374 | 1.830556 / 0.010166 | 0.000031 |
| geometry_annulus | 3 | 48/6 | 0.215073 | 2.023051 / 0.011779 | 0.000034 |
| parameter_cylinder | 3 | 192/8 | 0.011551 | 5.572526 / 0.233079 | 0.001706 |
| geometry_cylinder | 3 | 192/8 | 0.009434 | 4.682986 / 0.140868 | 0.000880 |
| heldout_airfoil | 3 | 48/6 | 0.087514 | 0.857011 / 0.319538 | 0.129326 |
| mixed_square | 3 | 6/1 | 0.316512 | 0.783801 / 0.053942 | 0.004577 |
| mixed_annulus | 3 | 72/8 | 0.081808 | 1.606324 / 0.084910 | 0.002794 |
| parameter_square | 5 | 16/2 | 0.193760 | 0.999565 / 0.241087 | 0.055391 |
| quality_square | 5 | 16/2 | 0.237207 | 1.716055 / 0.266037 | 0.025190 |
| parameter_annulus | 5 | 160/8 | 0.107313 | 5.679682 / 0.258806 | 0.002076 |
| geometry_annulus | 5 | 160/8 | 0.081105 | 6.611378 / 0.197384 | 0.000891 |
| parameter_cylinder | 5 | 640/8 | 0.004058 | 8.475893 / 0.790270 | 0.008556 |
| geometry_cylinder | 5 | 640/8 | 0.003231 | 7.615541 / 0.611804 | 0.006348 |
| heldout_airfoil | 5 | 160/8 | 0.028644 | 1.525998 / 0.448101 | 0.084066 |
| mixed_square | 5 | 20/2 | 0.120273 | 0.951645 / 0.255155 | 0.068914 |
| mixed_annulus | 5 | 200/8 | 0.031083 | 5.101255 / 0.380837 | 0.005573 |

The finer meshes both increase learned-initialization error and weaken contraction of the fixed numerical correction. For cylinder, free DOFs increase 192→640 while coarse DOFs stay eight; the Jacobi minimum eigenvalue drops .01155→.00406. The method is energy-stable but is not a demonstrated mesh-independent solver. These are observed contributing mechanisms, not a proof excluding all feature/optimization effects.

## Artifacts and reproduction

All outputs are under `outputs/pignn_recovery_20261117/`. Each training run has configuration, dependency/revision/source identity, normalization/input checksums, source ZIP, checkpoint, objective/gradient history, saved fields and diagnostic figures. Holdouts retain the locked protocol, every checkpoint checksum and seed/case/resolution row, reference/analytical errors, flux/conservation metrics and worst-case PNGs. The common hardware and timing-method record is `hardware.json`; no GPU experiment or speedup was claimed.

Use fresh outputs; never overwrite historical directories. Activate `.fenicsx-env` or prefix Python commands with `.fenicsx-env/bin/python`; set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/pignn-mpl-cache`.

```bash
python -m pytest -q
python scripts/mesh_convergence.py --sizes 8 16 32 --output outputs/recovery_replay/convergence
python scripts/audit_pignn.py --baseline outputs/pignn_development_20261116/final_communication_ablation --output outputs/recovery_replay/forensics
python scripts/run_pignn_study.py --config configs/experiments/pignn_recovery_architecture.json --output outputs/recovery_replay/architecture
python scripts/run_pignn_study.py --config configs/experiments/pignn_recovery_objectives.json --output outputs/recovery_replay/objectives
python scripts/evaluate_pignn_holdout.py --study outputs/recovery_replay/objectives --variant physics --protocol configs/experiments/pignn_recovery_locked_holdout.json --protocol-sha256 543242149a90c0321a7114d7a8bde696fa3d214cf906d3c07ec3aedbff068685 --output outputs/recovery_replay/locked_physics
python scripts/evaluate_pignn_holdout.py --study outputs/recovery_replay/objectives --variant hybrid_balanced --protocol configs/experiments/pignn_recovery_locked_holdout.json --protocol-sha256 543242149a90c0321a7114d7a8bde696fa3d214cf906d3c07ec3aedbff068685 --output outputs/recovery_replay/locked_hybrid
python scripts/diagnose_pignn_transfer.py --study outputs/recovery_replay/objectives --variant physics --protocol configs/experiments/pignn_recovery_locked_holdout.json --output outputs/recovery_replay/postmortem
```

These commands reproduce frozen decisions; they are not permission to tune against the now-inspected holdout. For exact old-source reproduction, overlay the per-run source ZIP on the recorded Git revision. Later diagnostic-only source revisions may change provenance while leaving model predictions unchanged.


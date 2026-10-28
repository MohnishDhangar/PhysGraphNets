# PhysGraph AI coding instructions

- The priority is mathematical correctness and reproducible evidence, not speed.
- **Do not add CUDA, Triton, GNN, or PINN dependencies without a phase decision.**
- Keep PDE definitions, finite-element assembly, learned models, plotting and CLI
  scripts in separate modules. CLI scripts must only orchestrate package code.
- Keep a strict local CPU baseline; cloud notebooks run the same Python package.
- For numerical changes, run `python -m pytest -q` and the mesh convergence demo.
- Check nonzero Dirichlet boundary conditions and orientation/area of triangles.
- Do not replace FEM reference values with neural predictions in validation.
- Never claim code passed tests that were not executed.
- Produce a short experiment manifest (Python version, case, mesh size, errors,
  code revision) for any future scientific experiment.
- Do not commit generated `outputs/`, large datasets, credentials or API tokens.
- New PDE families require a mathematical formulation plus manufactured cases.
- Future GPU work must report hardware, GPU VRAM, precision, timing methodology,
  and peak memory. Do not implement a custom GPU kernel until the verified
  reference implementation is profiled.

"""Train/evaluate a per-instance CPU PINN while preserving the FEM reference."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from physgraph.core.types import FieldResult
from physgraph.evaluation.pinn import evaluate_pinn, validate_evaluation
from physgraph.fem.elliptic import solve
from physgraph.fem.error import field_errors
from physgraph.io.manifest import write_manifest
from physgraph.io.results import save_field
from physgraph.models.pinn import TanhMLP
from physgraph.physics.pinn import get_pinn_case
from physgraph.training.pinn import TrainingConfig, train, load_checkpoint
from physgraph.visualization.pinn import plot_pinn_comparison, plot_training


def run_experiment(*, case='sine', width=8, depth=2, collocation=128, learning_rate=.003,
                   seed=7, steps=1000, nx=16, checkpoint_every=100, resume=None, output=None,
                   backend='numpy', torch_threads=1, validation_seed=10007, holdout_points=1024,
                   boundary_points=257, evaluation_quadrature_order=5, acceptance=None, neumann_points=32, neumann_weight=1.):
    started = perf_counter()
    validate_evaluation(evaluation_quadrature_order, validation_seed, holdout_points, boundary_points)
    from physgraph.evaluation.acceptance import validate_thresholds, assess
    validate_thresholds(acceptance)
    config = TrainingConfig(case, width, depth, collocation, learning_rate, seed, neumann_points, neumann_weight)
    if backend not in ('numpy', 'torch'):
        raise ValueError('backend must be numpy or torch')
    if isinstance(torch_threads, bool) or not isinstance(torch_threads, int) or torch_threads < 1:
        raise ValueError('torch_threads must be a positive integer')
    train_backend, load_backend, evaluate_backend = train, load_checkpoint, evaluate_pinn
    backend_options = {}
    backend_settings = {'backend': 'numpy_complex_step_adam', 'gradient_work_precision': 'complex128',
        'spatial_derivatives': 'analytical tanh chain rules', 'parameter_derivatives': 'complex-step 1e-30'}
    if backend == 'torch':
        try:
            from physgraph.core.torch_runtime import runtime_metadata
            from physgraph.training.torch_pinn import train as train_backend, load_checkpoint as load_backend
            from physgraph.evaluation.torch_pinn import evaluate_pinn as evaluate_backend
        except ImportError as exc:
            raise ValueError('PyTorch backend unavailable; use the CPU PyTorch environment described in docs/pinn.md') from exc
        backend_options = {'threads': torch_threads}
        backend_settings = {'backend': 'torch_autograd_adam', 'gradient_work_precision': 'float64',
            'spatial_derivatives': 'torch.autograd of full constrained field',
            'parameter_derivatives': 'torch.autograd backward', 'torch_runtime': runtime_metadata(torch_threads),
            'initialization_rng': 'NumPy default_rng; no Torch random operations'}
    manufactured = get_pinn_case(case)
    mesh = manufactured.domain.mesh(nx)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError('steps must be an integer >= 0')
    if isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int) or checkpoint_every < 1:
        raise ValueError('checkpoint_every must be a positive integer')
    resume_state = load_backend(resume, config, **backend_options) if resume is not None else None
    if resume_state is not None and steps < resume_state.step:
        raise ValueError('target steps precede checkpoint')
    dest = Path(output) if output is not None else Path('outputs/pinn') / (case+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError(f'output directory must be empty to preserve previous runs: {dest}')
    dest.mkdir(parents=True, exist_ok=True)
    settings = {**asdict(config), 'steps': steps, 'nx': nx, 'nodes': len(mesh.nodes), 'triangles': len(mesh.triangles),
        'checkpoint_every': checkpoint_every, 'resume': str(Path(resume).resolve()) if resume else None,
        'resume_sha256': hashlib.sha256(Path(resume).read_bytes()).hexdigest() if resume else None,
        'start_step': resume_state.step if resume_state else 0, **backend_settings,
        'parameter_precision': 'float64',
        'boundary_kind': get_pinn_case(case).boundary_kind,
        'boundary_enforcement': ('hard radial polynomial lifting/mask on exact circles' if manufactured.constraint is not None
                                 else 'hard affine lifting on Dirichlet edges; normalized flux loss on Neumann edge'),
        'evaluation_quadrature_order': evaluation_quadrature_order,
        'pinn_error_domain': 'true annulus' if manufactured.domain.name == 'annulus' else 'unit square',
        'fem_error_domain': 'polygonal mesh',
        'mesh_resolution': {'radial_cells': nx, 'angular_sectors': 8*nx} if manufactured.domain.name == 'annulus' else {'nx': nx, 'ny': nx},
        'validation_seed': validation_seed, 'holdout_points': holdout_points,
        'boundary_points': boundary_points, 'acceptance_thresholds': acceptance, 'fem_assembly_quadrature_order': 4,
        'cpu_count': os.cpu_count(), 'thread_environment': {key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        'output': str(dest.resolve()), 'scope': 'single-instance strong-form PINN', 'domain': asdict(manufactured.domain)}
    try:
        training_start = perf_counter()
        state = train_backend(config, steps, checkpoint=dest/'checkpoint.npz', resume=resume,
            checkpoint_every=checkpoint_every, progress=lambda step, value: print(f'Adam {step}: loss={value:.6e}', flush=True),
            **backend_options)
        training_seconds = perf_counter()-training_start
        model, manufactured = TanhMLP(width, depth), get_pinn_case(case)
        fem_start = perf_counter()
        reference = solve(mesh, manufactured.problem, 4)
        fem_seconds = perf_counter()-fem_start
        fem_metrics = field_errors(mesh, reference.values, manufactured.reference)
        prediction, pinn_metrics = evaluate_backend(mesh, model, state.parameters, manufactured,
            validation_seed=validation_seed, holdout_points=holdout_points, boundary_points=boundary_points,
            quadrature_order=evaluation_quadrature_order, **backend_options)
        exact = manufactured.reference.exact(mesh.nodes)
        field = FieldResult(mesh, reference.values, exact, case, reference.relative_residual,
                            fem_metrics['relative_l2'], fem_metrics['relative_h1_seminorm'], fem_metrics['max_nodal_error'],
                            '-div(k grad(u)) + c*u = f', manufactured.boundary_kind)
        save_field(dest/'fem_reference.npz', field)
        np.savez_compressed(dest/'comparison.npz', nodes=mesh.nodes, triangles=mesh.triangles,
            boundary=mesh.boundary, exact=exact, fem_solution=reference.values, pinn_prediction=prediction,
            metadata=np.array(json.dumps({'schema_version': 1, 'artifact': 'pinn_comparison', 'case': case,
                                         'backend': settings['backend'], 'domain': asdict(manufactured.domain)})),
            **({'boundary_edges': mesh.boundary_edges, 'boundary_edge_tags': mesh.boundary_edge_tags}
               if mesh.boundary_edges is not None else {}))
        np.savetxt(dest/'training.csv', np.column_stack((np.arange(len(state.losses)), state.losses)),
                   delimiter=',', header='step,normalized_residual_mse', comments='')
        plot_pinn_comparison(mesh, exact, reference.values, prediction, dest/'comparison.png')
        plot_training(state.losses, dest/'training.png')
        metrics = {'status': 'completed', 'acceptance': assess(pinn_metrics, acceptance), 'fem': {**fem_metrics, 'relative_residual': reference.relative_residual},
            'pinn': pinn_metrics, 'training': {'initial_loss': state.losses[0], 'final_loss': state.losses[-1],
            'steps': state.step, 'parameters': model.parameter_count, 'seconds_this_run': training_seconds},
            'fem_assembly_and_solve_seconds': fem_seconds}
        (dest/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
        return write_manifest(dest/'manifest.json', settings, metrics, perf_counter()-started)
    except Exception as exc:
        write_manifest(dest/'manifest.json', settings, {'status': 'failed', 'reason': str(exc)}, perf_counter()-started)
        raise

"""One isolated solver trial; JSON stdin/stdout, Linux peak RSS before evaluation."""
from dataclasses import asdict
import json
import resource
import sys
from time import perf_counter
import numpy as np
from physgraph.physics.pinn import get_pinn_case
from physgraph.fem.elliptic import solve
from physgraph.fem.error import field_errors


def trial(settings):
    case=get_pinn_case(settings['case'])
    baseline_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if settings['solver']=='fem':
        start=perf_counter();mesh=case.domain.mesh(settings['resolution'])
        setup=perf_counter()-start
        start=perf_counter();solution=solve(mesh,case.problem,4)
        solve_seconds=perf_counter()-start
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        errors=field_errors(mesh,solution.values,case.reference,quadrature_order=7)
        sensitivity=field_errors(mesh,solution.values,case.reference,quadrature_order=5)
        return {'mesh': {'nx':settings['resolution'],'nodes':len(mesh.nodes),'triangles':len(mesh.triangles)},
                'setup_seconds':setup,'solve_seconds':solve_seconds,'total_seconds':setup+solve_seconds,
                'errors':errors,'quadrature5_errors':sensitivity,'peak_rss_bytes_before_evaluation':peak_rss,
                'baseline_peak_rss_bytes':baseline_rss}
    from physgraph.core.torch_runtime import cpu_runtime
    from physgraph.training.pinn import TrainingConfig
    from physgraph.training.torch_pinn import train
    from physgraph.models.pinn import TanhMLP
    from physgraph.models.torch_pinn import TorchTanhMLP, constrained_value
    from physgraph.evaluation.torch_pinn import evaluate_pinn
    import torch
    cfg=TrainingConfig(case=settings['case'],**settings['training'])
    # Imported runtime memory is part of baseline, not an allocation delta claim.
    baseline_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    start=perf_counter();state=train(cfg,settings['steps'],threads=1)
    training=perf_counter()-start
    start=perf_counter();model=TorchTanhMLP(cfg.width,cfg.depth,state.parameters)
    restore=perf_counter()-start
    with cpu_runtime(1),torch.no_grad():
        points=torch.tensor(case.domain.sample(np.random.default_rng(20007),4096),dtype=torch.float64)
        constrained_value(model,points,case.lifting,case.boundary_kind,case.constraint) # untimed warmup
        times=[]
        for _ in range(20):
            start=perf_counter();constrained_value(model,points,case.lifting,case.boundary_kind,case.constraint)
            times.append(perf_counter()-start)
    peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    architecture=TanhMLP(cfg.width,cfg.depth)
    evaluations=[]
    for n,order in ((16,5),(32,5),(32,7)):
        mesh=case.domain.mesh(n)
        errors=evaluate_pinn(mesh,architecture,state.parameters,case,quadrature_order=order,
            validation_seed=20007,holdout_points=4096)[1]
        evaluations.append({'nx':n,'quadrature_order':order,'errors':errors})
    return {'training_config':asdict(cfg),'training_including_setup_seconds':training,
            'model_restore_seconds':restore,'total_seconds':training+restore,
            'inference_4096_points_seconds':times,'errors':evaluations[-1]['errors'],
            'evaluation_sensitivity':evaluations,'peak_rss_bytes_before_evaluation':peak_rss,
            'baseline_peak_rss_bytes':baseline_rss}


if __name__=='__main__':
    print(json.dumps(trial(json.load(sys.stdin)),allow_nan=False))

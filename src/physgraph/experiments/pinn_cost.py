"""Repeated, isolated CPU solver costs evaluated against common accuracy targets."""
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
from physgraph.io.manifest import write_manifest
from physgraph.training.pinn import TrainingConfig
from physgraph.physics.pinn import CASE_NAMES, get_pinn_case


def qualifies(errors, target):
    return all(np.isfinite(errors[k]) and errors[k] <= limit for k,limit in target.items())


def select_candidate(rows, target):
    """Require every repeat to meet both targets; never discard a failed repeat."""
    groups={}
    for row in rows:
        groups.setdefault(row['candidate'],[]).append(row)
    eligible=[]
    for candidate,trials in groups.items():
        if all(r['status']=='completed' and qualifies(r['result']['errors'],target) for r in trials):
            eligible.append({'candidate':candidate,'median_seconds':float(np.median([r['result']['total_seconds'] for r in trials]))})
    return min(eligible,key=lambda r:r['median_seconds']) if eligible else None


def run_cost_study(*, cases, resolutions, step_counts, repeats, target, output, training=None):
    if sys.platform != 'linux':
        raise ValueError('RSS units in this study are defined for Linux only')
    if not isinstance(repeats,int) or isinstance(repeats,bool) or repeats<2:
        raise ValueError('use at least two isolated repeats')
    if not isinstance(target,dict) or set(target)!={'relative_l2','relative_h1_seminorm'}:
        raise ValueError('declare common relative_l2 and relative_h1_seminorm targets')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for v in target.values()):
        raise ValueError('accuracy targets must be finite and positive')
    for values,name in ((cases,'cases'),(resolutions,'resolutions'),(step_counts,'step_counts')):
        if not isinstance(values,list) or not values or len(values)!=len(set(values)):
            raise ValueError(f'{name} must be a nonempty list without duplicates')
    if any(c not in CASE_NAMES for c in cases):
        raise ValueError('unsupported case')
    if any(isinstance(n,bool) or not isinstance(n,int) or n<2 for n in resolutions):
        raise ValueError('resolutions must be integers >=2')
    if any(isinstance(n,bool) or not isinstance(n,int) or n<0 for n in step_counts):
        raise ValueError('step_counts must be nonnegative integers')
    if any(get_pinn_case(c).domain.name != 'unit_square' for c in cases):
        raise ValueError('cost comparisons currently require square domains; curved geometry costs need separate acceptance rules')
    training=dict(training or {})
    for case in cases:
        TrainingConfig(case=case,**training)
    dest=Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('output directory must be empty')
    dest.mkdir(parents=True,exist_ok=True)
    config={'cases':cases,'resolutions':resolutions,'step_counts':step_counts,'repeats':repeats,
            'target':target,'training':training,'device':'cpu','torch_threads':1,
            'timing':'fresh process per trial; imports and error evaluation excluded; train includes optimizer setup; no checkpoint I/O',
            'memory':'Linux process high-water RSS, sampled before validation; includes imported libraries and inference; not incremental allocations',
            'selection':'all repeats must meet both error targets; cheapest median among supplied candidates only'}
    start=perf_counter();rows=[]
    def save(status):
        choices={case:{solver:select_candidate([r for r in rows if r['case']==case and r['solver']==solver],target)
                       for solver in ('fem','torch')} for case in cases}
        return write_manifest(dest/'cost_manifest.json',config,{'status':status,'runs':rows,'selected':choices},perf_counter()-start)
    save('running')
    for case in cases:
        for solver,candidates in (('fem',resolutions),('torch',step_counts)):
            for size in candidates:
                for repeat in range(repeats):
                    settings={'case':case,'solver':solver,'resolution':size,'steps':size,'training':training}
                    row={'case':case,'solver':solver,'candidate':f'{solver}_{size}','repeat':repeat}
                    try:
                        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
                        process=subprocess.run([sys.executable,'-m','physgraph.profiling.cost_worker'],input=json.dumps(settings),
                            capture_output=True,text=True,env=env,timeout=600,check=True)
                        row.update(status='completed',result=json.loads(process.stdout))
                    except (subprocess.SubprocessError,OSError,ValueError) as exc:
                        row.update(status='failed',reason=str(exc),stderr=getattr(exc,'stderr',None))
                    rows.append(row);save('running')
    return save('completed_with_failures' if any(r['status']=='failed' for r in rows) else 'completed')

"""Bounded trajectory fixtures from an immutable, completed CPU FEM archive."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from physgraph.profiling.flow_linear import array_frames,file_sha256
from physgraph.io.manifest import write_manifest


def prepare_flow_segments(*,trajectory,output,starts=(0,1200),steps=3):
    source,dest=Path(trajectory),Path(output)
    if type(steps) is not int or steps<1 or not starts or len(set(starts))!=len(starts):
        raise ValueError('require positive steps and unique starts')
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('seed output must be empty')
    start=perf_counter()
    with np.load(source,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
        times=archive['times']
        geometry={k:archive[k] for k in ('nodes','triangles','boundary_edges','boundary_edge_tags')}
    if metadata.get('viscous_form')!='laplacian' or metadata.get('scheme') not in ('bdf2','BDF2 with BE startup'):
        raise ValueError('requires archived Laplacian BDF2 cylinder trajectory')
    dt=float(metadata['dt'])
    if not np.array_equal(times,dt*np.arange(len(times))) or any(type(s) is not int or s<0 or s+steps>=len(times) for s in starts):
        raise ValueError('invalid seed steps or time grid')
    velocity_indices=sorted({i for s in starts for i in range(max(0,s-1),s+steps+1)})
    pressure_indices=sorted({i for s in starts for i in range(max(0,s-1),s+steps)})
    velocity=array_frames(source,'velocity',velocity_indices)
    pressure=array_frames(source,'pressure',pressure_indices)
    reactions=array_frames(source,'velocity_reactions',sorted({i for s in starts for i in range(s,s+steps)}))
    source_hash=file_sha256(source)
    dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for step in starts:
        label=f'start{step:05d}'
        meta=dict(case=metadata['case'],dt=dt,start_step=step,steps=steps,
            quadrature_order=metadata['quadrature_order'],trajectory_sha256=source_hash,
            archived_source_sha256=metadata.get('launch_source_sha256'),
            purpose='new backend comparison from archived BDF2 history; not resumed qualification or independent physical evidence')
        path=dest/(label+'.npz')
        np.savez(path,**geometry,metadata=np.array(json.dumps(meta)),
            previous=velocity[step],older=velocity[max(0,step-1)],
            previous_pressure=np.zeros(len(geometry['nodes'])) if step==0 else pressure[step-1],
            times=times[step:step+steps+1],
            expected_velocity=np.array([velocity[i] for i in range(step,step+steps+1)]),
            expected_pressure=np.array([pressure[i] for i in range(step,step+steps)]),
            expected_reactions=np.array([reactions[i] for i in range(step,step+steps)]))
        rows.append(dict(case=label,path=path.name,sha256=file_sha256(path),start_step=step,
                         start_time=float(times[step]),steps=steps))
    return write_manifest(dest/'manifest.json',dict(trajectory=str(source.resolve()),
        trajectory_sha256=source_hash,starts=list(starts),steps=steps),
        dict(status='prepared',cases=rows),perf_counter()-start)

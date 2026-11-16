"""Extract reproducible Newton systems from archived cylinder trajectories."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.sparse import save_npz
from physgraph.core.types import Mesh2D
from physgraph.fem.stokes import taylor_hood_space
from physgraph.fem.stokes_boundary import stokes_partition, stokes_velocity_values
from physgraph.fem.flow_operators import FlowOperators
from physgraph.fem.navier_stokes import _direct
from physgraph.physics.cylinder_flow import CylinderFlowCase
from physgraph.io.manifest import write_manifest


def array_frames(path, key, indices):
    """Read selected C-order float64 frames without allocating an entire history."""
    with zipfile.ZipFile(path) as archive, archive.open(key+'.npy') as stream:
        version = np.lib.format.read_magic(stream)
        reader = {(1,0):np.lib.format.read_array_header_1_0,
                  (2,0):np.lib.format.read_array_header_2_0}.get(version)
        if reader is None:
            raise ValueError('unsupported NPY header version')
        shape, fortran, dtype = reader(stream)
        if not shape or fortran or dtype != np.dtype('float64'):
            raise ValueError('fixture extraction requires C-order float64 frames')
        if any(type(i) is not int or not 0 <= i < shape[0] for i in indices):
            raise ValueError('frame index outside archive')
        offset, frame_size = stream.tell(), int(np.prod(shape[1:]))*dtype.itemsize
        values = {}
        for index in sorted(set(indices)):
            stream.seek(offset+index*frame_size)
            data = stream.read(frame_size)
            if len(data) != frame_size:
                raise ValueError('truncated frame')
            values[index] = np.frombuffer(data, dtype=dtype).reshape(shape[1:]).copy()
            if not np.isfinite(values[index]).all():
                raise ValueError('nonfinite archived frame')
    return values


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def prepare_systems(trajectory, output, steps=(1,1201)):
    """Capture first Newton corrections, not new independent physical evidence.

    The previous numerical states are read from a declared reference archive.
    This benchmarks algebraic backends only; it never labels them as independent
    PDE solutions or substitutes them for the original manufactured/FEM checks.
    """
    from time import perf_counter
    begun = perf_counter()
    path, destination = Path(trajectory), Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('fixture output must be empty')
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        times = archive['times']
        nodes, triangles = archive['nodes'], archive['triangles']
        edges, tags = archive['boundary_edges'], archive['boundary_edge_tags']
    if metadata.get('viscous_form') != 'laplacian' or metadata.get('scheme') not in ('bdf2', 'BDF2 with BE startup'):
        raise ValueError('requires declared Laplacian BDF2 cylinder archive')
    if not steps or len(set(steps)) != len(steps) or any(type(s) is not int or not 1 <= s < len(times) for s in steps):
        raise ValueError('steps must be distinct solved positive archive indices')
    dt = float(metadata['dt'])
    if not np.array_equal(times, dt*np.arange(len(times))):
        raise ValueError('requires fixed-step zero-origin time grid')
    case_data = dict(metadata['case'])
    case_data['center'] = tuple(case_data['center'])
    case = CylinderFlowCase(**case_data)
    boundary = np.zeros(len(nodes), dtype=bool)
    boundary[np.unique(edges)] = True
    mesh = Mesh2D(nodes, triangles, boundary, edges, tags)
    space = taylor_hood_space(mesh)
    prescribed, facets, tractions, selected = stokes_partition(space, case.dirichlet_tags, case.tractions)
    ops = FlowOperators(space, case.viscosity, metadata['quadrature_order'], 'laplacian')
    nv, npres = len(space.points), len(nodes)
    fixed = np.r_[prescribed, prescribed+nv]
    free = np.setdiff1d(np.arange(2*nv+npres), fixed)
    velocities = array_frames(path, 'velocity', [i for step in steps for i in (max(0,step-2),step-1)])
    pressures = array_frames(path, 'pressure', [step-2 for step in steps if step > 1])
    destination.mkdir(parents=True, exist_ok=True)
    cases = []
    archive_sha = file_sha256(path)
    for step in steps:
        time = float(times[step])
        previous = velocities[step-1].T.ravel()
        alpha = 1. if step == 1 else 1.5
        history = previous/dt if step == 1 else (2*previous-.5*velocities[step-2].T.ravel())/dt
        guess = np.r_[previous, np.zeros(npres) if step == 1 else pressures[step-2]]
        values = {tag:(lambda x,f=f:f(x,time)) for tag,f in case.boundary_velocity.items()}
        guess[fixed] = stokes_velocity_values(space,prescribed,facets,selected,values).T.ravel()
        linear = ops.saddle(alpha/dt*ops.mass+ops.viscous, False)
        body = ops.volume_load(lambda x:case.zero(x,time))
        applied = ops.traction_load(facets,{tag:(lambda x,n,f=f:f(x,n,time)) for tag,f in tractions.items()})
        rhs = np.r_[body+applied+ops.mass @ history,np.zeros(npres)]
        residual = linear @ guess-rhs
        convection,jacobian = ops.convection(guess[:2*nv],jacobian=True)
        residual[:2*nv] += convection
        matrix = ops.saddle(alpha/dt*ops.mass+ops.viscous+jacobian,False)[free][:,free].tocsr()
        matrix.sum_duplicates(); matrix.sort_indices()
        b = -residual[free]
        reference = _direct(matrix,b)
        residual_norm = float(np.linalg.norm(matrix @ reference-b)/max(np.linalg.norm(b),1.))
        if residual_norm > 1e-10:
            raise RuntimeError('reference matrix solve failed residual threshold')
        folder = destination/f'step{step:05d}'
        folder.mkdir()
        save_npz(folder/'matrix.npz',matrix,compressed=False)
        np.savez(folder/'vectors.npz',rhs=b,reference=reference,free=free)
        cases.append(dict(path=folder.name,step=step,time=time,shape=list(matrix.shape),nnz=matrix.nnz,
            reference_relative_residual=residual_norm,velocity_nodes=nv,pressure_nodes=npres,
            triangles=len(triangles),matrix_sha256=file_sha256(folder/'matrix.npz'),
            vectors_sha256=file_sha256(folder/'vectors.npz')))
    return write_manifest(destination/'manifest.json',dict(case='dfg_2d3 Newton correction systems',
        trajectory=str(path.resolve()),trajectory_sha256=archive_sha,dt=dt,steps=list(steps),
        archived_source_sha256=metadata.get('launch_source_sha256'),
        scope='algebraic performance fixtures from archived numerical states; not independent PDE verification'),
        dict(status='prepared',cases=cases),perf_counter()-begun)

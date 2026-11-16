"""Atomic, pickle-free flow checkpoints with exact scientific compatibility."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import numpy as np
import scipy
from physgraph.core.flow_state import FlowRestartState


def flow_source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ('core', 'physics', 'mesh', 'fem'):
        for path in sorted((root/folder).glob('*.py')):
            digest.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def mesh_fingerprint(space):
    digest = hashlib.sha256()
    mesh = space.mesh
    for value in (mesh.nodes, mesh.triangles, mesh.boundary, mesh.boundary_edges,
                  mesh.boundary_edge_tags, space.points, space.cells):
        if value is None:
            digest.update(b'None\0')
        else:
            a = np.ascontiguousarray(value)
            digest.update(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return digest.hexdigest()


def flow_runtime_identity():
    """Reject unverified migration between numerical library/runtime versions."""
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                machine=platform.machine(), byteorder=sys.byteorder)


def _digest(metadata, arrays):
    digest = hashlib.sha256(metadata.encode())
    for a in arrays:
        digest.update(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return digest.hexdigest()


def save_flow_checkpoint(path, state):
    """Replace one checkpoint atomically; preserve the old file if writing fails."""
    if not isinstance(state, FlowRestartState):
        raise ValueError('checkpoint requires a FlowRestartState')
    if type(state.step) is not int or state.step < 1:
        raise ValueError('checkpoint requires a completed positive step')
    arrays = [np.asarray(getattr(state, k)) for k in ('previous', 'older', 'guess')]
    if any(a.dtype != np.float64 or a.ndim != 1 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('checkpoint state must contain finite float64 vectors')
    metadata = json.dumps(dict(schema=1, contract=state.contract, step=state.step,
                               initial_diagnostics=state.initial_diagnostics), sort_keys=True, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, metadata=np.array(metadata),
                checksum=np.array(_digest(metadata, arrays)), previous=arrays[0], older=arrays[1], guess=arrays[2])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_flow_checkpoint(path):
    """Load without pickle, verifying integrity and numerical source identity.

    Mesh, runtime, solver and caller-owned problem identity are checked again by
    the solver before continuation. This checksum detects accidental corruption;
    it is not authentication of a file supplied by an untrusted author.
    """
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'metadata','checksum','previous','older','guess'}:
            raise ValueError('invalid checkpoint archive keys')
        metadata = str(archive['metadata'])
        arrays = [archive[k].copy() for k in ('previous','older','guess')]
        if any(a.dtype != np.float64 or a.ndim != 1 or not np.isfinite(a).all() for a in arrays):
            raise ValueError('invalid checkpoint vectors')
        if str(archive['checksum']) != _digest(metadata, arrays):
            raise ValueError('checkpoint checksum mismatch')
    data = json.loads(metadata)
    if (not isinstance(data, dict)
            or set(data) != {'schema', 'contract', 'step', 'initial_diagnostics'}
            or type(data['schema']) is not int or data['schema'] != 1
            or type(data['step']) is not int or data['step'] < 1
            or not isinstance(data['contract'], dict)
            or not isinstance(data['initial_diagnostics'], dict)):
        raise ValueError('invalid checkpoint schema or step')
    if data['contract'].get('source') != flow_source_fingerprint():
        raise ValueError('checkpoint source mismatch')
    return FlowRestartState(data['contract'], data['step'], *arrays, data['initial_diagnostics'])

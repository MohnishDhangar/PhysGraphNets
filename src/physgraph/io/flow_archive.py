"""Bounded-memory, single-writer flow segments with atomic commit publication."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from physgraph.core.types import Mesh2D
from physgraph.fem.navier_stokes import FlowTrajectory
from physgraph.fem.stokes import taylor_hood_space
from physgraph.io.flow_checkpoint import load_flow_checkpoint, save_flow_checkpoint, mesh_fingerprint


def _checksum(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        a = np.ascontiguousarray(value)
        digest.update(name.encode()+b'\0'+str(a.dtype).encode()+b'\0')
        digest.update(str(a.shape).encode()+b'\0'+a.tobytes())
    return digest.hexdigest()


def _atomic(path, write):
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Persist the rename as well as file contents on the supported POSIX hosts.
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save(path, **values):
    values = {key: np.asarray(value) for key, value in values.items()}
    if any(a.dtype.hasobject for a in values.values()):
        raise ValueError('flow archive must not contain object arrays')
    _atomic(path, lambda stream: np.savez_compressed(stream, checksum=np.array(_checksum(values)), **values))


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files if key != 'checksum'}
        if 'checksum' not in archive or str(archive['checksum']) != _checksum(values):
            raise ValueError('flow archive checksum mismatch: '+str(path))
    return values


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


@contextmanager
def exclusive_directory_writer(directory):
    """POSIX process lock released by the OS after an abrupt process exit."""
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError('flow archive writing requires POSIX file locking') from exc
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path/'writer.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another process is writing this flow archive') from exc
        try:
            yield path
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class FlowArchive:
    """Read an immutable committed prefix; iteration holds at most one segment.

    Writers use POSIX advisory locking (Linux/macOS CPU and cloud hosts). A crashed
    process releases its lock automatically. The index is the commit point: files
    beyond its chunk count are uncommitted and may be overwritten on resume.
    """
    def __init__(self, directory):
        self.directory = Path(directory)
        self.index = json.loads((self.directory/'index.json').read_text())
        index = self.index
        if (not isinstance(index, dict)
                or set(index) != {'schema', 'artifact', 'chunks', 'completed_steps', 'contract'}
                or type(index['schema']) is not int or index['schema'] != 1 or index['artifact'] != 'flow_segments'
                or type(index['chunks']) is not int or index['chunks'] < 0
                or type(index['completed_steps']) is not int or index['completed_steps'] < 0
                or (index['chunks'] == 0) != (index['completed_steps'] == 0)
                or (index['chunks'] == 0 and index['contract'] is not None)
                or (index['chunks'] > 0 and not isinstance(index['contract'], dict))):
            raise ValueError('invalid flow archive index')
        self._space = None

    @classmethod
    @contextmanager
    def writer(cls, directory, *, resume=False):
        with exclusive_directory_writer(directory) as path:
            archive = None
            try:
                if resume:
                    archive = cls(path)
                    archive.validate()
                else:
                    if any(p.name != 'writer.lock' for p in path.iterdir()):
                        raise ValueError('new flow archive directory must be empty')
                    cls._publish(path, dict(schema=1, artifact='flow_segments', chunks=0,
                                            completed_steps=0, contract=None))
                    archive = cls(path)
                archive._writing = True
                yield archive
            finally:
                if archive is not None:
                    archive._writing = False

    @staticmethod
    def _publish(path, index):
        _atomic(path/'index.json', lambda stream: stream.write((_json(index)+'\n').encode()))

    @property
    def completed_steps(self):
        return self.index['completed_steps']

    @property
    def space(self):
        if self._space is None:
            data = _load(self.directory/'mesh.npz')
            mesh = Mesh2D(data['nodes'], data['triangles'], data['boundary'],
                data['boundary_edges'] if bool(data['tagged']) else None,
                data['boundary_edge_tags'] if bool(data['tagged']) else None)
            self._space = taylor_hood_space(mesh)
            if (mesh_fingerprint(self._space) != self.index['contract']['mesh']
                    or not np.array_equal(data['velocity_nodes'], self._space.points)
                    or not np.array_equal(data['velocity_cells'], self._space.cells)):
                raise ValueError('flow archive mesh/DOF mismatch')
        return self._space

    def _path(self, kind, chunk):
        return self.directory/f'{kind}_{chunk:08d}.npz'

    def restart_state(self):
        if self.index['chunks'] == 0:
            return None
        state = load_flow_checkpoint(self._path('state', self.index['chunks']))
        if state.step != self.completed_steps or state.contract != self.index['contract']:
            raise ValueError('archive/checkpoint contract mismatch')
        return state

    def append(self, trajectory):
        if not getattr(self, '_writing', False):
            raise RuntimeError('append requires the exclusive archive writer context')
        state = trajectory.restart_state
        if state is None:
            raise ValueError('segment requires an explicit restart contract')
        start, end = self.completed_steps, state.step
        if end <= start or len(trajectory.times) != end-start+1:
            raise ValueError('segment does not extend committed trajectory')
        if self.index['contract'] is not None and state.contract != self.index['contract']:
            raise ValueError('segment restart contract differs from archive')
        if mesh_fingerprint(trajectory.space) != state.contract['mesh']:
            raise ValueError('segment mesh differs from restart contract')
        count, nv, npres = end-start, len(trajectory.space.points), len(trajectory.space.mesh.nodes)
        for name, shape in (('times', (count+1,)), ('velocity', (count+1,nv,2)),
                            ('pressure', (count,npres)), ('reactions', (count,nv,2))):
            a = getattr(trajectory, name)
            if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
                raise ValueError('invalid segment array: '+name)
        if (state.guess.shape != (2*nv+npres+int(state.contract['mean_gauge']),)
                or not np.array_equal(state.guess[:2*nv], state.previous)
                or not np.array_equal(state.guess[2*nv:2*nv+npres], trajectory.pressure[-1])
                or state.initial_diagnostics != trajectory.initial_diagnostics):
            raise ValueError('segment and checkpoint pressure or diagnostics disagree')
        if not np.array_equal(trajectory.velocity[-1].T.ravel(), state.previous):
            raise ValueError('segment and checkpoint terminal velocity disagree')
        if not np.array_equal(trajectory.velocity[-2].T.ravel(), state.older):
            raise ValueError('segment and checkpoint temporal history disagree')
        if start:
            previous = self.restart_state()
            if (not np.array_equal(trajectory.velocity[0].T.ravel(), previous.previous)
                    or trajectory.velocity_before_segment is None
                    or not np.array_equal(trajectory.velocity_before_segment.T.ravel(), previous.older)):
                raise ValueError('segment does not continue committed velocity histories')
        elif trajectory.velocity_before_segment is not None:
            raise ValueError('first segment must start at the original projection')
        expected_times = state.contract['initial_time']+state.contract['dt']*np.arange(start, end+1)
        if not np.array_equal(trajectory.times, expected_times):
            raise ValueError('segment time grid differs from contract')
        if ([row['step'] for row in trajectory.diagnostics] != list(range(start+1, end+1))
                or [row['time'] for row in trajectory.diagnostics] != expected_times[1:].tolist()):
            raise ValueError('segment diagnostic steps differ from contract')
        chunk = self.index['chunks']+1
        if start == 0:
            mesh = trajectory.space.mesh
            _save(self.directory/'mesh.npz', nodes=mesh.nodes, triangles=mesh.triangles,
                  boundary=mesh.boundary, tagged=np.array(mesh.boundary_edges is not None),
                  boundary_edges=np.empty((0,2), dtype=np.int64) if mesh.boundary_edges is None else mesh.boundary_edges,
                  boundary_edge_tags=np.array([], dtype='U1') if mesh.boundary_edge_tags is None else mesh.boundary_edge_tags,
                  velocity_nodes=trajectory.space.points, velocity_cells=trajectory.space.cells)
        metadata = dict(start=start, end=end, contract=state.contract,
                        diagnostics=trajectory.diagnostics, initial_diagnostics=trajectory.initial_diagnostics)
        _save(self._path('segment', chunk), times=trajectory.times, velocity=trajectory.velocity,
              pressure_times=trajectory.times[1:], pressure=trajectory.pressure, reactions=trajectory.reactions,
              velocity_before_segment=(np.empty((0, 2)) if trajectory.velocity_before_segment is None
                                       else trajectory.velocity_before_segment), metadata=np.array(_json(metadata)))
        save_flow_checkpoint(self._path('state', chunk), state)
        _sync_directory(self.directory)
        new_index = {**self.index, 'chunks': chunk, 'completed_steps': end, 'contract': state.contract}
        try:
            self._publish(self.directory, new_index)
        except Exception:
            # Rename may have succeeded before directory fsync failed. Reopening
            # and auditing the disk index is required before any further append.
            self._writing = False
            raise
        self.index = new_index

    def iter_segments(self):
        """Yield one validated segment at a time; do not accumulate for large runs."""
        previous_velocity = previous_older = None
        start = 0
        for chunk in range(1, self.index['chunks']+1):
            data = _load(self._path('segment', chunk))
            metadata = json.loads(str(data['metadata']))
            contract, end = self.index['contract'], metadata['end']
            if metadata['start'] != start or end <= start or metadata['contract'] != contract:
                raise ValueError('noncontiguous or incompatible flow archive segment')
            count, nv, npres = end-start, len(self.space.points), len(self.space.mesh.nodes)
            for key, shape in (('times',(count+1,)), ('pressure_times',(count,)),
                               ('velocity',(count+1,nv,2)), ('pressure',(count,npres)),
                               ('reactions',(count,nv,2)),
                               ('velocity_before_segment',(0,2) if start == 0 else (nv,2))):
                a = data[key]
                if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
                    raise ValueError('invalid flow archive array: '+key)
            times = contract['initial_time']+contract['dt']*np.arange(start, end+1)
            if (not np.array_equal(times, data['times'])
                    or not np.array_equal(times[1:], data['pressure_times'])
                    or [row['step'] for row in metadata['diagnostics']] != list(range(start+1,end+1))
                    or [row['time'] for row in metadata['diagnostics']] != times[1:].tolist()):
                raise ValueError('invalid flow archive temporal alignment')
            if start and (not np.array_equal(data['velocity'][0], previous_velocity)
                          or not np.array_equal(data['velocity_before_segment'], previous_older)):
                raise ValueError('flow archive history discontinuity')
            result = FlowTrajectory(self.space, data['times'], data['velocity'], data['pressure'],
                data['reactions'], metadata['diagnostics'], metadata['initial_diagnostics'],
                contract['viscosity'], None if contract['dirichlet_tags'] is None else tuple(contract['dirichlet_tags']),
                contract['scheme'], contract['viscous_form'], None,
                None if start == 0 else data['velocity_before_segment'])
            previous_velocity, previous_older = data['velocity'][-1].copy(), data['velocity'][-2].copy()
            start = end
            yield result
        if start != self.completed_steps:
            raise ValueError('flow archive index disagrees with committed segments')

    def validate(self):
        """Audit the committed prefix and final restart state with bounded memory."""
        final = None
        for segment in self.iter_segments():
            final = segment
        if final is not None:
            state = self.restart_state()
            nv = len(self.space.points)
            if (not np.array_equal(state.previous, final.velocity[-1].T.ravel())
                    or not np.array_equal(state.older, final.velocity[-2].T.ravel())
                    or not np.array_equal(state.guess[:2*nv], state.previous)
                    or not np.array_equal(state.guess[2*nv:2*nv+len(self.space.mesh.nodes)], final.pressure[-1])):
                raise ValueError('final checkpoint disagrees with archived trajectory')
        return dict(chunks=self.index['chunks'], completed_steps=self.completed_steps)

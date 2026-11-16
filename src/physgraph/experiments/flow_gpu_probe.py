"""Record optional ROCm sparse arithmetic separately from solver qualification."""
import json
from importlib.metadata import version
from pathlib import Path
import resource
import sys
from time import perf_counter
import numpy as np
from scipy.sparse import load_npz
from physgraph.io.manifest import write_manifest
from physgraph.profiling.flow_linear import file_sha256
from physgraph.profiling.rocm_sparse import probe_spmv


def run_gpu_probe(*, fixtures, output, sdk_site, device=0, repeats=30):
    root, dest = Path(fixtures), Path(output)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('GPU probe output must be empty')
    sys.path.insert(0, str(Path(sdk_site).resolve()))
    fixture_manifest = json.loads((root/'manifest.json').read_text())
    config = dict(fixtures=str(root.resolve()), fixture_manifest_sha256=file_sha256(root/'manifest.json'),
        device_index=device, repeats=repeats, sdk_site=str(Path(sdk_site).resolve()),
        scope='GPU float64 sparse matrix-vector capability only; not factorization or trajectory acceleration')
    metrics = dict(status='running', cases=[])
    start = perf_counter()
    def save():
        metrics['peak_host_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report = write_manifest(dest/'manifest.json', config, metrics, perf_counter()-start)
        report['device'] = 'gpu'
        (dest/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report
    save()
    try:
        config['rocm_packages'] = {name: version(name) for name in
            ('rocm','rocm-sdk-core','rocm-sdk-libraries','rocm-sdk-device-gfx1034')}
        for case in fixture_manifest['metrics']['cases']:
            folder = root/case['path']
            for name in ('matrix','vectors'):
                if file_sha256(folder/(name+'.npz')) != case[name+'_sha256']:
                    raise ValueError('fixture checksum mismatch')
            a = load_npz(folder/'matrix.npz')
            with np.load(folder/'vectors.npz', allow_pickle=False) as values:
                x = values['reference']
            result = probe_spmv(a, x, device=device, repeats=repeats)
            metrics['cases'].append(dict(case=case['path'], **result))
            save()
        metrics['status'] = 'passed_capability_probe' if all(c['passed'] for c in metrics['cases']) else 'failed_numerical_check'
    except Exception as exc:
        metrics.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        save()
        raise
    return save()

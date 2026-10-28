"""Artifact-producing independent comparison suite, including blocked status."""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from physgraph.io.manifest import write_manifest
from physgraph.verification.cases import CASE_IDS, comparison_case
from physgraph.verification.fenicsx import compare, require_fenicsx, FEniCSxUnavailable


def run_comparisons(*, cases=None, resolutions=(4, 8), output=None, tolerance=1e-8):
    started = perf_counter()
    cases = list(CASE_IDS if cases is None else cases)
    if not cases or any(name not in CASE_IDS for name in cases):
        raise ValueError('select at least one supported comparison case')
    if (not resolutions or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in resolutions)
            or any(b <= a for a, b in zip(resolutions, resolutions[1:]))):
        raise ValueError('resolutions must be strictly increasing integers >= 2')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be positive and finite')
    dest = Path(output) if output is not None else Path('outputs/fenicsx') / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError(f'output directory must be empty: {dest}')
    dest.mkdir(parents=True, exist_ok=True)
    config = {'cases': cases, 'resolutions': list(resolutions), 'tolerance': tolerance,
              'fenicsx_quadrature_degree': 12, 'physgraph_assembly_order': 7,
              'mesh_policy': 'identical input triangles and coordinates', 'output': str(dest.resolve())}
    rows = []
    try:
        require_fenicsx()
        for name in cases:
            for n in resolutions:
                row, arrays = compare(comparison_case(name, n), tolerance=tolerance,
                                      cache_dir=dest/'jit_cache')
                row['resolution'] = n
                rows.append(row)
                np.savez_compressed(dest/f'{name}_{n}.npz', **arrays)
                print(f'{name} n={n}: passed={row["passed"]}, max difference={row["max_nodal_difference"]:.3e}', flush=True)
        status = 'passed' if all(row['passed'] for row in rows) else 'failed'
        reason = None
    except FEniCSxUnavailable as exc:
        status, reason = 'blocked_missing_dependency', str(exc)
    except Exception as exc:
        write_manifest(dest/'manifest.json', config,
                       {'status': 'failed', 'reason': str(exc), 'comparisons': rows}, perf_counter()-started)
        raise
    metrics = {'status': status, 'reason': reason, 'comparisons': rows}
    (dest/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    return write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)

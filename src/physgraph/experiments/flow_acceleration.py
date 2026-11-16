"""Predeclared repeated CPU linear-backend comparison; preserve every failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
from physgraph.io.manifest import write_manifest
from physgraph.profiling.flow_linear import file_sha256


def run_linear_study(*, fixtures, output, backends=('scipy','mumps','superlu_dist'),
                     threads=(1,2,4,6), repeats=3):
    if type(repeats) is not int or repeats < 2:
        raise ValueError('at least two fresh-process repeats required')
    if (not backends or len(set(backends))!=len(backends) or set(backends)-{'scipy','mumps','superlu_dist'}
            or not threads or len(set(threads))!=len(threads)
            or any(type(n) is not int or n<1 for n in threads)):
        raise ValueError('invalid backends or thread counts')
    dest, fixture_root = Path(output), Path(fixtures)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('benchmark output must be empty')
    fixtures_manifest = json.loads((fixture_root/'manifest.json').read_text())
    cases = fixtures_manifest['metrics']['cases']
    for case in cases:
        for name in ('matrix','vectors'):
            if file_sha256(fixture_root/case['path']/(name+'.npz')) != case[name+'_sha256']:
                raise ValueError('fixture checksum mismatch')
    dest.mkdir(parents=True,exist_ok=True)
    config = dict(fixtures=str(fixture_root.resolve()),fixture_manifest_sha256=file_sha256(fixture_root/'manifest.json'),
        backends=list(backends),threads=list(threads),repeats=repeats,precision='float64',
        timing='one fresh process per trial; imports/file read excluded; matrix conversion, factorization, solve and answer copy included',
        memory='Linux process peak RSS including imports and fixture; not an allocation delta',
        residual_tolerance=1e-10,absolute_reference_tolerance=1e-9,
        scope='sparse Newton correction systems, not complete trajectory speedup',
        ordering='repeat-major, deterministic shuffle of cases/backends/thread counts')
    rows, start = [], perf_counter()
    def save(status):
        summaries=[]
        for case in cases:
            for backend in backends:
                for count in threads:
                    group=[r for r in rows if (r['case'],r['backend'],r['threads'])==(case['path'],backend,count)]
                    passed=len(group)==repeats and all(r['status']=='completed' and r['result']['passed'] for r in group)
                    summaries.append(dict(case=case['path'],backend=backend,threads=count,passed=passed,
                        median_seconds=float(np.median([r['result']['seconds'] for r in group])) if passed else None))
        return write_manifest(dest/'manifest.json',config,dict(status=status,runs=rows,summary=summaries),perf_counter()-start)
    save('running')
    combinations=[(case['path'],backend,count) for case in cases for backend in backends for count in threads]
    rng=np.random.default_rng(20261112)
    for repeat in range(repeats):
        for index in rng.permutation(len(combinations)):
            case,backend,count=combinations[index]
            settings=dict(fixture=str((fixture_root/case).resolve()),backend=backend)
            row=dict(case=case,backend=backend,threads=count,repeat=repeat)
            env={**os.environ,'OPENBLAS_NUM_THREADS':str(count),'OMP_NUM_THREADS':str(count),
                 'MKL_NUM_THREADS':str(count),'UCX_TLS':'self'}
            try:
                process=subprocess.run([sys.executable,'-m','physgraph.profiling.flow_linear_worker'],
                    input=json.dumps(settings),text=True,capture_output=True,env=env,timeout=180)
                label=f'{case}_{backend}_t{count}_r{repeat}'
                (dest/(label+'.log')).write_text(process.stdout+'\n'+process.stderr)
                if process.returncode:
                    raise RuntimeError(f'worker exit {process.returncode}; see {label}.log')
                result=json.loads(process.stdout.splitlines()[-1])
                row.update(status='completed',result=result)
            except (subprocess.SubprocessError,OSError,ValueError,RuntimeError) as exc:
                row.update(status='failed',reason=str(exc))
            rows.append(row); save('running')
            print(f'{case} {backend} threads={count} repeat={repeat}: {row["status"]}',flush=True)
    return save('completed' if all(r['status']=='completed' and r['result']['passed'] for r in rows) else 'completed_with_failures')

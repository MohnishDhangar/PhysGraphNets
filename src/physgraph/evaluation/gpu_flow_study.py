"""Adoption gate: numerical success in every repeat AND useful total speedup."""
import numpy as np


def assess_gpu_study(runs,comparisons,cases,*,repeats,minimum_speedup=1.5,baseline='mumps',candidate='rocm_gmres'):
    summaries=[]
    for case in cases:
        selected={backend:[r for r in runs if r['case']==case and r['backend']==backend]
                  for backend in (baseline,candidate)}
        valid=all(len(rows)==repeats and {r['repeat'] for r in rows}==set(range(repeats))
                  and all(r['status']=='passed' for r in rows) for rows in selected.values())
        parity=[r for r in comparisons if r['case']==case]
        valid=valid and len(parity)==repeats and {r['repeat'] for r in parity}==set(range(repeats)) and all(r['passed'] for r in parity)
        medians={b:float(np.median([r['process_wall_seconds'] for r in rows])) if valid else None
                 for b,rows in selected.items()}
        ratio=medians[baseline]/medians[candidate] if valid else None
        summaries.append(dict(case=case,numerically_valid=valid,median_process_seconds=medians,
                              end_to_end_speedup=ratio,performance_passed=bool(valid and ratio>=minimum_speedup)))
    return dict(cases=summaries,numerically_valid=bool(cases) and all(r['numerically_valid'] for r in summaries),
                adopt=bool(cases) and all(r['performance_passed'] for r in summaries))

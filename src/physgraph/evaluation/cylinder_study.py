"""Fixed, dimensionless force-history and refinement diagnostics for DFG 2D-3."""
import numpy as np

from physgraph.evaluation.cylinder_flow import published_comparison
from physgraph.physics.cylinder_flow import CylinderFlowCase


def compare_cylinder_histories(coarse, fine):
    """Compare on coarse solved times; no extrapolation or fitted phase shift.

    Scales are fixed published quantities, never a vanishing instantaneous load.
    These are differences between numerical solutions, not continuum error bounds.
    """
    reference = published_comparison({}, CylinderFlowCase(), 0.)['reference']
    a, b = coarse['rows'], fine['rows']
    ta = np.array([r['time'] for r in a])
    tb = np.array([r['time'] for r in b])
    if (len(ta) < 2 or len(tb) < 2 or not np.isfinite(ta).all()
            or not np.isfinite(tb).all() or np.any(np.diff(ta) <= 0)
            or np.any(np.diff(tb) <= 0) or ta[0] < tb[0]-1e-12
            or not np.isclose(ta[-1], tb[-1], rtol=0, atol=1e-12)):
        raise ValueError('histories require increasing, overlapping times and equal final time')
    summaries = {
        key: float(abs(coarse[key]-fine[key])/abs(reference[key]))
        for key in ('max_drag', 'max_lift', 'final_pressure_difference')
    }
    peaks = {key: float(abs(coarse[key]-fine[key]))
             for key in ('time_max_drag', 'time_max_lift')}
    histories = {}
    for key, scale in (('drag', reference['max_drag']), ('lift', reference['max_lift']),
                       ('pressure_difference', reference['final_pressure_difference'])):
        av = np.array([r[key] for r in a])
        bv = np.interp(ta, tb, [r[key] for r in b])
        if not np.isfinite(av).all() or not np.isfinite(bv).all():
            raise ValueError('history values must be finite')
        # Trapezoid time integration supports unequal output grids.
        squared = ((av-bv)/scale)**2
        integral = np.sum(np.diff(ta)*(squared[:-1]+squared[1:])/2)
        histories[key] = float(np.sqrt(integral/(ta[-1]-ta[0])))
    return {'summary_differences': summaries, 'peak_time_differences': peaks,
            'normalized_history_rms': histories, 'comparison_times': len(ta),
            'interval': [float(ta[0]), float(ta[-1])]}


def refinement_acceptance(comparison, criteria):
    return {
        'summary': max(comparison['summary_differences'].values()) <= criteria['summary'],
        'peak_times': max(comparison['peak_time_differences'].values()) <= criteria['peak_times'],
        'history_rms': max(comparison['normalized_history_rms'].values()) <= criteria['history_rms'],
    }

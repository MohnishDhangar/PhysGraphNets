"""Numerical acceptance is separate from successful program execution."""
import math

METRICS = ('relative_l2', 'relative_h1_seminorm', 'absolute_l2', 'absolute_h1_seminorm',
           'neumann_flux_rms', 'neumann_flux_relative_rms', 'max_neumann_flux_error', 'max_nodal_error', 'max_boundary_error', 'holdout_residual_rms', 'holdout_residual_relative_rms')


def validate_thresholds(thresholds):
    if thresholds is None:
        return
    if not isinstance(thresholds, dict) or not thresholds or set(thresholds)-set(METRICS):
        raise ValueError('acceptance must be a nonempty mapping of supported error metrics')
    for value in thresholds.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('acceptance thresholds must be finite nonnegative numbers')


def assess(metrics, thresholds):
    validate_thresholds(thresholds)
    if thresholds is None:
        return {'status': 'not_assessed'}
    checks = {key: math.isfinite(metrics[key]) and metrics[key] <= limit for key, limit in thresholds.items()}
    return {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks}

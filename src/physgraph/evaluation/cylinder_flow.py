"""Cylinder force histories and P1 pressure probes on the numerical mesh."""
import numpy as np
from physgraph.evaluation.obstacle_stokes import obstacle_dofs
from physgraph.evaluation.stokes import stokes_boundary_integrals


def pressure_probe_weights(mesh, points):
    vertices = mesh.nodes[mesh.triangles]
    matrix = np.stack((vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]), axis=2)
    answers = []
    for point in points:
        local = np.linalg.solve(matrix, (point-vertices[:, 0])[..., None])[..., 0]
        bary = np.column_stack((1-local.sum(axis=1), local))
        selected = np.flatnonzero(np.min(bary, axis=1) >= -1e-10)
        if not len(selected):
            raise ValueError('pressure probe is outside the numerical fluid domain')
        cell = int(selected[0])
        answers.append((mesh.triangles[cell], bary[cell]))
    return answers


def cylinder_history(trajectory, case):
    body = obstacle_dofs(trajectory.frame(1), ('cylinder',))
    x, y = case.center
    probes = pressure_probe_weights(trajectory.space.mesh, np.array([[x-case.radius,y],[x+case.radius,y]]))
    rows = []
    for step, time in enumerate(trajectory.times[1:], 1):
        frame = trajectory.frame(step)
        integrals = stokes_boundary_integrals(frame)
        surface = -np.asarray(integrals['by_tag']['cylinder']['force_on_fluid'])
        reaction = -frame.velocity_reactions[body].sum(axis=0)
        pressure = [float(weights @ frame.pressure[indices]) for indices, weights in probes]
        rows.append(dict(drag=float(reaction[0]/case.force_scale),
            lift=float(reaction[1]/case.force_scale), surface_drag=float(surface[0]/case.force_scale),
            surface_lift=float(surface[1]/case.force_scale), pressure_difference=pressure[0]-pressure[1],
            max_obstacle_speed=float(np.max(np.abs(frame.velocity[body]))),
            net_flux=integrals['net_flux'], inlet_flux=integrals['by_tag']['left']['flux'],
            outlet_flux=integrals['by_tag']['right']['flux'], load_kind=integrals['load_kind'],
            **frame.diagnostics))
    drag_peak = max(rows, key=lambda r:r['drag'])
    lift_peak = max(rows, key=lambda r:r['lift'])
    return {'rows': rows, 'max_drag': drag_peak['drag'], 'time_max_drag': drag_peak['time'],
            'max_lift': lift_peak['lift'], 'time_max_lift': lift_peak['time'],
            'final_pressure_difference': rows[-1]['pressure_difference']}


def published_comparison(history, case, final_time):
    # Finest tabulated Q2/P1disc reference, not an analytical solution.
    # The HTML summary loses the pressure sign. The official pointvalues_lv6
    # archive has p_front=-.43873062102 and p_rear=-.32718923230 at its final
    # stored time 7.9996875: p_front-p_rear=-.11154138872. Preserve that sign.
    reference = dict(max_drag=2.9437637214, time_max_drag=3.9365625,
                     max_lift=.47748781595, time_max_lift=5.6928125,
                     final_pressure_difference=-.11154138872)
    result = {'source': 'https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html',
              'pressure_source': 'https://wwwold.mathematik.tu-dortmund.de/~featflow/media/dfg_bench3_2d/pressure_q2_cn_lv1-6_dt4.zip',
              'pressure_archive_sha256': 'dc49ebecc647701f260646fdee02ee0bdca43d684350ac174939c9d2ead60b6f',
              'pressure_reference_time': 7.9996875,
              'pressure_convention': 'front minus rear; signed official raw data, not unsigned HTML table',
              'reference': reference, 'relative_quantity_tolerance': .03, 'peak_time_tolerance': .05}
    if not case.matches_published_case or not np.isclose(final_time, 8., rtol=0, atol=1e-12):
        return {**result, 'status': 'not_assessed', 'reason': 'requires original parameters and complete [0,8] interval'}
    relative = {key: abs(history[key]/reference[key]-1) for key in
                ('max_drag','max_lift','final_pressure_difference')}
    timing = {key: abs(history[key]-reference[key]) for key in ('time_max_drag','time_max_lift')}
    passed = max(relative.values()) <= .03 and max(timing.values()) <= .05
    return {**result, 'relative_errors': relative, 'peak_time_errors': timing,
            'status': 'passed' if passed else 'failed_acceptance'}

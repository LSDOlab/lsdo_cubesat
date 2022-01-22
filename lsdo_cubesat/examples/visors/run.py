import numpy as np
import time
import matplotlib.pyplot as plt
import pickle

from lsdo_cubesat.parameters.swarm import SwarmParams
from lsdo_cubesat.examples.visors.plot_sim_state import plot_sim_state
from lsdo_cubesat.examples.visors.get_all_varnames import get_all_varnames
from lsdo_cubesat.examples.visors.generate_reference_orbit import generate_reference_orbit
from lsdo_cubesat.examples.visors.warm_start import warm_start
from lsdo_cubesat.examples.visors.make_swarm import make_swarm

np.random.seed(0)

duration = 95.
# if True:
if False:
    num_times = 1501
    num_cp = 30
    num_times = 301
    num_cp = 30
# elif False:
elif True:
    num_times = 40
    num_cp = 16
else:
    num_times = 40
    num_cp = 5
step_size = duration * 60 / (num_times - 1)

check_derivatives = True
# check_derivatives = False
use_fd = True
# use_fd = False
fd_step = 1e-4
plot_reference_orbit = False
plot_initial_outputs = False
# plot_initial_outputs = True
# optimize_model = True
optimize_model = False
# ws = True
ws = False
# visualize_impl = True
visualize_impl = False
visualize_recursive = True
# visualize_recursive = False
save_all_iterations = True
# save_all_iterations = False

if check_derivatives is True and step_size > 1e-4:
    step_size *= 1e-10
print('STEP SIZE', step_size)
ref_orbit, ax = generate_reference_orbit(
    num_times,
    step_size,
    plot=plot_reference_orbit,
)
sim = make_swarm(
    SwarmParams(
        num_times=num_times,
        duration=duration,
        num_cp=num_cp,
        step_size=step_size,
        cross_threshold=0.7,
        # cross_threshold=0.857,
        # cross_threshold=0.2,
        # cross_threshold=-0.9,
        # cross_threshold=0.9,
    ))
sim['reference_orbit_state_km'] = ref_orbit

if ws is True:
    warm_start(sim)

if visualize_impl is True:
    sim.run()
    sim.visualize_implementation(recursive=visualize_recursive)
    exit()

if check_derivatives is True:
    start = time.time()
    sim.run()
    end = time.time()

    print('time to run once', end - start)
    if use_fd is True:
        sim.prob.check_totals(compact_print=True, method='fd', step=fd_step)
    else:
        sim.prob.check_totals(compact_print=True, method='cs')
    exit()

if optimize_model is False:
    sim.run()
    # sim.visualize_implementation()
    print('optics_cubesat.pitch', sim['optics_cubesat.pitch'])
    print('optics_cubesat.rw_torque', sim['optics_cubesat.rw_torque'])
    print('optics_cubesat.rw_torque_min', sim['optics_cubesat.rw_torque_min'])
    print('optics_cubesat.rw_torque_max', sim['optics_cubesat.rw_torque_max'])
    print('optics_cubesat.rw_speed', sim['optics_cubesat.rw_speed'])
    print('optics_cubesat.rw_speed_min', sim['optics_cubesat.rw_speed_min'])
    print('optics_cubesat.rw_speed_max', sim['optics_cubesat.rw_speed_max'])
    print('optics_cubesat.soc', sim['optics_cubesat.soc'])
    print('optics_cubesat.mmin_soc', sim['optics_cubesat.mmin_soc'])
    print('optics_cubesat.min_soc', sim['optics_cubesat.min_soc'])
    print('optics_cubesat.max_soc', sim['optics_cubesat.max_soc'])
    print('optics_cubesat.current', sim['optics_cubesat.current'])
    print('optics_cubesat.min_current', sim['optics_cubesat.min_current'])
    print('optics_cubesat.max_current', sim['optics_cubesat.max_current'])
    print('optics_cubesat.altitude_km', sim['optics_cubesat.altitude_km'])
    # print('optics_cubesat.min_altitude_m',
    #       sim['optics_cubesat.min_altitude_m'])
    print('optics_cubesat.min_altitude_km',
          sim['optics_cubesat.min_altitude_km'])
    print('separation_error_during_observation',
          sim['separation_error_during_observation'])
    print('min_separation_error', sim['min_separation_error'])
    print('max_separation_error', sim['max_separation_error'])
    print('view_plane_error_during_observation',
          sim['view_plane_error_during_observation'])
    print('min_view_plane_error', sim['min_view_plane_error'])
    print('max_view_plane_error', sim['max_view_plane_error'])
    print('optics_cubesat.current', sim['optics_cubesat.current'])
    print('optics_cubesat.max_current', sim['optics_cubesat.max_current'])
    print('optics_cubesat.min_current', sim['optics_cubesat.min_current'])
    print('detector_cubesat.current', sim['detector_cubesat.current'])
    print('detector_cubesat.max_current', sim['detector_cubesat.max_current'])
    print('detector_cubesat.min_current', sim['detector_cubesat.min_current'])

    if plot_reference_orbit is True:
        # plot sun direction
        sd = sim['sun_direction']
        ax.plot(
            sd[0, :] * 6000,
            sd[1, :] * 6000,
            sd[2, :] * 6000,
            'o',
        )
        plt.show()

    if plot_initial_outputs is True:
        plot_sim_state(sim)
else:
    from optimize import snoptc, SNOPT_options
    from optimization import get_obj_constraint_bounds, get_dv_bounds, get_problem_dimensions, inf

    varnames = get_all_varnames(sim)
    if save_all_iterations is True:
        from lsdo_cubesat.dash import Dash
        dashboard = Dash(varnames=varnames, run_file_name='run.py')
        sim.add_recorder(dashboard.get_recorder())

    def snopt_callback(mode, nnjac, x, fObj, gObj, fCon, gCon, nState):
        sim.update_design_variables(x)
        sim.run()
        totals = sim.compute_total_derivatives()

        # nonlinear objective term
        fObj = sim.objective()

        # nonlinear constraint terms
        fCon[:] = sim.constraints().flatten()

        # nonlinear objective gradient
        gObj[:] = totals[0, :].flatten()

        # nonlinear constraint Jacobian terms
        gCon[:] = totals[1:, :].T.flatten()

        return mode, fObj, gObj, fCon, gCon

    n, m = get_problem_dimensions(sim)
    print('Number of design variables:', n)
    print('Number of constraints:', m)
    xlower, xupper = get_dv_bounds(sim)
    Flow, Fupp = get_obj_constraint_bounds(sim)
    bl = np.concatenate((xlower, Flow[1:], Flow[0].reshape((1, ))))
    bu = np.concatenate((xupper, Fupp[1:], Fupp[0].reshape((1, ))))
    x0 = np.concatenate((sim.design_variables(), np.zeros(m)))

    options = SNOPT_options()
    options.setOption('Verbose', False)
    options.setOption('Solution print', False)
    options.setOption('Print filename', 'sntoya.out')
    options.setOption('Summary frequency', 1)
    options.setOption('Infinite bound', inf)
    options.setOption('Verify level', 0)
    options.setOption('Derivative level', 0)
    options.setOption('Print filename', 'visors.out')

    result = snoptc(
        snopt_callback,
        nnObj=n,
        nnCon=m - 1,
        nnJac=n,
        x0=x0,
        J=np.ones((m, n)),
        name='visors',
        iObj=0,
        bl=bl,
        bu=bu,
        options=options,
    )

    if save_all_iterations is False:
        data = dict()
        for name in varnames:
            data[name] = sim[name]

        with open('filename.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plot_sim_state(sim)

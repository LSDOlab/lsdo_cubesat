from re import T
import numpy as np
import time

from lsdo_cubesat.parameters.swarm import SwarmParams
from lsdo_cubesat.parameters.cubesat import CubesatParams
from lsdo_cubesat.parameters.ground_station import GroundStationParams
from lsdo_cubesat.swarm_group import Swarm
from lsdo_cubesat.orbit.reference_orbit_group import ReferenceOrbit

from csdl_om import Simulator

import matplotlib.pyplot as plt


def make_swarm(swarm):

    # initial state relative to reference orbit -- same for all s/c
    initial_orbit_state = np.array([1e-3] * 3 + [1e-3] * 3)

    cubesats = dict()
    # cubesats['sunshade'] = CubesatParams(
    #     name='sunshade',
    #     dry_mass=1.3,
    #     initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
    #     approx_altitude_km=500.,
    #     specific_impulse=47.,
    #     apogee_altitude=500.001,
    #     perigee_altitude=499.99,
    # )

    cubesats['optics'] = CubesatParams(
        name='optics',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state * np.random.rand(6),
        approx_altitude_km=500.,
        specific_impulse=47.,
        perigee_altitude=500.,
        apogee_altitude=500.,
    )

    cubesats['detector'] = CubesatParams(
        name='detector',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state * np.random.rand(6),
        approx_altitude_km=500.,
        specific_impulse=47.,
        perigee_altitude=500.002,
        apogee_altitude=499.98,
    )

    # cubesats['sunshade'].add(
    #     GroundStationParams(
    #         name='UCSD',
    #         lon=-117.1611,
    #         lat=32.7157,
    #         alt=0.4849,
    #     ))
    # cubesats['sunshade'].add(
    #     GroundStationParams(
    #         name='UIUC',
    #         lon=-88.2272,
    #         lat=32.8801,
    #         alt=0.2329,
    #     ))
    # cubesats['sunshade'].add(
    #     GroundStationParams(
    #         name='Georgia',
    #         lon=-84.3963,
    #         lat=33.7756,
    #         alt=0.2969,
    #     ))
    # cubesats['sunshade'].add(
    #     GroundStationParams(
    #         name='Montana',
    #         lon=-109.5337,
    #         lat=33.7756,
    #         alt=1.04,
    #     ))

    # cubesats['detector'].add(
    #     GroundStationParams(
    #         name='UCSD',
    #         lon=-117.1611,
    #         lat=32.7157,
    #         alt=0.4849,
    #     ))
    # cubesats['detector'].add(
    #     GroundStationParams(
    #         name='UIUC',
    #         lon=-88.2272,
    #         lat=32.8801,
    #         alt=0.2329,
    #     ))
    # cubesats['detector'].add(
    #     GroundStationParams(
    #         name='Georgia',
    #         lon=-84.3963,
    #         lat=33.7756,
    #         alt=0.2969,
    #     ))
    # cubesats['detector'].add(
    #     GroundStationParams(
    #         name='Montana',
    #         lon=-109.5337,
    #         lat=33.7756,
    #         alt=1.04,
    #     ))

    # cubesats['optics'].add(
    #     GroundStationParams(
    #         name='UCSD',
    #         lon=-117.1611,
    #         lat=32.7157,
    #         alt=0.4849,
    #     ))
    # cubesats['optics'].add(
    #     GroundStationParams(
    #         name='UIUC',
    #         lon=-88.2272,
    #         lat=32.8801,
    #         alt=0.2329,
    #     ))
    # cubesats['optics'].add(
    #     GroundStationParams(
    #         name='Georgia',
    #         lon=-84.3963,
    #         lat=33.7756,
    #         alt=0.2969,
    #     ))
    # cubesats['optics'].add(
    #     GroundStationParams(
    #         name='Montana',
    #         lon=-109.5337,
    #         lat=33.7756,
    #         alt=1.04,
    #     ))

    for v in cubesats.values():
        swarm.add(v)
    m = Swarm(swarm=swarm)
    start_compile = time.time()
    sim = Simulator(m)
    end_compile = time.time()
    total_compile_time = end_compile - start_compile
    print('========== TOTAL COMPILE TIME ==========')
    print(total_compile_time, 's')
    print('=========================================')
    return sim


if True:
    # if False:
    num_times = 1501
    num_cp = 30
elif False:
    # elif True:
    num_times = 40
    num_cp = 5
else:
    num_times = 4
    num_cp = 3

step_size = 95 * 60 / (num_times - 1)

swarm = SwarmParams(
    num_times=num_times,
    num_cp=num_cp,
    step_size=step_size,
    # cross_threshold=0.882,
    cross_threshold=0.857,
    # cross_threshold=-0.9,
    # cross_threshold=0.9,
)


def generate_reference_orbit(num_times, step_size):
    ref_orbit = Simulator(
        ReferenceOrbit(
            num_times=num_times,
            step_size=step_size,
        ))
    ref_orbit.run()
    return ref_orbit['reference_orbit_state_km']


sim = make_swarm(swarm)
sim['reference_orbit_state_km'] = generate_reference_orbit(
    num_times, step_size)

sim.run(time_run=True)
# sim.check_partials(compact_print=True)
# sim.visualize_implementation(time_run=True)
# exit()

plt.plot(sim['optics_cubesat_group.relative_orbit_state_m'][0, :])
plt.plot(sim['optics_cubesat_group.relative_orbit_state_m'][1, :])
plt.plot(sim['optics_cubesat_group.relative_orbit_state_m'][2, :])
# plt.plot(sim['detector_cubesat_group.soc'])
plt.show()

plt.plot(sim['observation_phase_indicator'])
plt.show()

plt.plot(sim['optics_cubesat_group.sun_LOS'][0, :])
plt.plot(sim['detector_cubesat_group.sun_LOS'][0, :])
plt.show()

plt.plot(sim['optics_cubesat_group.percent_exposed_area'][0, :])
plt.plot(sim['detector_cubesat_group.percent_exposed_area'][0, :])
plt.show()

plt.plot(sim['optics_cubesat_group.body_rates'][0, :])
plt.plot(sim['optics_cubesat_group.body_rates'][1, :])
plt.plot(sim['optics_cubesat_group.body_rates'][2, :])
plt.show()

plt.plot(sim['optics_cubesat_group.rw_torque'][0, :])
plt.plot(sim['optics_cubesat_group.rw_torque'][1, :])
plt.plot(sim['optics_cubesat_group.rw_torque'][2, :])
plt.show()

plt.plot(sim['optics_cubesat_group.body_torque'][0, :])
plt.plot(sim['optics_cubesat_group.body_torque'][1, :])
plt.plot(sim['optics_cubesat_group.body_torque'][2, :])
plt.show()

plt.plot(sim['optics_cubesat_group.battery_power'])
plt.show()

# Requires battery pack
# plt.plot(sim['optics_cubesat_group.soc'])
# plt.show()

# plt.plot(sim['optics_cubesat_group.voltage'])
# plt.show()
exit()
# NOTE: These must be cleaned up before using lsdo_dashboard
f = open('/Users/victor/owncloud/sandbox/python/output_variables.txt', 'w')
sim.prob.model.list_outputs(prom_name=True, out_stream=f)
f = open('/Users/victor/owncloud/sandbox/python/input_variables.txt', 'w')
sim.prob.model.list_inputs(prom_name=True, out_stream=f)
f = open('/Users/victor/owncloud/sandbox/python/design_variables.txt', 'w')
# NOTE: These do not return promoted names
f.write(repr(list(sim.prob.model.get_design_vars(use_prom_ivc=False).keys())))
f = open('/Users/victor/owncloud/sandbox/python/response_variables.txt', 'w')
f.write(repr(list(sim.prob.model.get_responses(use_prom_ivc=False).keys())))
f = open('/Users/victor/owncloud/sandbox/python/constraint_variables.txt', 'w')
f.write(repr(list(sim.prob.model.get_constraints().keys())))
exit()

sim.run(time_run=True)
sim.run(time_run=True)

print(sim['reference_orbit_state_km'][:3, :])
print(sim['optics_cubesat_group.soc'])
print(sim['detector_cubesat_group.soc'])

print(sim['optics_cubesat_group.sun_LOS'])
print(sim['detector_cubesat_group.sun_LOS'])
exit()
import matplotlib.pyplot as plt
# TODO: use ModOpt
# rely on OpenMDAO to solve optimization problem
prob = sim.prob
from openmdao.api import ScipyOptimizeDriver

prob.driver = ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.opt_settings['Major feasibility tolerance'] = 1e-7
prob.driver.opt_settings['Major optimality tolerance'] = 1e-7
prob.driver.opt_settings['Iterations limit'] = 500000000
prob.driver.opt_settings['Major iterations limit'] = 1000000
prob.driver.opt_settings['Minor iterations limit'] = 500000
# prob.driver.opt_settings['Iterations limit'] = 3
# prob.driver.opt_settings['Major iterations limit'] = 3
# prob.driver.opt_settings['Minor iterations limit'] = 1

# # print(prob['total_data_downloaded'])
prob.setup(check=True)
print('setup complete')
# prob.model.list_inputs()
# prob.model.list_outputs()
# prob.model.swarm_group.sunshade_cubesat_group.list_outputs(prom_name=True)
# prob.mode = 'run_model'
# prob.run()
# print('first run complete')
# prob.mode = 'run_driver'
# prob.run()
# print('optimization converged')

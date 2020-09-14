import numpy as np
import openmdao.api as om

from openmdao.api import pyOptSparseDriver, ExecComp
import openmdao.api as om
from lsdo_viz.api import Problem

from lsdo_cubesat.api import Swarm, Cubesat, SwarmGroup
from lsdo_cubesat.communication.ground_station import Ground_station

num_times = 1501
num_cp = 300
step_size = 95 * 60 / (num_times - 1)

if 0:
    num_times = 30
    num_cp = 3
    # step_size = 50.
    step_size = 1e-5

swarm = Swarm(
    num_times=num_times,
    num_cp=num_cp,
    step_size=step_size,
    cross_threshold=0.882,
)

# initial_orbit_state_magnitude = np.array([0.001] * 3 + [0.001] * 3)

initial_orbit_state_magnitude = np.array([1e-3] * 3 + [1e-3] * 3)

np.random.seed(6)
# A = np.random.rand(6)

print(initial_orbit_state_magnitude * np.random.rand(6))

Cubesat_sunshade = Cubesat(
    name='sunshade',
    dry_mass=1.3,
    initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
    approx_altitude_km=500.,
    specific_impulse=47.,
    apogee_altitude=500.001,
    perigee_altitude=499.99,
)

Cubesat_optics = Cubesat(
    name='optics',
    dry_mass=1.3,
    initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
    approx_altitude_km=500.,
    specific_impulse=47.,
    perigee_altitude=500.,
    apogee_altitude=500.,
)

Cubesat_detector = Cubesat(
    name='detector',
    dry_mass=1.3,
    initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
    approx_altitude_km=500.,
    specific_impulse=47.,
    perigee_altitude=500.002,
    apogee_altitude=499.98,
)

Cubesat_sunshade.add(
    Ground_station(
        name='UCSD',
        lon=-117.1611,
        lat=32.7157,
        alt=0.4849,
    ))
Cubesat_sunshade.add(
    Ground_station(
        name='UIUC',
        lon=-88.2272,
        lat=32.8801,
        alt=0.2329,
    ))
Cubesat_sunshade.add(
    Ground_station(
        name='Georgia',
        lon=-84.3963,
        lat=33.7756,
        alt=0.2969,
    ))
Cubesat_sunshade.add(
    Ground_station(
        name='Montana',
        lon=-109.5337,
        lat=33.7756,
        alt=1.04,
    ))

Cubesat_detector.add(
    Ground_station(
        name='UCSD',
        lon=-117.1611,
        lat=32.7157,
        alt=0.4849,
    ))
Cubesat_detector.add(
    Ground_station(
        name='UIUC',
        lon=-88.2272,
        lat=32.8801,
        alt=0.2329,
    ))
Cubesat_detector.add(
    Ground_station(
        name='Georgia',
        lon=-84.3963,
        lat=33.7756,
        alt=0.2969,
    ))
Cubesat_detector.add(
    Ground_station(
        name='Montana',
        lon=-109.5337,
        lat=33.7756,
        alt=1.04,
    ))

Cubesat_optics.add(
    Ground_station(
        name='UCSD',
        lon=-117.1611,
        lat=32.7157,
        alt=0.4849,
    ))
Cubesat_optics.add(
    Ground_station(
        name='UIUC',
        lon=-88.2272,
        lat=32.8801,
        alt=0.2329,
    ))
Cubesat_optics.add(
    Ground_station(
        name='Georgia',
        lon=-84.3963,
        lat=33.7756,
        alt=0.2969,
    ))
Cubesat_optics.add(
    Ground_station(
        name='Montana',
        lon=-109.5337,
        lat=33.7756,
        alt=1.04,
    ))

swarm.add(Cubesat_sunshade)
swarm.add(Cubesat_optics)
swarm.add(Cubesat_detector)

prob = Problem()
prob.swarm = swarm

swarm_group = SwarmGroup(swarm=swarm)
prob.model.add_subsystem('swarm_group', swarm_group, promotes=['*'])

# # obj_comp = ExecComp(
# #     'obj= 0.01 * total_propellant_used- 0.001 * total_data_downloaded + 1e-4 * (0'
# #     '+ masked_normal_distance_sunshade_detector_mm_sq_sum'
# #     '+ masked_normal_distance_optics_detector_mm_sq_sum'
# #     '+ masked_distance_sunshade_optics_mm_sq_sum'
# #     '+ masked_distance_optics_detector_mm_sq_sum'
# #     '+ sunshade_cubesat_group_relative_orbit_state_sq_sum'
# #     '+ optics_cubesat_group_relative_orbit_state_sq_sum'
# #     '+ detector_cubesat_group_relative_orbit_state_sq_sum'
# #     ') / {}'.format(num_times))

obj_comp = ExecComp(
    'obj= 0.01 * total_propellant_used- 1e-5 * total_data_downloaded + 1e-4 * (0'
    '+ masked_normal_distance_sunshade_detector_mm_sq_sum'
    '+ masked_normal_distance_optics_detector_mm_sq_sum'
    '+ masked_distance_sunshade_optics_mm_sq_sum'
    '+ masked_distance_optics_detector_mm_sq_sum)/{}'
    '+ 1e-3 * (sunshade_cubesat_group_relative_orbit_state_sq_sum'
    '+ optics_cubesat_group_relative_orbit_state_sq_sum'
    '+ detector_cubesat_group_relative_orbit_state_sq_sum'
    ') / {}'.format(num_times, num_times))

obj_comp.add_objective('obj', scaler=1.e-3)
# obj_comp.add_objective('obj')
prob.model.add_subsystem('obj_comp', obj_comp, promotes=['*'])
for cubesat_name in ['sunshade', 'optics', 'detector']:
    prob.model.connect(
        '{}_cubesat_group.relative_orbit_state_sq_sum'.format(cubesat_name),
        '{}_cubesat_group_relative_orbit_state_sq_sum'.format(cubesat_name),
    )

prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'
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
# prob.model.list_inputs()
# prob.model.list_outputs()
# prob.model.swarm_group.sunshade_cubesat_group.list_outputs(prom_name=True)
prob.list_problem_vars()
print('setup complete')
# prob.run_driver()
prob.mode = 'run_driver'
prob.run()
# prob.run_model()
# prob.check_partials(compact_print=True)

# import matplotlib.pyplot as plt

# for sc in ['sunshade', 'optics', 'detector']:

#     print('ns')
#     print(prob[sc + '_cubesat_group.num_series'])
#     print('np')
#     print(prob[sc + '_cubesat_group.num_parallel'])
#     print('total_propellant_used')
#     print(prob[sc + '_cubesat_group.total_propellant_used'])

#     # print(prob['_cubesat_group.external_torques_x_cp'])
#     # print(prob['_cubesat_group.external_torques_y_cp'])
#     # print(prob['_cubesat_group.external_torques_z_cp'])
#     # plt.plot(prob[sc + '_cubesat_group.thrust_scalar_mN_cp'])
#     # plt.title(sc + ' thrust scalar')
#     # plt.show()

#     # plt.plot(prob[sc + '_cubesat_group.UCSD_comm_group.P_comm_cp'])
#     # plt.plot(prob[sc + '_cubesat_group.UIUC_comm_group.P_comm_cp'])
#     # plt.plot(prob[sc + '_cubesat_group.Georgia_comm_group.P_comm_cp'])
#     # plt.plot(prob[sc + '_cubesat_group.Montana_comm_group.P_comm_cp'])
#     # plt.title(sc + ' P_comm_cp')
#     # plt.show()

#     # ux = prob[sc + '_cubesat_group.external_torques_x']
#     # uy = prob[sc + '_cubesat_group.external_torques_y']
#     # uz = prob[sc + '_cubesat_group.external_torques_z']
#     # plt.plot(ux)
#     # plt.plot(uy)
#     # plt.plot(uz)
#     # plt.title(sc + ' external torques')
#     # plt.show()

#     # ux = prob[sc + '_cubesat_group.external_torques_x_cp']
#     # uy = prob[sc + '_cubesat_group.external_torques_y_cp']
#     # uz = prob[sc + '_cubesat_group.external_torques_z_cp']
#     # plt.plot(ux)
#     # plt.plot(uy)
#     # plt.plot(uz)
#     # plt.title(sc + ' external torques (ctrl pts)')
#     # plt.show()

#     # roll = prob[sc + '_cubesat_group.roll']
#     # pitch = prob[sc + '_cubesat_group.pitch']
#     # yaw = prob[sc + '_cubesat_group.yaw']
#     # plt.plot(roll)
#     # plt.plot(pitch)
#     # plt.plot(yaw)
#     # # plt.title(sc + ' roll and pitch')
#     # plt.title(sc + ' roll, pitch, and yaw')
#     # plt.show()

#     # orbit = prob[sc + '_cubesat_group.orbit_state_km'][:3, :]
#     # plt.plot(orbit[0, :])
#     # plt.plot(orbit[1, :])
#     # plt.plot(orbit[2, :])
#     # plt.title(sc + ' orbit x,y,z')
#     # plt.show()

#     # soc = prob[sc + '_cubesat_group.cell_model.soc']
#     # plt.plot(soc.flatten())
#     # plt.title(sc + ' soc')
#     # plt.show()
#     # print(soc.shape)
#     # print(soc)

#     print(prob['obj'])
#     print('total_data_downloaded')
#     print(prob['total_data_downloaded'])

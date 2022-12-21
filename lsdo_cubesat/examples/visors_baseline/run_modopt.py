import numpy as np

from lsdo_cubesat.specifications.swarm_spec import SwarmSpec
from lsdo_cubesat.examples.visors_utils.generate_reference_orbit import generate_reference_orbit
from lsdo_cubesat.examples.visors_baseline.make_swarm import make_swarm

from python_csdl_backend import Simulator
from lsdo_cubesat.examples.visors_baseline.dashboard import Dash

from modopt.csdl_library import CSDLProblem
from modopt.snopt_library import SNOPT

np.random.seed(0)


def get_data(file_name):
    """
    return the python object saved by user in save_python_object.

    Parameters:
    -----------
        file_name:
            name of file saved using save_python_object
    """
    import pickle
    with open(file_name, 'rb') as handle:
        data_dict = pickle.load(handle)

    return data_dict


warm_start = False
num_times = 301
num_cp = int((num_times - 1) / 5)
duration = 95.
step_size = duration * 60 / (num_times - 1)

# t = (np.random.rand(num_cp * 3).reshape((num_cp, 3)) - 0.5) * 1e-9
t = np.zeros((num_cp, 3))
optics_thrust_cp = t
detector_thrust_cp = t
optics_initial_propellant_mass = 0.17
detector_initial_propellant_mass = 0.17
optics_accel_cp = 0
detector_accel_cp = 0

# orient s/c to face sun
# optics cubesat orients so that -y-axis is aligned with sun
# detector cubesat orients so that -x-axis is aligned with sun
# -optics_B_from_ECI[:,0,:][1, :] -> -optics_B_from_ECI[1,0,:]
# -(sr * sp * cy - cr * sy) == 1
# -detector_B_from_ECI[:,0,:][0, :] -> -detector_B_from_ECI[0,0,:]
# -(cp * cy) == 1
# cos(pitch) * cos(yaw) = -1
optics_cubesat_yaw_cp = np.ones(num_cp) * np.pi / 2
optics_cubesat_pitch_cp = np.zeros(num_cp)
optics_cubesat_roll_cp = np.zeros(num_cp)
detector_cubesat_yaw_cp = np.ones(num_cp) * np.pi
detector_cubesat_pitch_cp = np.zeros(num_cp)
detector_cubesat_roll_cp = np.zeros(num_cp)

from accel_cp import optics_acceleration_due_to_thrust_cp, detector_acceleration_due_to_thrust_cp

# load data to warm start problem
if warm_start is True:
    warm_start_data = get_data(
        '/Users/victor/packages/lsdo_cubesat/lsdo_cubesat/GOOD-CANDIDATES/2022-12-07-16_16_36/_data/data_entry.simulator.2022-12-07-21_08_49_879394.pkl'
    )

    optics_accel_cp = optics_acceleration_due_to_thrust_cp
    detector_accel_cp = detector_acceleration_due_to_thrust_cp
    # optics_thrust_cp = warm_start_data['optics_cubesat.thrust_cp']
    # detector_thrust_cp = warm_start_data['detector_cubesat.thrust_cp']
    # optics_initial_propellant_mass = warm_start_data[
    #     'optics_cubesat.initial_propellant_mass']
    # detector_initial_propellant_mass = warm_start_data[
    #     'detector_cubesat.initial_propellant_mass']
ref_orbit, ax = generate_reference_orbit(
    num_times,
    step_size,
    plot=False,
)
rep = make_swarm(
    SwarmSpec(
        num_times=num_times,
        duration=duration,
        num_cp=num_cp,
        step_size=step_size,
        cross_threshold=7.0,
        # cross_threshold=0.2,
        # cross_threshold=0.857,
        # cross_threshold=0.2,
        # cross_threshold=-0.9,
        # cross_threshold=0.9,
    ))

# a, b = rep.influences()
# print(a)
# print(b)
# exit()

sim = Simulator(rep)
sim['reference_orbit_state_km'] = ref_orbit
sim.run()

# set inputs
# sim['optics_cubesat.acceleration_due_to_thrust_cp'] = optics_accel_cp
# sim['detector_cubesat.acceleration_due_to_thrust_cp'] = detector_accel_cp
# sim['detector_cubesat.thrust_cp'] = 1.3 * optics_accel_cp
# sim['detector_cubesat.thrust_cp'] = 1.3 * detector_accel_cp
# sim['optics_cubesat.initial_propellant_mass'] = optics_initial_propellant_mass
# sim['detector_cubesat.initial_propellant_mass'] = detector_initial_propellant_mass
# sim['optics_cubesat.yaw_cp'] = optics_cubesat_yaw_cp
# sim['optics_cubesat.pitch_cp'] = optics_cubesat_pitch_cp
# sim['optics_cubesat.roll_cp'] = optics_cubesat_roll_cp
# sim['detector_cubesat.yaw_cp'] = detector_cubesat_yaw_cp
# sim['detector_cubesat.pitch_cp'] = detector_cubesat_pitch_cp
# sim['detector_cubesat.roll_cp'] = detector_cubesat_roll_cp

sim.add_recorder(Dash().get_recorder())
sim.run()
# sim.check_totals(compact_print=True, step=1e-8)
# exit()

# import matplotlib.pyplot as plt
# plt.subplot(411)
# plt.plot(sim['telescope_vector'][:,0], 'x')
# plt.plot(sim['telescope_vector'][:,1], 'o')
# plt.plot(sim['telescope_vector'][:,2], '.')
# plt.subplot(412)
# plt.plot(sim['telescope_vector_component_in_sun_direction'][:,0], 'x')
# plt.plot(sim['telescope_vector_component_in_sun_direction'][:,1], 'o')
# plt.plot(sim['telescope_vector_component_in_sun_direction'][:,2], '.')
# plt.subplot(413)
# plt.plot(sim['telescope_direction_in_view_plane'][:,0], 'x')
# plt.plot(sim['telescope_direction_in_view_plane'][:,1], 'o')
# plt.plot(sim['telescope_direction_in_view_plane'][:,2], '.')
# plt.subplot(414)
# plt.plot(sim['view_plane_error'])
# plt.show()
# plt.plot(sim['telescope_cos_view_angle'])
# plt.show()
# exit()

prob = CSDLProblem(
    problem_name='simple_optimal_ctrl',
    simulator=sim,
)

optimizer = SNOPT(
    prob,
    Major_iterations=10000,
    Major_optimality=1e-6,
    Major_feasibility=1e-3,
)

# Solve your optimization problem
optimizer.solve()

# import matplotlib.pyplot as plt
# plt.subplot(411)
# plt.plot(sim['telescope_vector'][:,0], 'x')
# plt.plot(sim['telescope_vector'][:,1], 'o')
# plt.plot(sim['telescope_vector'][:,2], '.')
# plt.subplot(412)
# plt.plot(sim['telescope_vector_component_in_sun_direction'][:,0], 'x')
# plt.plot(sim['telescope_vector_component_in_sun_direction'][:,1], 'o')
# plt.plot(sim['telescope_vector_component_in_sun_direction'][:,2], '.')
# plt.subplot(413)
# plt.plot(sim['telescope_direction_in_view_plane'][:,0], 'x')
# plt.plot(sim['telescope_direction_in_view_plane'][:,1], 'o')
# plt.plot(sim['telescope_direction_in_view_plane'][:,2], '.')
# plt.subplot(414)
# plt.plot(sim['view_plane_error'])
# plt.show()
# exit()

# use this to find the NaN values
print(sim.keys())
# for k, v in sim.items():
#     if np.any(np.isnan(v)):
#         print(k, v)

print('optics_cubesat.acceleration_due_to_thrust',
      sim['optics_cubesat.acceleration_due_to_thrust'])
print('detector_cubesat.acceleration_due_to_thrust',
      sim['detector_cubesat.acceleration_due_to_thrust'])
print('optics_velocity', sim['optics_relative_orbit_state_m'][:, 3:])
print('detector_velocity', sim['detector_relative_orbit_state_m'][:, 3:])
print('optics_observation_dot', sim['optics_observation_dot'])
print('detector_observation_dot', sim['detector_observation_dot'])
print('sun_direction', sim['sun_direction'])
print('observation_phase_indicator', sim['observation_phase_indicator'])
print('telescope_view_angle', sim['telescope_view_angle'])
print('max_telescope_view_angle', sim['max_telescope_view_angle'])

show_optimization_results = True
for k, v in sim.items():
    if np.any(np.isnan(v)):
        show_optimization_results = False
        print(k, 'contains NaN values')

# Print results of optimization
# if show_optimization_results is True:
#     optimizer.print_results()

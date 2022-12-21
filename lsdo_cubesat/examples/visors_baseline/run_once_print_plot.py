import numpy as np
import matplotlib.pyplot as plt

from lsdo_cubesat.specifications.swarm_spec import SwarmSpec
from lsdo_cubesat.examples.visors_baseline.plot_sim_state import plot_sim_state
from lsdo_cubesat.examples.visors_utils.generate_reference_orbit import generate_reference_orbit
from lsdo_cubesat.examples.visors_baseline.make_swarm import make_swarm

from python_csdl_backend import Simulator

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
    num_times = 60
    num_cp = int(num_times / 10) * 4
else:
    num_times = 40
    num_cp = 5
step_size = duration * 60 / (num_times - 1)

ref_orbit, ax = generate_reference_orbit(
    num_times,
    step_size,
    plot=True,
)
rep = make_swarm(
    SwarmSpec(
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

sim = Simulator(rep)
sim['reference_orbit_state_km'] = ref_orbit
sim.run()

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
print('optics_cubesat.min_altitude_km', sim['optics_cubesat.min_altitude_km'])
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

# plot sun direction
sd = sim['sun_direction']

ax.plot(
    sd[0, :] * 6000,
    sd[1, :] * 6000,
    sd[2, :] * 6000,
    'o',
)
plt.show()

plot_sim_state(sim)

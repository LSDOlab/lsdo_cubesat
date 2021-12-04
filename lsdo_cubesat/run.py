import numpy as np

from lsdo_cubesat.api import SwarmParams, CubesatParams, Swarm
from lsdo_cubesat.communication.ground_station import GroundStationParams

num_times = 15
num_cp = 3
step_size = 95 * 60 / (num_times - 1)

swarm = SwarmParams(
    num_times=num_times,
    num_cp=num_cp,
    step_size=step_size,
    cross_threshold=0.882,
)

# initial_orbit_state_magnitude = np.array([0.001] * 3 + [0.001] * 3)

initial_orbit_state_magnitude = np.array([1e-3] * 3 + [1e-3] * 3)

# np.random.seed(6)
# A = np.random.rand(6)

# print(initial_orbit_state_magnitude * np.random.rand(6))

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
    initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
    approx_altitude_km=500.,
    specific_impulse=47.,
    perigee_altitude=500.,
    apogee_altitude=500.,
)

cubesats['detector'] = CubesatParams(
    name='detector',
    dry_mass=1.3,
    initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
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

cubesats['detector'].add(
    GroundStationParams(
        name='UCSD',
        lon=-117.1611,
        lat=32.7157,
        alt=0.4849,
    ))
cubesats['detector'].add(
    GroundStationParams(
        name='UIUC',
        lon=-88.2272,
        lat=32.8801,
        alt=0.2329,
    ))
cubesats['detector'].add(
    GroundStationParams(
        name='Georgia',
        lon=-84.3963,
        lat=33.7756,
        alt=0.2969,
    ))
cubesats['detector'].add(
    GroundStationParams(
        name='Montana',
        lon=-109.5337,
        lat=33.7756,
        alt=1.04,
    ))

cubesats['optics'].add(
    GroundStationParams(
        name='UCSD',
        lon=-117.1611,
        lat=32.7157,
        alt=0.4849,
    ))
cubesats['optics'].add(
    GroundStationParams(
        name='UIUC',
        lon=-88.2272,
        lat=32.8801,
        alt=0.2329,
    ))
cubesats['optics'].add(
    GroundStationParams(
        name='Georgia',
        lon=-84.3963,
        lat=33.7756,
        alt=0.2969,
    ))
cubesats['optics'].add(
    GroundStationParams(
        name='Montana',
        lon=-109.5337,
        lat=33.7756,
        alt=1.04,
    ))

for v in cubesats.values():
    swarm.add(v)


# compile
def make_swarm(swarm):
    from csdl_om import Simulator
    swarm_group = Swarm(swarm=swarm)
    return Simulator(swarm_group)


sim = make_swarm(swarm)
sim.visualize_implementation()
exit()

sim.run()
import matplotlib.pyplot as plt

plt.plot(sim['optics_cubesat_group.soc'])
plt.plot(sim['detector_cubesat_group.soc'])
plt.show()
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

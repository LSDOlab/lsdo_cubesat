from re import T
import numpy as np
import time

from lsdo_cubesat.parameters.swarm import SwarmParams
from lsdo_cubesat.parameters.cubesat import CubesatParams
from lsdo_cubesat.examples.visors.swarm_group import Swarm
from lsdo_cubesat.orbit.reference_orbit_group import ReferenceOrbit

from csdl_om import Simulator

# from openmdao.api import pyOptSparseDriver
from lsdo_cubesat.examples.visors.initial_plots import plot_initial
from lsdo_cubesat.examples.visors.save_data import save_data
from openmdao.api import ScipyOptimizeDriver


def make_swarm(swarm):

    # initial state relative to reference orbit
    initial_orbit_state = np.array([1e-3] * 3 + [1e-3] * 3)

    np.random.seed(0)

    cubesats = dict()

    cubesats['optics'] = CubesatParams(
        name='optics',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state * np.random.rand(6) * 1e-3,
        approx_altitude_km=500.,
        specific_impulse=47.,
        perigee_altitude=500.,
        apogee_altitude=500.,
    )

    cubesats['detector'] = CubesatParams(
        name='detector',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state * np.random.rand(6) * 1e-3,
        approx_altitude_km=500.,
        specific_impulse=47.,
        perigee_altitude=500.002,
        apogee_altitude=499.98,
    )

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


def generate_reference_orbit(num_times, step_size):
    ref_orbit = Simulator(
        ReferenceOrbit(
            num_times=num_times,
            step_size=step_size,
        ))
    ref_orbit.run()
    return ref_orbit['reference_orbit_state_km']


# if True:
if False:
    num_times = 1501
    num_cp = 30
# elif False:
elif True:
    num_times = 40
    num_cp = 5
else:
    num_times = 4
    num_cp = 3

step_size = 95 * 60 / (num_times - 1)

ref_orbit = generate_reference_orbit(num_times, step_size)
sim = make_swarm(
    SwarmParams(
        num_times=num_times,
        num_cp=num_cp,
        step_size=step_size,
        # cross_threshold=0.882,
        # cross_threshold=0.857,
        cross_threshold=0.2,
        # cross_threshold=-0.9,
        # cross_threshold=0.9,
    ))
sim['reference_orbit_state_km'] = ref_orbit
plot_initial(sim, threeD=True)
plot_initial(sim, threeD=False)
save_data(sim)
optimize = True
if optimize == True:
    # driver = sim.prob.driver = pyOptSparseDriver()
    # driver.options['optimizer'] = 'SNOPT'
    sim.prob.driver = ScipyOptimizeDriver()
    sim.prob.driver.options['optimizer'] = 'SLSQP'
    sim.prob.driver.options['tol'] = 1e-6
    sim.prob.driver.options['disp'] = True
    sim.prob.run_driver()

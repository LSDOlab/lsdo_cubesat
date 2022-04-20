from lsdo_cubesat.parameters.cubesat import CubesatParams
from lsdo_cubesat.examples.visors_baseline.telescope import Telescope

from csdl_om import Simulator
import numpy as np
import time


def make_swarm(swarm):

    # initial state relative to reference orbit
    initial_orbit_state = np.array([1e-3] * 3 + [1e-3] * 3)

    np.random.seed(0)

    cubesats = dict()

    cubesats['optics'] = CubesatParams(
        name='optics',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state * np.random.rand(6) * 1e-3,
        specific_impulse=47.,
        perigee_altitude=500.,
        apogee_altitude=500.,
    )

    cubesats['detector'] = CubesatParams(
        name='detector',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state * np.random.rand(6) * 1e-3,
        specific_impulse=47.,
        perigee_altitude=500.002,
        apogee_altitude=499.98,
    )

    for v in cubesats.values():
        swarm.add(v)
    m = Telescope(swarm=swarm)
    start_compile = time.time()
    sim = Simulator(m)
    end_compile = time.time()
    total_compile_time = end_compile - start_compile
    print('========== TOTAL COMPILE TIME ==========')
    print(total_compile_time, 's')
    print('=========================================')
    return sim

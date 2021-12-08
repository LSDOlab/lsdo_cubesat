import numpy as np
import time

from lsdo_cubesat.parameters.swarm import SwarmParams
from lsdo_cubesat.parameters.cubesat import CubesatParams
from lsdo_cubesat.swarm_group import Swarm
from lsdo_cubesat.orbit.reference_orbit_group import ReferenceOrbit

from csdl_om import Simulator


def make_swarm(swarm):

    # initial state relative to reference orbit -- same for all s/c
    initial_orbit_state = np.array([1e-3] * 3 + [1e-3] * 3)

    np.random.seed(0)

    cubesats = dict()

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


def split_top_level_low_level_varnames(t, l, s):
    try:
        _ = s.rindex('.')
        l.append(s)
    except:
        t.append(s)


def remove_automatically_named_variables(t, l):
    remove = []
    for s in l:
        try:
            _ = s.rindex('._')
            remove.append(s)
        except:
            pass
    for r in remove:
        l.remove(r)

    remove = []
    for s in t:
        if s[0] == '_':
            remove.append(s)
    for r in remove:
        t.remove(r)


num_times = 1501
num_cp = 30
step_size = 95 * 60 / (num_times - 1)

swarm = SwarmParams(
    num_times=num_times,
    num_cp=num_cp,
    step_size=step_size,
    cross_threshold=0.857,
)

sim = make_swarm(swarm)
sim['reference_orbit_state_km'] = generate_reference_orbit(
    num_times, step_size)

# save data
a = sim.prob.model.list_outputs(out_stream=None)
all_variable_names = []
for (k, v) in a:
    all_variable_names.append(v['prom_name'])
t = []
l = []
_ = [split_top_level_low_level_varnames(t, l, s) for s in all_variable_names]
remove_automatically_named_variables(t, l)
varnames = t + l
from lsdo_cubesat.dash import Dash

dashboard = Dash(varnames)
sim.add_recorder(dashboard.get_recorder())

# run optimization
from openmdao.api import pyOptSparseDriver

driver = sim.prob.driver = pyOptSparseDriver()
driver.options['optimizer'] = 'SNOPT'
sim.prob.run_driver()

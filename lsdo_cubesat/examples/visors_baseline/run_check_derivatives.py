import numpy as np
import time

from lsdo_cubesat.specifications.swarm_spec import SwarmSpec
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
step_size *= 1e-10 if step_size > 1e-4 else step_size
print('TIME STEP SIZE', step_size)

method = 'fd'
# method = 'cs'
step = 1e-4
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
        cross_threshold=0.7,
        # cross_threshold=0.857,
        # cross_threshold=0.2,
        # cross_threshold=-0.9,
        # cross_threshold=0.9,
    ))

sim = Simulator(rep)
sim['reference_orbit_state_km'] = ref_orbit

start = time.time()
sim.run()
end = time.time()

print('time to run once', end - start)
if method == 'fd':
    sim.check_totals(compact_print=True, method=method, step=fd_step)
else:
    sim.check_totals(compact_print=True, method=method)

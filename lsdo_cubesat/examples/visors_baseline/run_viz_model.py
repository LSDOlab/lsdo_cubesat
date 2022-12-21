import numpy as np

from lsdo_cubesat.specifications.swarm_spec import SwarmSpec
from lsdo_cubesat.examples.visors_utils.generate_reference_orbit import generate_reference_orbit
from lsdo_cubesat.examples.visors_baseline.make_swarm import make_swarm

from csdl_om import Simulator
# from python_csdl_backend import Simulator

np.random.seed(0)

visualize_recursive = False

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
print(ref_orbit.shape)

sim.run()

sim.executable.list_problem_vars()
sim.executable.model.list_outputs()
sim.visualize_implementation(recursive=visualize_recursive)
for k, v in sim.items():
    if np.any(np.isnan(v)):
        print(k, 'contains NaN values')

# telescope_direction
# telescope_cos_view_angle

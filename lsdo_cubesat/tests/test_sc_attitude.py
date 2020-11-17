from openmdao.api import Problem, Group
from openmdao.api import IndepVarComp
from lsdo_cubesat.utils.random_arrays import make_random_bounded_array
from lsdo_cubesat.attitude.new.attitude_rk4_gravity_comp import AttitudeRK4GravityComp
import numpy as np

np.random.seed(0)
num_times = 100
step_size = 1e-8
I = np.array([90, 100, 80])
wq0 = np.random.rand(7) - 0.5
wq0[3:] /= np.linalg.norm(wq0[3:])

comp = IndepVarComp()
comp.add_output('initial_angular_velocity_orientation', val=wq0)
comp.add_output(
    'external_torques_x',
    val=make_random_bounded_array(num_times, bound=1).reshape((1, num_times)),
    shape=(1, num_times),
)
comp.add_output(
    'external_torques_y',
    val=make_random_bounded_array(num_times, bound=1).reshape((1, num_times)),
    shape=(1, num_times),
)
comp.add_output(
    'external_torques_z',
    val=make_random_bounded_array(num_times, bound=1).reshape((1, num_times)),
    shape=(1, num_times),
)
prob = Problem()
prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])
prob.model.add_subsystem(
    'comp',
    AttitudeRK4GravityComp(
        num_times=num_times,
        step_size=step_size,
        moment_inertia_ratios=np.array([2.0 / 3.0, -2.0 / 3.0, 0]),
    ),
    promotes=['*'],
)

prob.setup(check=True, force_alloc_complex=True)
cp = prob.check_partials(compact_print=True)


# pytest
def test_fn():
    from openmdao.utils.assert_utils import assert_check_partials
    assert_check_partials(cp, atol=1.e-6, rtol=1.e-6)

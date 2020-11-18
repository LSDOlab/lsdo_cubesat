from openmdao.api import Problem, IndepVarComp
from lsdo_cubesat.orbit.orbit_angular_speed_group import OrbitAngularSpeedGroup
import numpy as np

np.random.seed(0)
num_times = 100

leo = np.abs(np.random.rand(3, num_times)) * 10 + 6371 + 150

comp = IndepVarComp()
comp.add_output('position_km', val=leo)
comp.add_output('velocity_km_s', val=np.random.rand(3, num_times))

prob = Problem()
prob.model.add_subsystem(
    'indeps',
    comp,
    promotes=['*'],
)
prob.model.add_subsystem(
    'orbit_angular_speed_group',
    OrbitAngularSpeedGroup(num_times=num_times),
    promotes=['*'],
)
prob.setup()
prob.run_model()
cp = prob.check_partials(compact_print=True)


# pytest
def test_fn():
    from openmdao.utils.assert_utils import assert_check_partials
    assert_check_partials(cp, atol=1.e-6, rtol=1.e-6)

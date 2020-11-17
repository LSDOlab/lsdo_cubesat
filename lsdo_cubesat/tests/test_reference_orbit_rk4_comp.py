from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp
import numpy as np
from lsdo_cubesat.orbit.reference_orbit_rk4_comp import ReferenceOrbitRK4Comp

np.random.seed(0)

group = Group()

comp = IndepVarComp()
n = 1500
m = 1
npts = 1
h = 1.5e-4

r_e2b_I0 = np.empty(6)
r_e2b_I0[:3] = 1000. * np.random.rand(3)
r_e2b_I0[3:] = 1. * np.random.rand(3)

thrust_ECI = np.random.rand(3, n)
mass = np.random.rand(1, n)

comp.add_output('force_3xn', val=thrust_ECI)
comp.add_output('initial_orbit_state_km', val=r_e2b_I0)
comp.add_output('mass', val=mass)

group.add_subsystem('inputs_comp', comp, promotes=['*'])

comp = ReferenceOrbitRK4Comp(num_times=n, step_size=h)
group.add_subsystem('comp', comp, promotes=['*'])
prob = Problem()
prob.model = group
prob.setup(check=True)
cp = prob.check_partials(compact_print=True)


# pytest
def test_fn():
    from openmdao.utils.assert_utils import assert_check_partials
    assert_check_partials(cp, atol=1.e-6, rtol=1.e-6)

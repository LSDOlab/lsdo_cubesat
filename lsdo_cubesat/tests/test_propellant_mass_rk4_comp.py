from openmdao.api import Problem
from openmdao.api import IndepVarComp
from lsdo_cubesat.propulsion.propellant_mass_rk4_comp import PropellantMassRK4Comp
import numpy as np

np.random.seed(0)

comp = IndepVarComp()
n = 20
h = 6000.

dm_dt = np.random.rand(1, n)
Mass0 = np.random.rand(1)
comp.add_output('num_times', val=n)
comp.add_output('mass_flow_rate', val=dm_dt)
comp.add_output('initial_propellant_mass', val=Mass0)
prob = Problem()
prob.model.add_subsystem('Inputcomp', comp, promotes=['*'])
prob.model.add_subsystem(
    'Statecomp_Implicit',
    PropellantMassRK4Comp(num_times=n, step_size=h),
    promotes=['*'],
)
prob.setup()
prob.check_partials(compact_print=True)


# pytest
def test_fn():
    from openmdao.utils.assert_utils import assert_check_partials
    assert_check_partials(cp, atol=1.e-6, rtol=1.e-6)

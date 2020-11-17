from openmdao.api import Problem, IndepVarComp
from lsdo_cubesat.orbit.orbit_state_decomposition_comp import OrbitStateDecompositionComp
import numpy as np

num_times = 3

prob = Problem()
comp = IndepVarComp()
comp.add_output('orbit_state', np.random.rand(6, num_times))
prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

comp = OrbitStateDecompositionComp(num_times=num_times,
                                   orbit_state_name='orbit_state',
                                   position_name='position',
                                   velocity_name='velocity')
prob.model.add_subsystem('comp', comp, promotes=['*'])

prob.setup(check=True)
cp = prob.check_partials(compact_print=True)


# pytest
def test_fn():
    from openmdao.utils.assert_utils import assert_check_partials
    assert_check_partials(cp, atol=1.e-6, rtol=1.e-6)

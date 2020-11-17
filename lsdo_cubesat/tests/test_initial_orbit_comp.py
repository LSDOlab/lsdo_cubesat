from openmdao.api import Problem, Group, IndepVarComp
from lsdo_cubesat.orbit.initial_orbit_comp import InitialOrbitComp
import numpy as np

np.random.seed(0)

group = Group()

comp = IndepVarComp()

perigee_altitude = np.random.rand(1)
apogee_altitude = np.random.rand(1)
RAAN = np.random.rand(1)
inclination = np.random.rand(1)
argument_of_periapsis = np.random.rand(1)
true_anomaly = np.random.rand(1)

comp.add_output('perigee_altitude', val=perigee_altitude)
comp.add_output('apogee_altitude', val=apogee_altitude)
comp.add_output('RAAN', val=RAAN)
comp.add_output('inclination', val=inclination)
comp.add_output('argument_of_periapsis', val=argument_of_periapsis)
comp.add_output('true_anomaly', val=true_anomaly)

group.add_subsystem('Inputcomp', comp, promotes=['*'])

group.add_subsystem('Statecomp_Implicit', InitialOrbitComp(), promotes=['*'])

prob = Problem()
prob.model = group
prob.setup(check=True)
prob.run_model()
cp = prob.check_partials(compact_print=True)


# pytest
def test_fn():
    from openmdao.utils.assert_utils import assert_check_partials
    assert_check_partials(cp, atol=1.e-6, rtol=1.e-6)

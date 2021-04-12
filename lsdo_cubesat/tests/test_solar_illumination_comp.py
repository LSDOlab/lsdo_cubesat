from openmdao.api import Problem, IndepVarComp, Group
from lsdo_cubesat.solar.smt_exposure import smt_exposure
from lsdo_cubesat.utils.random_arrays import make_random_bounded_array
from lsdo_cubesat.solar.solar_illumination import SolarIllumination
import numpy as np
from lsdo_cubesat.examples.data.cubesat_xdata import cubesat_xdata as az
from lsdo_cubesat.examples.data.cubesat_ydata import cubesat_ydata as el
from lsdo_cubesat.examples.data.cubesat_zdata import cubesat_zdata as yt

np.random.seed(0)

times = 1500

# load training data
# az = np.genfromtxt(here + '/../examples/data/cubesat_xdata.csv', delimiter=',')
# el = np.genfromtxt(here + '/../examples/data/cubesat_ydata.csv', delimiter=',')
# yt = np.genfromtxt(here + '/../examples/data/cubesat_zdata.csv', delimiter=',')

# generate surrogate model with 20 training points
# must be the same as the number of points used to create model
sm = smt_exposure(20, az, el, yt)

roll = make_random_bounded_array(times, 10)
pitch = make_random_bounded_array(times, 10)

# check partials
ivc = IndepVarComp()
ivc.add_output('roll', val=roll)
ivc.add_output('pitch', val=pitch)
prob = Problem()
prob.model = Group()
prob.model.add_subsystem(
    'ivc',
    ivc,
    promotes=['*'],
)
prob.model.add_subsystem(
    'spm',
    SolarIllumination(num_times=times, sm=sm),
    promotes=['*'],
)
prob.setup()
cp = prob.check_partials(compact_print=True)


# pytest
def test_fn():
    from openmdao.utils.assert_utils import assert_check_partials
    assert_check_partials(cp, atol=1.e-6, rtol=1.e-6)

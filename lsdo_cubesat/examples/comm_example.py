import matplotlib.pyplot as plt
import numpy as np
from openmdao.api import IndepVarComp, Problem

from lsdo_cubesat.api import Cubesat
from lsdo_cubesat.attitude.attitude_group import AttitudeGroup
from lsdo_cubesat.communication.comm_group import CommGroup
from lsdo_cubesat.options.ground_station import GroundStation
from lsdo_cubesat.orbit.orbit_group import OrbitGroup
from lsdo_cubesat.utils.api import (ArrayExpansionComp, BsplineComp,
                                    ElementwiseMaxComp, LinearCombinationComp,
                                    PowerCombinationComp, get_bspline_mtx)

prob = Problem()
num_times = 20
step_size = 0.1
num_cp = 5
mtx = get_bspline_mtx(num_cp, num_times, order=4)
initial_orbit_state_magnitude = np.array([1e-3] * 3 + [1e-3] * 3)
cubesat = Cubesat(
    name='optics',
    dry_mass=1.3,
    initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
    approx_altitude_km=500.,
    specific_impulse=47.,
    perigee_altitude=500.,
    apogee_altitude=500.,
)

# prob.model.add_subsystem(
#     'orbit',
#     OrbitGroup(
#         num_times=num_times,
#         num_cp=num_cp,
#         step_size=step_size,
#         mtx=mtx,
#         cubesat=cubesat,
#     ),
#     promotes=['*'],
# )
# prob.model.add_subsystem(
#     'att',
#     AttitudeGroup(
#         num_times=num_times,
#         num_cp=num_cp,
#         step_size=step_size,
#         mtx=mtx,
#         cubesat=cubesat,
#     ),
#     promotes=['*'],
# )
prob.model.add_subsystem(
    'comm_group',
    CommGroup(
        num_times=num_times,
        num_cp=num_cp,
        step_size=step_size,
        mtx=mtx,
        ground_station=ground_station(
            name='UCSD',
            lon=-117.1611,
            lat=32.7157,
            alt=0.4849,
        ),
    ),
    promotes=['*'],
)
prob.setup()
prob.run_model()
t = np.arange(num_times) * step_size
plt.plot(t, prob['P_comm'])
plt.plot(t, prob['Download_rate'])
plt.show()

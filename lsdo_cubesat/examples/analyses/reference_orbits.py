from csdl import Model, GraphRepresentation
import numpy as np
from ozone.api import ODEProblem
from csdl import Model
import csdl

mu = 398600.44
Re = 6378.137
J2 = 1.08264e-3
J3 = -2.51e-6
J4 = -1.60e-6

A = -(3 / 2 * mu * J2 * Re**2)
B = -(5 / 2 * mu * J3 * Re**3)
C = (15 / 8 * mu * J4 * Re**4)

from lsdo_cubesat.disciplines.orbit.reference_orbit import ReferenceOrbitTrajectory
from lsdo_cubesat.constants import RADII

Re = RADII['Earth']


class VehicleDynamics(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']

        # time steps for all integrators
        self.create_input('h', val=step_size, shape=(num_times - 1, ))

        r_0 = np.array([
            -6.25751454e+03, 3.54135435e+02, 2.72669181e+03, 3.07907496e+00,
            9.00070771e-01, 6.93016106e+00
        ])

        self.add(
            ReferenceOrbitTrajectory(
                num_times=num_times,
                r_0=r_0,
            ),
            name='reference_cubesat',
            promotes=['h'],
        )
        self.add(
            ReferenceOrbitTrajectory(
                num_times=num_times,
                r_0=r_0+np.array([0.02, 0, 0, 0, 0, 0]),
            ),
            name='optics_cubesat',
            promotes=['h'],
        )
        self.add(
            ReferenceOrbitTrajectory(
                num_times=num_times,
                r_0=r_0+np.array([-0.02, 0, 0, 0, 0, 0]),
            ),
            name='detector_cubesat',
            promotes=['h'],
        )


num_orbits = 3
duration = 90
step_size = 19.
num_times = int(duration * 60 * num_orbits / step_size)

# num_times = 9001
# duration = 95.
# step_size = 60 * duration / (num_times - 1)
# num_cp = int(num_times / 5)

m = VehicleDynamics(
    num_times=num_times,
    step_size=step_size,
)
rep = GraphRepresentation(m)

# from csdl_om import Simulator
# sim = Simulator(rep)
# sim.visualize_implementation()
# exit()

from python_csdl_backend import Simulator

sim = Simulator(rep)
sim.run()

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# plot trajectory
x1 = sim['optics_cubesat.orbit_state'][:, 0]
y1 = sim['optics_cubesat.orbit_state'][:, 1]
z1 = sim['optics_cubesat.orbit_state'][:, 2]
ax.plot(x1[0], y1[0], z1[0], 'o')
ax.plot(x1, y1, z1)
x2 = sim['detector_cubesat.orbit_state'][:, 0]
y2 = sim['detector_cubesat.orbit_state'][:, 1]
z2 = sim['detector_cubesat.orbit_state'][:, 2]
ax.plot(x2[0], y2[0], z2[0], 'x')
ax.plot(x2, y2, z2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('absolute orbit, both spacecraft')
plt.show()

x0 = sim['reference_cubesat.orbit_state'][:, 0]
y0 = sim['reference_cubesat.orbit_state'][:, 1]
z0 = sim['reference_cubesat.orbit_state'][:, 2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x0, y0, z0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('absolute orbit, reference spacecraft')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x1 - x0, y1 - y0, z1 - z0)
ax.plot(x2 - x0, y2 - y0, z2 - z0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('relative orbit, both spacecraft')
plt.show()

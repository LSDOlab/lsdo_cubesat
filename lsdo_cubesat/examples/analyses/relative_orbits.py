import matplotlib.pyplot as plt
from csdl import Model, GraphRepresentation
from python_csdl_backend import Simulator
from lsdo_cubesat.disciplines.orbit.reference_orbit import ReferenceOrbitTrajectory
import numpy as np

num_orbits = 30
duration = 90
step_size = 19.
num_times = int(duration * 60 * num_orbits / step_size)


class RelativeOrbits(Model):

    def define(self):
        r = self.create_input('r0',
                              val=np.array([
                                  -6257.51,
                                  354.136,
                                  2726.69,
                                  3.08007,
                                  0.901071,
                                  6.93116,
                              ]))
        self.register_output('initial_position_km', r[:3])
        self.register_output('initial_velocity_km_s', r[3:])

        self.add(
            ReferenceOrbitTrajectory(
                num_times=num_times,
                step_size=step_size,
            ),
            name='r',
            promotes=['initial_position_km', 'initial_velocity_km_s'],
        )
        self.add(
            ReferenceOrbitTrajectory(
                num_times=num_times,
                step_size=step_size,
                relative_initial_state=np.array([20, 0, 0, 0, 0, 0]),
            ),
            name='a',
            promotes=['initial_position_km', 'initial_velocity_km_s'],
        )

        self.add(
            ReferenceOrbitTrajectory(
                num_times=num_times,
                step_size=step_size,
                relative_initial_state=np.array([-20, 0, 0, 0, 0, 0]),
            ),
            name='b',
            promotes=['initial_position_km', 'initial_velocity_km_s'],
        )


rep = GraphRepresentation(RelativeOrbits())
sim = Simulator(rep, mode='rev')
sim.run()
# sim.compute_total_derivatives()
# exit()
# print(sim['reference_orbit_state_km'])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
r = sim['r.orbit_state']
x0 = r[:, 0]
y0 = r[:, 1]
z0 = r[:, 2]
a = sim['a.orbit_state']
x = a[:, 0] - x0
y = a[:, 1] - y0
z = a[:, 2] - z0
ax.plot(x, y, z)
ax.plot(x[0], y[0], z[0], 'o')
b = sim['b.orbit_state']
x = b[:, 0] - x0
y = b[:, 1] - y0
z = b[:, 2] - z0
ax.plot(x, y, z)
ax.plot(x[0], y[0], z[0], 'o')
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')
plt.show()

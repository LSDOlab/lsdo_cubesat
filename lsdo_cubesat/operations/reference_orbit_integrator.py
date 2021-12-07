"""
RK4 component for orbit compute
"""
import os
from six.moves import range

import numpy as np
import scipy.sparse

from lsdo_cubesat.operations.rk4_op import RK4

# Constants
mu = 398600.44
Re = 6378.137
J2 = 1.08264e-3
J3 = -2.51e-6
J4 = -1.60e-6

C1 = -mu
C2 = -1.5 * mu * J2 * Re**2
C3 = -2.5 * mu * J3 * Re**3
C4 = 1.875 * mu * J4 * Re**4


class ReferenceOrbitIntegrator(RK4):
    def initialize(self):
        super().initialize()
        self.parameters.declare('num_times', types=int)

        self.parameters['state_var'] = 'reference_orbit_state_km'
        self.parameters['init_state_var'] = 'initial_orbit_state_km'
        self.parameters['external_vars'] = []

    def define(self):
        n = self.parameters['num_times']

        self.add_input(
            'initial_orbit_state_km',
            shape=6,  # fd_step=1e-2,
            desc='Initial position and velocity vectors from earth to '
            'satellite in Earth-centered inertial frame')

        self.add_output(
            'reference_orbit_state_km',
            shape=(6, n),
            desc='Position and velocity vectors from earth to satellite '
            'in Earth-centered inertial frame over time')

        self.dfdx = np.zeros((6, 1))
        # self.dfdy = np.zeros((6, 6))

    def f_dot(self, external, state):
        # Px = external[0]
        # Py = external[1]
        # Pz = external[2]
        #
        # a_Tx = Tx/m
        # a_Ty = Ty/m
        # a_Tz = Tz/m

        x = state[0]
        y = state[1]
        z = state[2] if abs(state[2]) > 1e-15 else 1e-5

        z2 = z * z
        z3 = z2 * z
        z4 = z3 * z

        r = np.sqrt(x * x + y * y + z2)

        r2 = r * r
        r3 = r2 * r
        r4 = r3 * r
        r5 = r4 * r
        r7 = r5 * r * r

        T2 = 1 - 5 * z2 / r2
        T3 = 3 * z - 7 * z3 / r2
        T4 = 1 - 14 * z2 / r2 + 21 * z4 / r4
        T3z = 3 * z - 0.6 * r2 / z
        T4z = 4 - 28.0 / 3.0 * z2 / r2

        f_dot = np.zeros((6, ))
        f_dot[0:3] = state[3:]
        f_dot[3:] = state[0:3] * (C1 / r3 + C2 / r5 * T2 + C3 / r7 * T3 +
                                  C4 / r7 * T4)
        f_dot[5] += z * (2.0 * C2 / r5 + C3 / r7 * T3z + C4 / r7 * T4z)

        return f_dot

    def df_dy(self, external, state):

        x = state[0]
        y = state[1]
        z = state[2] if abs(state[2]) > 1e-15 else 1e-5

        z2 = z * z
        z3 = z2 * z
        z4 = z3 * z

        r = np.sqrt(x * x + y * y + z2)

        r2 = r * r
        r3 = r2 * r
        r4 = r3 * r
        r5 = r4 * r
        r6 = r5 * r
        r7 = r6 * r
        r8 = r7 * r

        dr = np.array([x, y, z]) / r

        T2 = 1 - 5 * z2 / r2
        T3 = 3 * z - 7 * z3 / r2
        T4 = 1 - 14 * z2 / r2 + 21 * z4 / r4
        T3z = 3 * z - 0.6 * r2 / z
        T4z = 4 - 28.0 / 3.0 * z2 / r2

        dT2 = (10 * z2) / (r3) * dr
        dT2[2] -= 10. * z / r2

        dT3 = 14 * z3 / r3 * dr
        dT3[2] -= 21. * z2 / r2 - 3

        dT4 = (28 * z2 / r3 - 84. * z4 / r5) * dr
        dT4[2] -= 28 * z / r2 - 84 * z3 / r4

        dT3z = -1.2 * r / z * dr
        dT3z[2] += 0.6 * r2 / z2 + 3

        dT4z = 56.0 / 3.0 * z2 / r3 * dr
        dT4z[2] -= 56.0 / 3.0 * z / r2

        eye = np.identity(3)

        # dfdy = self.dfdy
        # dfdy[:, :] = 0.
        dfdy = np.zeros((6, 6))

        dfdy[0:3, 3:] += eye

        dfdy[3:, :3] += eye * (C1 / r3 + C2 / r5 * T2 + C3 / r7 * T3 +
                               C4 / r7 * T4)
        fact = (-3 * C1 / r4 - 5 * C2 / r6 * T2 - 7 * C3 / r8 * T3 -
                7 * C4 / r8 * T4)
        dfdy[3:, 0] += dr[0] * state[:3] * fact
        dfdy[3:, 1] += dr[1] * state[:3] * fact
        dfdy[3:, 2] += dr[2] * state[:3] * fact
        dfdy[3:, 0] += state[:3] * (C2 / r5 * dT2[0] + C3 / r7 * dT3[0] +
                                    C4 / r7 * dT4[0])
        dfdy[3:, 1] += state[:3] * (C2 / r5 * dT2[1] + C3 / r7 * dT3[1] +
                                    C4 / r7 * dT4[1])
        dfdy[3:, 2] += state[:3] * (C2 / r5 * dT2[2] + C3 / r7 * dT3[2] +
                                    C4 / r7 * dT4[2])
        dfdy[5, :3] += dr * z * (-5 * C2 / r6 * 2 - 7 * C3 / r8 * T3z -
                                 7 * C4 / r8 * T4z)
        dfdy[5, :3] += z * (C3 / r7 * dT3z + C4 / r7 * dT4z)
        dfdy[5, 2] += (C2 / r5 * 2 + C3 / r7 * T3z + C4 / r7 * T4z)

        return dfdy

    def df_dx(self, external, state):
        return self.dfdx


if __name__ == '__main__':

    from csdl import Model
    import csdl
    import matplotlib.pyplot as plt
    from csdl_om import Simulator

    np.random.seed(0)

    n = 1500
    m = 1
    npts = 1
    h = 1.5e-4

    r_e2b_I0 = np.empty(6)
    r_e2b_I0[:3] = 1000. * np.random.rand(3)
    r_e2b_I0[3:] = 1. * np.random.rand(3)

    class ReferenceOrbit(Model):
        def initialize(self):
            self.parameters.declare('num_times', types=int)
            self.parameters.declare('step_size', types=float)

        def define(self):
            num_times = self.parameters['num_times']
            step_size = self.parameters['step_size']

            initial_orbit_state_km = self.declare_variable(
                'initial_orbit_state_km',
                val=np.array([
                    Re + 500.,
                    0.,
                    0.,
                    0.,
                    7.5,
                    0.,
                    # 1.76002146e+03,
                    # 6.19179823e+03,
                    # 6.31576531e+03,
                    # 4.73422022e-05,
                    # 1.26425269e-04,
                    # 5.39731211e-05,
                ]))

            reference_orbit_state_km = csdl.custom(
                initial_orbit_state_km,
                op=ReferenceOrbitIntegrator(
                    num_times=num_times,
                    step_size=step_size,
                ),
            )
            self.register_output(
                'reference_orbit_state_km',
                reference_orbit_state_km,
            )

    num_times = 1501
    step_size = 95. * 60. / (num_times - 1)
    t = np.arange(num_times) * step_size
    print('SIMULATING TIME: {} min'.format(t[-1] / 60.))

    sim = Simulator(ReferenceOrbit(
        num_times=num_times,
        step_size=step_size,
    ))
    sim.check_partials(compact_print=True)
    exit()
    sim.run(time_run=True)

    print(sim['initial_orbit_state_km'])
    plt.plot(t, sim['reference_orbit_state_km'][0, :])
    plt.show()
    plt.plot(t, sim['reference_orbit_state_km'][1, :])
    plt.show()
    plt.plot(t, sim['reference_orbit_state_km'][2, :])
    plt.show()

    plt.plot(sim['reference_orbit_state_km'][0, :],
             sim['reference_orbit_state_km'][1, :])
    plt.show()
    plt.plot(sim['reference_orbit_state_km'][0, :],
             sim['reference_orbit_state_km'][2, :])
    plt.show()
    plt.plot(sim['reference_orbit_state_km'][1, :],
             sim['reference_orbit_state_km'][2, :])
    plt.show()

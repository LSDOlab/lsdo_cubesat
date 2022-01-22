"""
RK4 component for orbit compute
"""

from re import S
import numpy as np

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

# rho = 3.89e-12 # kg/m**3 atmoshperic density at altitude = 400 km with mean solar activity
# C_D = 2.2 # Drag coefficient for cube
# area = 0.1 * 0.1 # m**2 cross sectional area
# drag = 1.e-6


def perturbations(r, T2, T3, T4):
    r7 = r**7
    return C2 / r**5 * T2 + C3 / r7 * T3 + C4 / r7 * T4


def dperturbationsdr(r, T2, T3, T4):
    r8 = r**8
    return -5 * C2 / r**6 * T2 - 7 * C3 / r8 * T3 - 7 * C4 / r8 * T4


class RelativeOrbitIntegrator(RK4):

    def initialize(self):
        super().initialize()
        self.parameters.declare('num_times', types=int)

        self.parameters['state_var'] = 'relative_orbit_state_m'
        self.parameters['init_state_var'] = 'initial_orbit_state'
        self.parameters['external_vars'] = ['force_3xn', 'mass', 'radius_m']

    def define(self):
        n = self.parameters['num_times']

        self.add_input('force_3xn', shape=(3, n), desc='Thrust on the cubesat')

        self.add_input('mass', shape=(1, n), desc='mass of Cubesat')

        self.add_input('radius_m', shape=(1, n))

        self.add_input(
            'initial_orbit_state',
            shape=6,
            desc='Initial position and velocity vectors from earth to '
            'satellite in Earth-centered inertial frame')

        self.add_output(
            'relative_orbit_state_m',
            shape=(6, n),
            desc='Position and velocity vectors from earth to satellite '
            'in Earth-centered inertial frame over time')

    def f_dot(self, external, state):

        # TODO: not state, but z component of posiiton in reference orbit?
        z = state[2] if abs(state[2]) > 1e-15 else 1e-5
        z2 = z * z
        z3 = z2 * z
        z4 = z2 * z2

        f = external[:3]
        m = external[3]
        r = external[4]

        r2 = r * r
        r3 = r2 * r
        r4 = r3 * r

        T2 = 1 - 5 * z2 / r2
        T3 = 3 * z - 7 * z3 / r2
        T4 = 1 - 14 * z2 / r2 + 21 * z4 / r4
        T2z = 2 * z
        T3z = 3 * z - 3 * r2 / (5 * z)
        T4z = 4 - 28.0 / 3.0 * z2 / r2

        f_dot = np.zeros((6, ))
        f_dot[0:3] = state[3:]
        f_dot[3:] = f / m
        # f_dot[3:] -= 1e-6 * state[3:] / m  # drag
        f_dot[3:] = f_dot[3:] + state[0:3] * (C1 / r3 +
                                              perturbations(r, T2, T3, T4))
        f_dot[5] = f_dot[5] + z * perturbations(r, T2z, T3z, T4z)

        return f_dot

    def df_dy(self, external, state):

        z = state[2] if abs(state[2]) > 1e-15 else 1e-5

        z2 = z * z
        z3 = z2 * z
        z4 = z3 * z

        r = external[4]

        r2 = r * r
        r3 = r2 * r
        r4 = r3 * r
        r5 = r4 * r
        r6 = r5 * r
        r7 = r6 * r

        T2 = 1 - 5 * z2 / r2
        T3 = 3 * z - 7 * z3 / r2
        T4 = 1 - 14 * z2 / r2 + 21 * z4 / r4
        T3z = 3 * z - 0.6 * r2 / z
        T4z = 4 - 28.0 / 3.0 * z2 / r2

        dT2 = np.zeros(3)
        dT2[2] -= 10. * z / r2

        dT3 = np.zeros(3)
        dT3[2] -= 21. * z2 / r2 - 3

        dT4 = np.zeros(3)
        dT4[2] -= 28 * z / r2 - 84 * z3 / r4

        dT3z = np.zeros(3)
        dT3z[2] += 0.6 * r2 / z2 + 3

        dT4z = np.zeros(3)
        dT4z[2] -= 56.0 / 3.0 * z / r2

        eye = np.identity(3)
        dfdy = np.zeros((6, 6))

        dfdy[0:3, 3:] += eye

        dfdy[3:, :3] += eye * (C1 / r3 + C2 / r5 * T2 + C3 / r7 * T3 +
                               C4 / r7 * T4)
        dfdy[3:, 0] += state[:3] * (C2 / r5 * dT2[0] + C3 / r7 * dT3[0] +
                                    C4 / r7 * dT4[0])
        dfdy[3:, 1] += state[:3] * (C2 / r5 * dT2[1] + C3 / r7 * dT3[1] +
                                    C4 / r7 * dT4[1])
        dfdy[3:, 2] += state[:3] * (C2 / r5 * dT2[2] + C3 / r7 * dT3[2] +
                                    C4 / r7 * dT4[2])
        dfdy[5, :3] += z * (C3 / r7 * dT3z + C4 / r7 * dT4z)
        dfdy[5, 2] += (C2 / r5 * 2 + C3 / r7 * T3z + C4 / r7 * T4z)

        return dfdy

    def df_dx(self, external, state):

        z = state[2] if abs(state[2]) > 1e-15 else 1e-5

        z2 = z * z
        z3 = z2 * z
        z4 = z3 * z

        f = external[:3]
        m = external[3]
        r = external[4]

        r3 = r**3
        r4 = r**4
        r5 = r**5

        # T2 = 1 - 5 * z2 / r2
        dT2dr = -2 * (-5 * z2 / r3)
        # T3 = 3 * z - 7 * z3 / r2
        dT3dr = -2 * (-7 * z3 / r3)
        # T4 = 1 - 14 * z2 / r2 + 21 * z4 / r4
        dT4dr = -2 * (-14 * z2 / r3) - 4 * (21 * z4 / r5)
        # T2z = 2 * z
        # T3z = 3 * z - 3 * r2 / (5 * z)
        dT3zdr = 2 * (-0.6 * r / z)
        # T4z = 4 - 28.0 / 3.0 * z2 / r2
        dT4zdr = -2 * (-28.0 / 3.0 * z2 / r3)

        dfdx = np.zeros((6, 5))

        # wrt mass
        dfdx[3:, 3] = -f / m**2

        # wrt force
        dfdx[3:, :3] = np.eye(3) / m

        # wrt reference radius
        # dfdx[3:, 4] = state[:3] * (-3 * C1 / r4 +
        #                            dperturbationsdr(r, dT2dr, dT3dr, dT4dr))
        # dfdx[5, 4] += z * dperturbationsdr(r, 0, dT3zdr, dT4zdr)

        return dfdx


if __name__ == '__main__':

    from csdl import Model
    import csdl
    import numpy as np
    import matplotlib.pyplot as plt

    class M(Model):

        def initialize(self):
            self.parameters.declare('num_times')
            self.parameters.declare('step_size')

        def define(self):
            num_times = self.parameters['num_times']
            step_size = self.parameters['step_size']

            np.random.seed(0)

            r_e2b_I0 = np.empty(6)
            r_e2b_I0[:3] = 1. * np.random.rand(3)
            r_e2b_I0[3:] = 1. * np.random.rand(3)

            force_3xn = self.declare_variable('force_3xn',
                                              val=np.random.rand(3, num_times))
            initial_orbit_state = self.declare_variable('initial_orbit_state',
                                                        val=r_e2b_I0)
            radius_m = self.declare_variable('radius_m',
                                             val=6400e3 +
                                             np.random.rand(1, num_times))
            mass = self.declare_variable('mass',
                                         val=1.e-2,
                                         shape=(1, num_times))

            relative_orbit_state_m = csdl.custom(
                force_3xn,
                mass,
                radius_m,
                initial_orbit_state,
                op=RelativeOrbitIntegrator(
                    num_times=num_times,
                    step_size=step_size,
                ),
            )
            self.register_output(
                'relative_orbit_state_m',
                relative_orbit_state_m,
            )

    from csdl_om import Simulator
    num_times = 40
    step_size = 95 * 60 / (num_times - 1) * 1e-10

    sim = Simulator(M(
        num_times=num_times,
        step_size=step_size,
    ))
    sim.run()
    # orbit_X = sim['relative_orbit_state_m'][0, :]
    # orbit_Y = sim['relative_orbit_state_m'][1, :]
    # plt.plot(orbit_X, orbit_Y)
    # plt.show()

    sim.check_partials(compact_print=True, step=1e-4)
    # sim.check_partials(step=1e-4)

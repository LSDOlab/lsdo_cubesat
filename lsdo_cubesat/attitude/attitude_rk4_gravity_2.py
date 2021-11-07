from lsdo_cubesat.utils.rk4_comp import RK4Comp
import numpy as np

# Constants
mu = 398600.44
Re = 6378.137


def skew(a, b, c):
    return np.array([
        [0, -c, b],
        [c, 0, -a],
        [-b, a, 0],
    ])


class AttitudeRK4GravityComp(RK4Comp):
    """
    Attitude dynamics model for spacecraft in orbit about point mass.
    The dynamics are integrated using the Runge-Kutta 4 method.

    Options
    -------
    num_times : int
        Number of time steps over which to integrate dynamics
    step_size : float
        Constant time step size to use for integration
    moment_inertia_ratios: array
        Ratio of moments of inertia along principal axes,
        ``(I[1] - I[2])/I[0]``, ``(I[2] - I[0])/I[1]``,
        ``(I[0] - I[1])/I[2]``

    Parameters
    ----------
    initial_angular_velocity_orientation : shape=7
        Initial angular velocity and orientation. First three
        elements correspond to angular velocity. Fourth element
        corresponds to scalar part of unit quaternion. Last three
        elements correspond to vector part of unit quaternion.
    osculating_orbit_angular_speed : shape=(1,num_times)
        Orbit angular speed. Remains constant for circular orbit.
    external_torques_x : shape=num_times
        Exogenous inputs (x), can be from any actuator or external
        moment other than gravity (e.g. atmospheric drag)
    external_torques_y : shape=num_times
        Exogenous inputs (y), can be from any actuator or external
        moment other than gravity (e.g. atmospheric drag)
    external_torques_z : shape=num_times
        Exogenous inputs (z), can be from any actuator or external
        moment other than gravity (e.g. atmospheric drag)

    Returns
    -------
    angular_velocity_orientation : shape=(7,num_times)
        Time history of angular velocity and orientation. First three
        elements correspond to angular velocity. Fourth element
        corresponds to scalar part of unit quaternion. Last three
        elements correspond to vector part of unit quaternion.
    """
    def initialize(self):
        super().initialize()
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('moment_inertia_ratios')

        self.parameters['state_var'] = 'angular_velocity_orientation'
        self.parameters[
            'init_state_var'] = 'initial_angular_velocity_orientation'

        # Mass moment of inertia in body frame coordinates (i.e. nonzero
        # values only on diagonal of inertia matrix)
        self.parameters['external_vars'] = [
            'external_torques_x',
            'external_torques_y',
            'external_torques_z',
            'osculating_orbit_angular_speed',
        ]

    def define(self):
        n = self.parameters['num_times']

        self.add_input(
            'external_torques_x',
            val=0,
            shape=n,
            desc=
            'External torques applied to spacecraft, e.g. ctrl inputs, drag')

        self.add_input(
            'external_torques_y',
            val=0,
            shape=n,
            desc=
            'External torques applied to spacecraft, e.g. ctrl inputs, drag')

        self.add_input(
            'external_torques_z',
            val=0,
            shape=n,
            desc=
            'External torques applied to spacecraft, e.g. ctrl inputs, drag')

        self.add_input('osculating_orbit_angular_speed',
                       shape=(1, n),
                       val=1,
                       desc='Angular speed of oscullating orbit')

        self.add_input('initial_angular_velocity_orientation',
                       shape=12,
                       desc='Initial angular velocity in body frame')

        self.add_output('angular_velocity_orientation',
                        shape=(12, n),
                        desc='Angular velocity in body frame over time')

    def f_dot(self, external, state):
        state_dot = np.zeros(12)
        # K = external[3:6]
        K = self.parameters['moment_inertia_ratios']
        Omega = external[-1]
        omega = state[:3]
        C0 = state[3:6]
        C1 = state[6:9]
        C2 = state[9:]

        # Compute angular acceleration for torque-free motion
        state_dot[:3] = K * np.array([
            omega[1] * omega[2] - 3 * Omega**2 * C0[1] * C0[2],
            omega[2] * omega[0] - 3 * Omega**2 * C0[2] * C0[0],
            omega[0] * omega[1] - 3 * Omega**2 * C0[0] * C0[1],
        ], ) + external[:3]

        # Update direction cosine matrix for body rotating in frame
        # fixed in orbit
        state_dot[3:] = np.array([
            C0[1] * omega[2] - C0[2] * omega[1] + Omega *
            (C0[2] * C2[1] - C0[1] * C2[2]),
            C0[2] * omega[0] - C0[0] * omega[2] + Omega *
            (C0[0] * C2[2] - C0[2] * C2[0]),
            C0[0] * omega[1] - C0[1] * omega[0] + Omega *
            (C0[1] * C2[0] - C0[0] * C2[1]),
            C1[1] * omega[2] - C1[2] * omega[1] + Omega *
            (C1[2] * C2[1] - C1[1] * C2[2]),
            C1[2] * omega[0] - C1[0] * omega[2] + Omega *
            (C1[0] * C2[2] - C1[2] * C2[0]),
            C1[0] * omega[1] - C1[1] * omega[0] + Omega *
            (C1[1] * C2[0] - C1[0] * C2[1]),
            C2[1] * omega[2] - C2[2] * omega[1],
            C2[2] * omega[0] - C2[0] * omega[2],
            C2[0] * omega[1] - C2[1] * omega[0],
        ])

        return state_dot

    def df_dy(self, external, state):
        omega = state[:3]
        # K = external[3:6]
        K = self.parameters['moment_inertia_ratios']
        osculating_orbit_angular_speed = external[-1]
        C0 = state[3:6]
        C1 = state[6:9]
        C2 = state[9:]
        dfdy = np.zeros((12, 12))

        # d omega dot/d omega
        dfdy[:3, :3] = np.matmul(
            np.diag(K),
            skew(omega[0], omega[1], omega[2]),
        )

        # d omega dot/d C
        dfdy[:3, 3:6] = -3 * Omega**2 * np.matmul(
            np.diag(K),
            skew(C0[0], C0[1], C0[2]),
        )

        # d C dot/d omega
        dfdy[3:6, :3] = skew(
            C0[0],
            C0[1],
            C0[2],
        )
        dfdy[6:9, :3] = skew(
            C1[0],
            C1[1],
            C1[2],
        )
        dfdy[9:, :3] = skew(
            C2[0],
            C2[1],
            C2[2],
        )

        # d C dot/d C
        dfdy[3:6, 3:6] = skew(
            Omega * C2[0] - omega[0],
            Omega * C2[1] - omega[1],
            Omega * C2[2] - omega[2],
        )
        dfdy[3:6, 9:] = -Omega * skew(
            C0[0],
            C0[1],
            C0[2],
        )
        dfdy[6:9, 3:6] = skew(
            Omega * C2[0] - omega[0],
            Omega * C2[1] - omega[1],
            Omega * C2[2] - omega[2],
        )
        dfdy[6:9, 9:] = -Omega * skew(
            C1[0],
            C1[1],
            C1[2],
        )
        dfdy[9:, 3:6] = -skew(
            omega[0],
            omega[1],
            omega[2],
        )

        return dfdy

    def df_dx(self, external, state):
        omega = state[:3]
        # K = external[3:6]
        K = self.parameters['moment_inertia_ratios']
        Omega = external[-1]
        dfdx = np.zeros((12, 4))
        C0 = state[3:6]
        C1 = state[6:9]
        C2 = state[9:]

        # d omega/d external torques x, y, z
        dfdx[0, 0] = 1.0
        dfdx[1, 1] = 1.0
        dfdx[2, 2] = 1.0

        # d omega/d Omega
        dfdx[0, -1] = -6 * K[0] * Omega * C0[1] * C0[2]
        dfdx[1, -1] = -6 * K[1] * Omega * C0[2] * C0[0]
        dfdx[2, -1] = -6 * K[2] * Omega * C0[0] * C0[1]

        # d C/d Omega
        dfdx[3, -1] = C0[2] * C2[1] - C0[1] * C2[2]
        dfdx[4, -1] = C0[0] * C2[2] - C0[2] * C2[0]
        dfdx[5, -1] = C0[1] * C2[0] - C0[0] * C2[1]
        dfdx[6, -1] = C1[2] * C2[1] - C1[1] * C2[2]
        dfdx[7, -1] = C1[0] * C2[2] - C1[2] * C2[0]
        dfdx[8, -1] = C1[1] * C2[0] - C1[0] * C2[1]

        return dfdx


if __name__ == "__main__":
    from csdl_om import Simulator
    from csdl import Model
    import csdl

    from lsdo_cubesat.constants import RADII, GRAVITATIONAL_PARAMTERS

    step_size = 0.1
    num_times = 1250
    # num_times = 10
    # 25 time steps/orbit
    time = num_times * step_size

    # 0.0012394474623254955
    # Omega = np.sqrt(GRAVITATIONAL_PARAMTERS['Earth'] /
    #                 (1000 * RADII['Earth'])**3)
    Omega = 1  #0.0011023132117858924

    class M(Model):
        def initialize(self):
            self.parameters.declare('Omega')

        def define(self):
            Omega = self.parameters['Omega']
            w12 = 0.1 * Omega
            w3 = 1.1 * Omega

            C0 = np.array([0.9924, -0.0868, 0.0872])
            C2 = np.array([-0.0789, 0.0944, 0.9924])
            C1 = np.sqrt(1 - C0**2 - C2**2)
            C1[2] = -C1[2]
            C = np.zeros((3, 3))
            C[0, :] = C0
            C[1, :] = C1
            C[2, :] = C2

            test = np.matmul(C, C.T)
            np.testing.assert_almost_equal(test, np.eye(3), decimal=4)

            # stable motion (constant nutation angle)
            # w12 = 0.1 * Omega
            # w3 = 1.1 * Omega
            # C1 = np.array([1, 0, 0])
            # C3 = np.array([0, 0, 1])

            external_torques_x = self.declare_variable(
                'external_torques_x',
                val=np.random.rand(num_times) - 0.5 if num_times < 15 else 0,
                shape=num_times,
                desc=
                'External torques applied to spacecraft, e.g. ctrl inputs, drag'
            )

            external_torques_y = self.declare_variable(
                'external_torques_y',
                val=np.random.rand(num_times) - 0.5 if num_times < 15 else 0,
                shape=num_times,
                desc=
                'External torques applied to spacecraft, e.g. ctrl inputs, drag'
            )

            external_torques_z = self.declare_variable(
                'external_torques_z',
                val=np.random.rand(num_times) - 0.5 if num_times < 15 else 0,
                shape=num_times,
                desc=
                'External torques applied to spacecraft, e.g. ctrl inputs, drag'
            )

            osculating_orbit_angular_speed = self.declare_variable(
                'osculating_orbit_angular_speed',
                shape=(1, num_times),
                val=Omega,
                desc='Angular speed of oscullating orbit')

            v = np.concatenate((np.array([w12, w12,
                                          w3]), np.concatenate(
                                              (C0, C1, C2)))).flatten()
            initial_angular_velocity_orientation = self.declare_variable(
                'initial_angular_velocity_orientation',
                shape=12,
                val=v,
                desc='Initial angular velocity in body frame')

            K1 = -0.5
            K2 = 0.9
            K3 = -(K1 + K2) / (1 + K1 * K2)

            angular_velocity_orientation = csdl.custom(
                external_torques_x,
                external_torques_y,
                external_torques_z,
                osculating_orbit_angular_speed,
                initial_angular_velocity_orientation,
                op=AttitudeRK4GravityComp(
                    num_times=num_times,
                    step_size=step_size,
                    moment_inertia_ratios=np.array([K1, K2, K3]),
                ),
            )
            self.register_output(
                'angular_velocity_orientation',
                angular_velocity_orientation,
            )

    sim = Simulator(M(Omega=Omega, ))
    sim.run()
    if num_times < 15:
        sim.check_partials(compact_print=True, method='cs')
        exit()

    import matplotlib.pyplot as plt
    state = sim['angular_velocity_orientation']
    omega = state[:3, :]

    # NOTE: can't get precession or roll from only first and third row of C
    C0 = np.array(state[3:6, :])
    C1 = np.array(state[6:9, :])
    C2 = np.array(state[9:, :])
    precession = 180 / np.pi * np.arctan2(C2[0, :], -C2[1, :])
    nutation = 180 / np.pi * np.arccos(C2[2, :])
    spin = 180 / np.pi * np.arctan2(C0[2, :], C1[2, :])
    roll = 180 / np.pi * np.arctan2(-C2[1, :], C2[2, :])
    pitch = 180 / np.pi * np.arcsin(C2[0, :])
    yaw = 180 / np.pi * np.arctan2(-C1[0, :], C0[0, :])

    # plt.plot(np.unwrap(precession, 180.))
    plt.plot(nutation)
    # plt.plot(np.unwrap(spin, 180.))
    # plt.plot(roll)
    # plt.plot(pitch)
    # plt.plot(yaw)
    # plt.plot(omega.T)
    # plt.plot(C0.T)
    # plt.plot(C1.T)
    # plt.plot(C2.T)
    plt.show()

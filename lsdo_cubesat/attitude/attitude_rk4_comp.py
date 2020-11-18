import os

import numpy as np
import scipy.sparse
from openmdao.api import ExplicitComponent
from six.moves import range

from lsdo_cubesat.utils.rk4_comp import RK4Comp

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


class AttitudeRK4Comp(RK4Comp):
    """
    Attitude dynamics model for rigid body. The model does not include
    any forces other than exogenous inputs.
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
    external_torques_x : shape=num_times
        Exogenous inputs (x), can be from any actuator or external
        moment
    external_torques_y : shape=num_times
        Exogenous inputs (y), can be from any actuator or external
        moment
    external_torques_z : shape=num_times
        Exogenous inputs (z), can be from any actuator or external
        moment

    Returns
    -------
    angular_velocity_orientation : shape=(7,num_times)
        Time history of angular velocity and orientation. First three
        elements correspond to angular velocity. Fourth element
        corresponds to scalar part of unit quaternion. Last three
        elements correspond to vector part of unit quaternion.
    """
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('moment_inertia_ratios')

        self.options['state_var'] = 'angular_velocity_orientation'
        self.options['init_state_var'] = 'initial_angular_velocity_orientation'

        # Mass moment of inertia in body frame coordinates (i.e. nonzero
        # values only on diagonal of inertia matrix)
        self.options['external_vars'] = [
            'external_torques_x',
            'external_torques_y',
            'external_torques_z',
        ]

    def setup(self):
        n = self.options['num_times']

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

        self.add_input('initial_angular_velocity_orientation',
                       shape=7,
                       desc='Initial angular velocity in body frame')

        self.add_output('angular_velocity_orientation',
                        shape=(7, n),
                        desc='Angular velocity in body frame over time')

    def f_dot(self, external, state):
        state_dot = np.zeros(7)
        # K = external[3:6]
        K = self.options['moment_inertia_ratios']
        omega = state[0:3]

        # Normalize quaternion vector
        # DONE
        state[3:] /= np.linalg.norm(state[3:])

        # Update quaternion rates
        # https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf
        # (151, 159)
        # transpose W' to get dqdw below
        # Compare to Kane, sec 1.13, (6)

        # Compute angular acceleration for torque-free motion
        # DONE
        state_dot[0:3] = K * np.array([
            omega[1] * omega[2],
            omega[2] * omega[0],
            omega[0] * omega[1],
        ])

        # Move last row to top, remove last column, to get dqdw below
        # DONE
        q = state[3:]
        dqdw = 0.5 * np.array([
            [-q[1], -q[2], -q[3]],
            [q[0], -q[3], q[2]],
            [q[3], q[0], -q[1]],
            [-q[2], q[1], q[0]],
        ], )
        state_dot[3:] = np.matmul(dqdw, omega)

        # External forces
        state_dot[0] += external[0]
        state_dot[1] += external[1]
        state_dot[2] += external[2]

        return state_dot

    def df_dy(self, external, state):
        omega = state[:3]
        q = state[3:]
        # K = external[3:6]
        K = self.options['moment_inertia_ratios']
        dfdy = np.zeros((7, 7))

        # quaternion rate wrt angular velocity
        # DONE
        dfdy[3:, :3] = 0.5 * np.array([
            [-q[1], -q[2], -q[3]],
            [q[0], -q[3], q[2]],
            [q[3], q[0], -q[1]],
            [-q[2], q[1], q[0]],
        ], )

        # quaternion rate wrt quaternion
        # DONE
        d_qdot_dq = np.zeros((4, 4))
        d_qdot_dq[0, 1] = -omega[0]
        d_qdot_dq[0, 2] = -omega[1]
        d_qdot_dq[0, 3] = -omega[2]
        d_qdot_dq[1, 0] = omega[0]
        d_qdot_dq[1, 2] = omega[2]
        d_qdot_dq[1, 3] = -omega[1]
        d_qdot_dq[2, 0] = omega[1]
        d_qdot_dq[2, 1] = -omega[2]
        d_qdot_dq[2, 3] = omega[0]
        d_qdot_dq[3, 0] = omega[2]
        d_qdot_dq[3, 1] = omega[1]
        d_qdot_dq[3, 2] = -omega[0]
        d_qdot_dq /= 2.0

        # Take into account normalization of quaternion
        # DONE
        q_norm = np.linalg.norm(q)
        d_qdot_dq = np.matmul(
            (1 / q_norm - np.outer(q, q)) / q_norm**2,
            d_qdot_dq,
        )
        dfdy[3:, 3:] = d_qdot_dq

        # angular acceleration wrt angular velocity (torque-free)
        # DONE
        d_wdot_dw = np.zeros((3, 3))
        d_wdot_dw[0, 0] = 0
        d_wdot_dw[0, 1] = K[0] * omega[2]
        d_wdot_dw[0, 2] = K[0] * omega[1]
        d_wdot_dw[1, 0] = K[1] * omega[2]
        d_wdot_dw[1, 1] = 0
        d_wdot_dw[1, 2] = K[1] * omega[0]
        d_wdot_dw[2, 0] = K[2] * omega[1]
        d_wdot_dw[2, 1] = K[2] * omega[0]
        d_wdot_dw[2, 2] = 0
        dfdy[:3, :3] = d_wdot_dw

        return dfdy

    def df_dx(self, external, state):
        omega = state[:3]
        q = state[3:]
        # K = external[3:6]
        K = self.options['moment_inertia_ratios']
        dfdx = np.zeros((7, 3))

        # angular acceleration wrt external torques
        # state_dot[0] += external[0]
        # state_dot[1] += external[1]
        # state_dot[2] += external[2]
        dfdx[0, 0] = 1.0
        dfdx[1, 1] = 1.0
        dfdx[2, 2] = 1.0

        return dfdx


if __name__ == '__main__':

    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp
    from lsdo_cubesat.utils.random_arrays import make_random_bounded_array
    import matplotlib.pyplot as plt

    np.random.seed(0)
    num_times = 6000
    step_size = 95 * 60 / (num_times - 1)
    step_size = 0.218
    print(step_size)
    # CADRE mass props (3U)
    # Region 6 (unstable under influence of gravity)
    # I = np.array([18, 18, 6]) * 1e-3
    # Region 1 (not necessarily unstable under influence of gravity)
    I = np.array([30, 40, 50])
    # wq0 = np.array([-1, 0.2, 0.3, 0, 0, 0, 1])
    # wq0 = np.array([-0.3, -1, 0.2, 0, 0, 0, 1])
    wq0 = np.array([-0.3, -0.2, 1, 0, 0, 0, 1])
    # Region 7 (not necessarily unstable under influence of gravity)
    I = np.array([90, 100, 80])
    wq0 = np.array([-1, 0.2, 0.3, 0, 0, 0, 1])
    # wq0 = np.array([-0.3, -1, 0.2, 0, 0, 0, 1])
    # wq0 = np.array([-0.3, -0.2, 1, 0, 0, 0, 1])

    wq0 = np.random.rand(7) - 0.5
    wq0[3:] /= np.linalg.norm(wq0[3:])
    print(wq0[3:])
    print(np.linalg.norm(wq0[3:]))

    class TestGroup(Group):
        def setup(self):
            comp = IndepVarComp()
            # comp.add_output('mass_moment_inertia_b_frame_km_m2',
            #                 val=np.random.rand(3))
            comp.add_output('initial_angular_velocity_orientation', val=wq0)
            comp.add_output(
                'osculating_orbit_angular_speed',
                val=2 * np.pi,
                shape=(1, num_times),
            )
            comp.add_output(
                'external_torques_x',
                val=make_random_bounded_array(num_times, bound=1).reshape(
                    (1, num_times)),
                # val=0,
                shape=(1, num_times),
            )
            comp.add_output(
                'external_torques_y',
                val=make_random_bounded_array(num_times, bound=1).reshape(
                    (1, num_times)),
                # val=0,
                shape=(1, num_times),
            )
            comp.add_output(
                'external_torques_z',
                val=make_random_bounded_array(num_times, bound=1).reshape(
                    (1, num_times)),
                # val=0,
                shape=(1, num_times),
            )
            self.add_subsystem('inputs_comp', comp, promotes=['*'])
            # self.add_subsystem('inertia_ratios_comp',
            #                    InertiaRatiosComp(),
            #                    promotes=['*'])
            # self.add_subsystem('expand_inertia_ratios',
            #                    ArrayExpansionComp(
            #                        shape=(3, num_times),
            #                        expand_indices=[1],
            #                        in_name='moment_inertia_ratios',
            #                        out_name='moment_inertia_ratios_3xn',
            #                    ),
            #                    promotes=['*'])
            self.add_subsystem('comp',
                               AttitudeRK4Comp(num_times=num_times,
                                               step_size=step_size,
                                               moment_inertia_ratios=np.array(
                                                   [2.0 / 3.0, -2.0 / 3.0,
                                                    0])),
                               promotes=['*'])

    prob = Problem()
    prob.model = TestGroup()
    prob.setup(check=True, force_alloc_complex=True)
    if num_times < 10:
        prob.check_partials(compact_print=True)
    else:
        prob.run_model()
        w = prob['angular_velocity_orientation'][:3, :]
        q = prob['angular_velocity_orientation'][3:, :]

        fig = plt.figure()
        t = np.arange(num_times) * step_size

        plt.plot(t, w[0, :])
        plt.plot(t, w[1, :])
        plt.plot(t, w[2, :])
        plt.title('angular velocity')
        plt.show()

        plt.plot(t[:-1], np.linalg.norm(q[:, :-1], axis=0) - 1)
        plt.title('quaternion magnitude error')
        plt.show()

        # plt.plot(t[:-1], q[0, :-1])
        # plt.plot(t[:-1], q[1, :-1])
        # plt.plot(t[:-1], q[2, :-1])
        # plt.plot(t[:-1], q[3, :-1])
        # plt.title('unit quaternion')
        # plt.show()

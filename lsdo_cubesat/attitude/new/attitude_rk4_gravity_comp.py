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


class AttitudeRK4GravityComp(RK4Comp):
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
            'osculating_orbit_angular_speed',
        ]
        self.print_qnorm = True

    def setup(self):
        n = self.options['num_times']

        self.add_input(
            'external_torques_x',
            shape=n,
            desc=
            'External torques applied to spacecraft, e.g. ctrl inputs, drag')

        self.add_input(
            'external_torques_y',
            shape=n,
            desc=
            'External torques applied to spacecraft, e.g. ctrl inputs, drag')

        self.add_input(
            'external_torques_z',
            shape=n,
            desc=
            'External torques applied to spacecraft, e.g. ctrl inputs, drag')

        self.add_input('osculating_orbit_angular_speed',
                       shape=(1, n),
                       val=0.0011023132117858924,
                       desc='Mean motion of oscullating orbit')

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
        osculating_orbit_angular_speed = external[-1]
        omega = state[:3]

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

        # Add effects of gravity assuming Earth is point mass;
        # Use mean motion from osculating orbit;
        # Orbit not affected by attitude, energy not conserved
        R11 = 1 - 2 * (q[2]**2 + q[3]**2)
        R21 = 2 * (q[1] * q[2] - q[3] * q[0])
        R31 = 2 * (q[3] * q[1] + q[2] * q[0])

        state_dot[
            0] += -3 * osculating_orbit_angular_speed**2 * K[0] * R21 * R31
        state_dot[
            1] += -3 * osculating_orbit_angular_speed**2 * K[1] * R31 * R11
        state_dot[
            2] += -3 * osculating_orbit_angular_speed**2 * K[2] * R11 * R21

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
        osculating_orbit_angular_speed = external[-1]
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

        # # angular acceleration wrt quaternions (due to gravity torque)
        # DONE
        R11 = 1 - 2.0 * (q[2]**2 + q[3]**2)
        R21 = 2.0 * (q[1] * q[2] - q[3] * q[0])
        R31 = 2.0 * (q[3] * q[1] + q[2] * q[0])

        dR11_dq = np.zeros(4)
        dR11_dq[2] = -4.0 * q[3]
        dR11_dq[3] = -4.0 * q[2]
        # dR11_dq[2] = -4.0 * q[2]
        # dR11_dq[3] = -4.0 * q[3]

        dR21_dq = np.zeros(4)
        dR21_dq[0] = -2.0 * q[3]
        dR21_dq[1] = 2.0 * q[2]
        dR21_dq[2] = 2.0 * q[1]
        dR21_dq[3] = -2.0 * q[0]
        # dR21_dq[0] = -2.0 * q[0]
        # dR21_dq[1] = 2.0 * q[1]
        # dR21_dq[2] = 2.0 * q[2]
        # dR21_dq[3] = -2.0 * q[3]

        dR31_dq = np.zeros(4)
        dR31_dq[0] = 2.0 * q[2]
        dR31_dq[1] = 2.0 * q[3]
        dR31_dq[2] = 2.0 * q[0]
        dR31_dq[3] = 2.0 * q[1]
        # dR31_dq[0] = 2.0 * q[0]
        # dR31_dq[1] = 2.0 * q[1]
        # dR31_dq[2] = 2.0 * q[2]
        # dR31_dq[3] = 2.0 * q[3]

        # state_dot[0] += -3 * osculating_orbit_angular_speed**2 * K[0] * R21 * R31
        # state_dot[1] += -3 * osculating_orbit_angular_speed**2 * K[1] * R31 * R11
        # state_dot[2] += -3 * osculating_orbit_angular_speed**2 * K[2] * R11 * R21

        d_wdot_dq = np.zeros((3, 4))
        d_wdot_dq[0, 0] = -3 * osculating_orbit_angular_speed**2 * K[0] * (
            dR21_dq[0] * R31 + R21 * dR31_dq[0])
        d_wdot_dq[0, 1] = -3 * osculating_orbit_angular_speed**2 * K[0] * (
            dR21_dq[1] * R31 + R21 * dR31_dq[1])
        d_wdot_dq[0, 2] = -3 * osculating_orbit_angular_speed**2 * K[0] * (
            dR21_dq[2] * R31 + R21 * dR31_dq[2])
        d_wdot_dq[0, 3] = -3 * osculating_orbit_angular_speed**2 * K[0] * (
            dR21_dq[3] * R31 + R21 * dR31_dq[3])
        d_wdot_dq[1, 0] = -3 * osculating_orbit_angular_speed**2 * K[
            1] * dR31_dq[0] * R11
        d_wdot_dq[1, 1] = -3 * osculating_orbit_angular_speed**2 * K[
            1] * dR31_dq[1] * R11
        d_wdot_dq[1, 2] = -3 * osculating_orbit_angular_speed**2 * K[1] * (
            dR31_dq[2] * R11 + R31 * dR11_dq[2])
        d_wdot_dq[1, 3] = -3 * osculating_orbit_angular_speed**2 * K[1] * (
            dR31_dq[3] * R11 + R31 * dR11_dq[3])
        d_wdot_dq[2, 0] = -3 * osculating_orbit_angular_speed**2 * K[
            2] * R11 * dR21_dq[0]
        d_wdot_dq[2, 1] = -3 * osculating_orbit_angular_speed**2 * K[
            2] * R11 * dR21_dq[1]
        d_wdot_dq[2, 2] = -3 * osculating_orbit_angular_speed**2 * K[2] * (
            dR11_dq[2] * R21 + R11 * dR21_dq[2])
        d_wdot_dq[2, 3] = -3 * osculating_orbit_angular_speed**2 * K[2] * (
            dR11_dq[3] * R21 + R11 * dR21_dq[3])

        dfdy[:3, 3:] = d_wdot_dq

        return dfdy

    def df_dx(self, external, state):
        omega = state[:3]
        q = state[3:]
        # K = external[3:6]
        K = self.options['moment_inertia_ratios']
        osculating_orbit_angular_speed = external[-1]
        dfdx = np.zeros((7, 4))

        # angular acceleration wrt external torques
        # state_dot[0] += external[0]
        # state_dot[1] += external[1]
        # state_dot[2] += external[2]
        dfdx[0, 0] = 1.0
        dfdx[1, 1] = 1.0
        dfdx[2, 2] = 1.0

        # angular acceleration wrt inertia ratios (torque-free motion)
        # state_dot[0] = K[0] * omega[1] * omega[2]
        # state_dot[1] = K[1] * omega[2] * omega[0]
        # state_dot[2] = K[2] * omega[0] * omega[1]
        # dfdx[0, 4] = omega[1] * omega[2]
        # dfdx[1, 5] = omega[2] * omega[0]
        # dfdx[2, 6] = omega[0] * omega[1]

        # angular acceleration wrt inertia ratios (gravity torque)
        # state_dot[0] += -3 * osculating_orbit_angular_speed**2 * K[0] * R21 * R31
        # state_dot[1] += -3 * osculating_orbit_angular_speed**2 * K[1] * R31 * R11
        # state_dot[2] += -3 * osculating_orbit_angular_speed**2 * K[2] * R11 * R21
        # dfdx[0, 4] += -3 * osculating_orbit_angular_speed**2 * omega[1] * omega[2]
        # dfdx[1, 5] += -3 * osculating_orbit_angular_speed**2 * omega[2] * omega[0]
        # dfdx[2, 6] += -3 * osculating_orbit_angular_speed**2 * omega[0] * omega[1]

        # angular acceleration wrt osculating mean motion
        # state_dot[0] += -3 * osculating_orbit_angular_speed**2 * K[0] * R21 * R31
        # state_dot[1] += -3 * osculating_orbit_angular_speed**2 * K[1] * R31 * R11
        # state_dot[2] += -3 * osculating_orbit_angular_speed**2 * K[2] * R11 * R21
        R11 = 1 - 2 * (q[2]**2 + q[3]**2)
        R21 = 2 * (q[1] * q[2] - q[3] * q[0])
        R31 = 2 * (q[3] * q[1] + q[2] * q[0])
        dfdx[0, -1] += -6 * osculating_orbit_angular_speed * K[0] * R21 * R31
        dfdx[1, -1] += -6 * osculating_orbit_angular_speed * K[1] * R31 * R11
        dfdx[2, -1] += -6 * osculating_orbit_angular_speed * K[2] * R11 * R21

        return dfdx


if __name__ == '__main__':

    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp
    from lsdo_cubesat.utils.random_arrays import make_random_bounded_array
    import matplotlib.pyplot as plt

    np.random.seed(0)
    num_times = 6000
    num_times = 100
    step_size = 95 * 60 / (num_times - 1)
    # step_size = 95 * 60 / (14000 - 1)
    step_size = 0.12
    step_size = 1e-2  # 3e+02
    step_size = 1e-3  # 6e-02
    step_size = 1e-4  # 1.46e-04
    step_size = 1e-5  # 1.46e-07
    step_size = 1e-6  # 1.46e-10
    step_size = 1e-7  # 1.46e-13
    step_size = 1e-8  # 1.46e-16
    step_size = 1e-9  # 1.46e-19
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
                               AttitudeRK4GravityComp(
                                   num_times=num_times,
                                   step_size=step_size,
                                   moment_inertia_ratios=np.array(
                                       [2.0 / 3.0, -2.0 / 3.0, 0])),
                               promotes=['*'])

    prob = Problem()
    prob.model = TestGroup()
    prob.setup(check=True, force_alloc_complex=True)
    if num_times < 101:
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

        plt.plot(t[:-1], q[0, :-1])
        plt.plot(t[:-1], q[1, :-1])
        plt.plot(t[:-1], q[2, :-1])
        plt.plot(t[:-1], q[3, :-1])
        plt.title('unit quaternion')
        plt.show()

    # # Polhode plot
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # w0 = np.linalg.norm(wq0[:3])
    # print(w0)
    # x = omega[0, :] / w0
    # y = omega[1, :] / w0
    # z = omega[2, :] / w0
    # ax.plot(
    #     x,
    #     y,
    #     z,
    # )
    # plt.xlabel('x')
    # plt.ylabel('y')
    # # plt.xlim((-0.2, 0.2))
    # # plt.ylim((-0.2, 0.2))
    # plt.show()
    # plt.plot(np.linalg.norm(q, axis=0))
    # plt.show()
    # # prob.model.list_outputs()

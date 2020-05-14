"""
RK4 component for orbit compute
"""
import os
from six.moves import range

import numpy as np
import scipy.sparse

from openmdao.api import ExplicitComponent
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
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('step_size', types=float)

        self.options['state_var'] = 'angular_velocity_orientation'
        self.options['init_state_var'] = 'initial_angular_velocity_orientation'

        # Mass moment of inertia in body frame coordinates (i.e. nonzero
        # values only on diagonal of inertia matrix)
        self.options['external_vars'] = [
            'external_torques_x',
            'external_torques_y',
            'external_torques_z',
            'moment_inertia_ratios_3xn',
            'osculating_mean_motion',
        ]

    def setup(self):
        n = self.options['num_times']

        self.add_input(
            'external_torques_x',
            shape=n,
            desc='External torques applied to spacecraft, e.g. ctrl inputs')

        self.add_input(
            'external_torques_y',
            shape=n,
            desc='External torques applied to spacecraft, e.g. ctrl inputs')

        self.add_input(
            'external_torques_z',
            shape=n,
            desc='External torques applied to spacecraft, e.g. ctrl inputs')

        self.add_input('moment_inertia_ratios_3xn',
                       shape=(3, n),
                       desc='Inertia ratios of spacecraft')

        self.add_input('osculating_mean_motion',
                       shape=(1, n),
                       desc='Mean motion of oscullating orbit')

        self.add_input('initial_angular_velocity_orientation',
                       shape=7,
                       desc='Initial angular velocity in body frame')

        self.add_output('angular_velocity_orientation',
                        shape=(7, n),
                        desc='Angular velocity in body frame over time')

    def f_dot(self, external, state):
        state_dot = np.zeros(7)
        K = external[3:6]
        osculating_mean_motion = external[-1]
        omega = state[0:3]

        # Normalize quaternion vector
        # state[3:] /= np.linalg.norm(state[3:])

        # Update quaternion rates
        q = state[3:]
        # E = np.array(
        #     [
        #         [q[0], -q[1], -q[2], -q[3]], \
        #         [q[1], q[0], -q[3], q[2]], \
        #         [q[2], q[3], q[0], -q[1]], \
        #         [q[3], -q[2], q[1], q[0]]
        #     ])
        # state_dot[3:] = 0.5 * np.matmul(
        #     E, np.array([0, omega[0], omega[1], omega[2]]))
        state_dot[3] = -q[1] * omega[0] - q[2] * omega[1] - q[3] * omega[2]
        state_dot[4] = q[0] * omega[0] - q[3] * omega[1] + q[2] * omega[2]
        state_dot[5] = q[3] * omega[0] + q[0] * omega[1] - q[1] * omega[2]
        state_dot[6] = -q[2] * omega[0] + q[1] * omega[1] + q[0] * omega[2]
        state_dot[3:] /= 2.0

        # Compute angular acceleration for torque-free motion
        state_dot[0] = K[0] * omega[1] * omega[2]
        state_dot[1] = K[1] * omega[2] * omega[0]
        state_dot[2] = K[2] * omega[0] * omega[1]

        # Add effects of gravity assuming Earth is point mass;
        # Use mean motion from osculating orbit;
        # Orbit not affected by attitude, energy not conserved
        R11 = 1 - 2 * (q[2]**2 + q[3]**2)

        R21 = 2 * (q[1] * q[2] - q[3] * q[0])
        R31 = 2 * (q[3] * q[1] + q[2] * q[0])

        state_dot[0] += -3 * osculating_mean_motion**2 * K[0] * R21 * R31
        state_dot[1] += -3 * osculating_mean_motion**2 * K[1] * R31 * R11
        state_dot[2] += -3 * osculating_mean_motion**2 * K[2] * R11 * R21

        # Control Inputs
        state_dot[0] += external[0]
        state_dot[1] += external[1]
        state_dot[2] += external[2]

        # TODO: drag force (depends on quaternions)

        return state_dot

    def df_dy(self, external, state):
        omega = state[:3]
        q = state[3:]
        K = external[3:6]
        osculating_mean_motion = external[-1]
        dfdy = np.zeros((7, 7))

        # quaternion rate wrt quaternion
        E = np.array(
            [
                [q[0], -q[1], -q[2], -q[3]], \
                [q[1], q[0], -q[3], q[2]], \
                [q[2], q[3], q[0], -q[1]], \
                [q[3], -q[2], q[1], q[0]]
            ])
        dfdy[3:, :3] = E[:, 1:] / 2

        d_qdot_dq = np.zeros((4, 4))
        d_qdot_dq[0, 1] = -omega[0]
        d_qdot_dq[0, 2] = -omega[1]
        d_qdot_dq[0, 3] = -omega[2]
        d_qdot_dq[1, 0] = omega[0]
        d_qdot_dq[1, 2] = -omega[1]
        d_qdot_dq[1, 3] = omega[2]
        d_qdot_dq[2, 0] = omega[1]
        d_qdot_dq[2, 1] = omega[2]
        d_qdot_dq[2, 3] = -omega[0]
        d_qdot_dq[3, 0] = omega[2]
        d_qdot_dq[3, 1] = -omega[1]
        d_qdot_dq[3, 2] = omega[0]
        dfdy[3:, 3:] = d_qdot_dq

        # angular acceleration wrt angular velocity (torque-free)
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

        # angular acceleration wrt quaternions (due to gravity torque)
        R11 = 1 - 2 * (q[2]**2 + q[3]**2)
        R21 = 2 * (q[1] * q[2] - q[3] * q[0])
        R31 = 2 * (q[3] * q[1] + q[2] * q[0])

        dR11_dq = np.zeros(4)
        dR11_dq[2] = -4 * q[3]
        dR11_dq[3] = -4 * q[2]

        dR21_dq = np.zeros(4)
        dR21_dq[0] = -2 * q[0]
        dR21_dq[1] = 2 * q[1]
        dR21_dq[2] = 2 * q[2]
        dR21_dq[3] = -2 * q[3]

        dR31_dq = np.zeros(4)
        dR31_dq[0] = 2 * q[0]
        dR31_dq[1] = 2 * q[1]
        dR31_dq[2] = 2 * q[2]
        dR31_dq[3] = 2 * q[3]

        d_wdot_dq = np.zeros((3, 4))
        d_wdot_dq[0, 0] = -3 * osculating_mean_motion**2 * K[0] * (
            dR21_dq[0] * R31 + R21 * dR31_dq[0])
        d_wdot_dq[0, 1] = -3 * osculating_mean_motion**2 * K[0] * (
            dR21_dq[1] * R31 + R21 * dR31_dq[1])
        d_wdot_dq[0, 2] = -3 * osculating_mean_motion**2 * K[0] * (
            dR21_dq[2] * R31 + R21 * dR31_dq[2])
        d_wdot_dq[0, 3] = -3 * osculating_mean_motion**2 * K[0] * (
            dR21_dq[3] * R31 + R21 * dR31_dq[3])
        d_wdot_dq[1,
                  0] = -3 * osculating_mean_motion**2 * K[1] * dR31_dq[0] * R11
        d_wdot_dq[1,
                  1] = -3 * osculating_mean_motion**2 * K[1] * dR31_dq[1] * R11
        d_wdot_dq[1, 2] = -3 * osculating_mean_motion**2 * K[1] * (
            dR31_dq[2] * R11 + R31 * dR11_dq[2])
        d_wdot_dq[1, 3] = -3 * osculating_mean_motion**2 * K[1] * (
            dR31_dq[3] * R11 + R31 * dR11_dq[3])
        d_wdot_dq[2,
                  0] = -3 * osculating_mean_motion**2 * K[2] * R11 * dR21_dq[0]
        d_wdot_dq[2,
                  1] = -3 * osculating_mean_motion**2 * K[2] * R11 * dR21_dq[1]
        d_wdot_dq[2, 2] = -3 * osculating_mean_motion**2 * K[2] * (
            dR11_dq[2] * R21 + R11 * dR21_dq[2])
        d_wdot_dq[2, 3] = -3 * osculating_mean_motion**2 * K[2] * (
            dR11_dq[3] * R21 + R11 * dR21_dq[3])
        dfdy[:3, 3:] = d_wdot_dq

        return dfdy

    def df_dx(self, external, state):
        omega = state[:3]
        q = state[3:]
        K = external[3:6]
        osculating_mean_motion = external[-1]
        dfdx = np.zeros((7, 7))

        # NOTE: quaternions do not depend on external variables

        # angular acceleration wrt inertia ratios (torque-free motion)
        # state_dot[0] = K[0] * omega[1] * omega[2]
        # state_dot[1] = K[1] * omega[2] * omega[0]
        # state_dot[2] = K[2] * omega[0] * omega[1]
        dfdx[0, 4] = omega[1] * omega[2]
        dfdx[1, 5] = omega[2] * omega[0]
        dfdx[2, 6] = omega[0] * omega[1]

        # angular acceleration wrt inertia ratios (gravity torque)
        # state_dot[0] += -3 * osculating_mean_motion**2 * K[0] * R21 * R31
        # state_dot[1] += -3 * osculating_mean_motion**2 * K[1] * R31 * R11
        # state_dot[2] += -3 * osculating_mean_motion**2 * K[2] * R11 * R21
        dfdx[0, 4] += -3 * osculating_mean_motion**2 * omega[1] * omega[2]
        dfdx[1, 5] += -3 * osculating_mean_motion**2 * omega[2] * omega[0]
        dfdx[2, 6] += -3 * osculating_mean_motion**2 * omega[0] * omega[1]

        # angular acceleration wrt osculating mean motion
        # state_dot[0] += -3 * osculating_mean_motion**2 * K[0] * R21 * R31
        # state_dot[1] += -3 * osculating_mean_motion**2 * K[1] * R31 * R11
        # state_dot[2] += -3 * osculating_mean_motion**2 * K[2] * R11 * R21
        R11 = 1 - 2 * (q[2]**2 + q[3]**2)
        R21 = 2 * (q[1] * q[2] - q[3] * q[0])
        R31 = 2 * (q[3] * q[1] + q[2] * q[0])
        dfdx[0, -1] += -6 * osculating_mean_motion * K[0] * R21 * R31
        dfdx[1, -1] += -6 * osculating_mean_motion * K[1] * R31 * R11
        dfdx[2, -1] += -6 * osculating_mean_motion * K[2] * R11 * R21

        # angular acceleration wrt external torques
        # state_dot[0] += external[0]
        # state_dot[1] += external[1]
        # state_dot[2] += external[2]
        dfdx[0, 0] = 1.0
        dfdx[1, 1] = 1.0
        dfdx[2, 2] = 1.0

        # TODO: add drag force, which does depend on s/c orientation

        return dfdx


if __name__ == '__main__':

    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp
    from lsdo_utils.api import ArrayExpansionComp
    from lsdo_cubesat.attitude.inertia_ratios_comp import InertiaRatiosComp
    import matplotlib.pyplot as plt

    np.random.seed(0)
    h = 1.5e-4
    num_times = 30
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

    # wq0[3:] = np.random.rand(4)
    # wq0 = np.random.rand(7)
    wq0[3:] /= np.linalg.norm(wq0[3:])

    class TestGroup(Group):
        def setup(self):
            comp = IndepVarComp()
            comp.add_output('mass_moment_inertia_b_frame_km_m2', val=I)
            comp.add_output('initial_angular_velocity_orientation', val=wq0)
            comp.add_output('osculating_mean_motion', val=np.ones(num_times))
            comp.add_output(
                'external_torques_x',
                val=np.random.rand(num_times),
            )
            comp.add_output(
                'external_torques_y',
                val=np.random.rand(num_times),
            )
            comp.add_output(
                'external_torques_z',
                val=np.random.rand(num_times),
            )
            self.add_subsystem('inputs_comp', comp, promotes=['*'])
            self.add_subsystem('inertia_ratios_comp',
                               InertiaRatiosComp(),
                               promotes=['*'])
            self.add_subsystem('expand_inertia_ratios',
                               ArrayExpansionComp(
                                   shape=(3, num_times),
                                   expand_indices=[1],
                                   in_name='moment_inertia_ratios',
                                   out_name='moment_inertia_ratios_3xn',
                               ),
                               promotes=['*'])
            self.add_subsystem('comp',
                               AttitudeRK4Comp(num_times=num_times,
                                               step_size=h),
                               promotes=['*'])

    prob = Problem()
    prob.model = TestGroup()
    prob.setup(check=True, force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

    # omega = prob['angular_velocity_orientation'][:3]
    # q = prob['angular_velocity_orientation'][3:]
    # t = np.arange(num_times) * h
    # plt.plot(t, omega[0, :])
    # plt.plot(t, omega[1, :])
    # plt.plot(t, omega[2, :])
    # plt.show()
    # plt.plot(t, q[0, :])
    # plt.plot(t, q[1, :])
    # plt.plot(t, q[2, :])
    # plt.plot(t, q[3, :])
    # plt.show()

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

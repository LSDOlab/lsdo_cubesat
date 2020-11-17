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
                       val=0.0011023132117858924,
                       desc='Angular speed of oscullating orbit')

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
        state[3:] /= np.linalg.norm(state[3:])

        # Update quaternion rates
        # https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf
        # (151, 159)
        # transpose W' to get dqdw below
        # Compare to Kane, sec 1.13, (6)

        # Compute angular acceleration for torque-free motion
        state_dot[0:3] = K * np.array([
            omega[1] * omega[2],
            omega[2] * omega[0],
            omega[0] * omega[1],
        ])

        # Move last row to top, remove last column, to get dqdw below
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
        dfdy[3:, :3] = 0.5 * np.array([
            [-q[1], -q[2], -q[3]],
            [q[0], -q[3], q[2]],
            [q[3], q[0], -q[1]],
            [-q[2], q[1], q[0]],
        ], )

        # quaternion rate wrt quaternion
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
        q_norm = np.linalg.norm(q)
        d_qdot_dq = np.matmul(
            (1 / q_norm - np.outer(q, q)) / q_norm**2,
            d_qdot_dq,
        )
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
        R11 = 1 - 2.0 * (q[2]**2 + q[3]**2)
        R21 = 2.0 * (q[1] * q[2] - q[3] * q[0])
        R31 = 2.0 * (q[3] * q[1] + q[2] * q[0])

        dR11_dq = np.zeros(4)
        dR11_dq[2] = -4.0 * q[3]
        dR11_dq[3] = -4.0 * q[2]

        dR21_dq = np.zeros(4)
        dR21_dq[0] = -2.0 * q[3]
        dR21_dq[1] = 2.0 * q[2]
        dR21_dq[2] = 2.0 * q[1]
        dR21_dq[3] = -2.0 * q[0]

        dR31_dq = np.zeros(4)
        dR31_dq[0] = 2.0 * q[2]
        dR31_dq[1] = 2.0 * q[3]
        dR31_dq[2] = 2.0 * q[0]
        dR31_dq[3] = 2.0 * q[1]

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

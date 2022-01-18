from lsdo_cubesat.operations.rk4_op import RK4
import numpy as np


def skew(a, b, c):
    return np.array([
        [0, -c, b],
        [c, 0, -a],
        [-b, a, 0],
    ])


def skew_array(a):
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0],
    ])


class RWSpeed(RK4):

    def initialize(self):
        super().initialize()
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('rw_mmoi', types=np.ndarray)

        self.parameters['external_vars'] = [
            'body_torque',
            'body_rates',
        ]
        self.parameters['init_state_var'] = 'initial_rw_speed'
        self.parameters['state_var'] = 'rw_speed'

    def define(self):
        num_times = self.parameters['num_times']
        rw_mmoi = self.parameters['rw_mmoi']
        # external
        self.add_input(
            'body_torque',
            shape=(3, num_times),
        )
        self.add_input(
            'body_rates',
            shape=(3, num_times),
        )
        # initial
        self.add_input(
            'initial_rw_speed',
            val=0,
            shape=3,
        )
        # integrated
        self.add_output(
            'rw_speed',
            shape=(3, num_times),
        )
        self.dfdx = np.zeros((3, 6))
        self.rw_mmoi = np.diag(rw_mmoi)
        self.rw_mmoi_inv = np.diag(1 / rw_mmoi)
        self.dfdx[:, :3] = -self.rw_mmoi_inv

    def f_dot(self, external, state):
        J = self.parameters['rw_mmoi']
        body_torque = external[:3]
        body_rates = external[3:6]
        return -np.matmul(self.rw_mmoi_inv,
                          (np.cross(body_rates, J * state) + body_torque))

    # ODE wrt state variables
    def df_dy(self, external, state):
        body_rates = external[3:6]
        return -np.matmul(self.rw_mmoi_inv,
                          np.matmul(skew_array(body_rates), self.rw_mmoi))

    # ODE wrt external inputs
    def df_dx(self, external, state):
        Jw = np.matmul(self.rw_mmoi, state)
        self.dfdx[:, 3:] = np.matmul(self.rw_mmoi_inv, skew_array(Jw))
        return self.dfdx

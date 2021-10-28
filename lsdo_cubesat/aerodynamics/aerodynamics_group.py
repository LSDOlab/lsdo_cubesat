import numpy as np

from csdl import Model


class AerodynamicsGroup(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        # self.parameters.declare('num_cp', types=int)
        # self.parameters.declare('step_size', types=float)
        # self.parameters.declare('cubesat')
        # self.parameters.declare('mtx')

    def define(self):
        num_times = self.parameters['num_times']
        # num_cp = self.parameters['num_cp']
        # step_size = self.parameters['step_size']
        # cubesat = self.parameters['cubesat']
        # mtx = self.parameters['mtx']

        drag_scalar_3xn = self.create_input(
            'drag_scalar_3xn',
            val=1.e-6,
            shape=(3, num_times),
        )

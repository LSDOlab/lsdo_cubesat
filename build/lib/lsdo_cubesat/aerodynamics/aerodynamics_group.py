import numpy as np

from openmdao.api import Group, IndepVarComp


class AerodynamicsGroup(Group):

    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('cubesat')
        self.options.declare('mtx')

    def setup(self):
        num_times = self.options['num_times']
        num_cp = self.options['num_cp']
        step_size = self.options['step_size']
        cubesat = self.options['cubesat']
        mtx = self.options['mtx']

        shape = (3, num_times)

        comp = IndepVarComp()
        comp.add_output('drag_scalar_3xn', val=1.e-6, shape=shape)
        self.add_subsystem('inputs_comp', comp, promotes=['*'])
"""
Determine if the Satellite has line of sight with the ground stations
"""

from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent
from lsdo_cubesat.utils.utils import get_array_indices

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return np.exp(x)/(np.exp(x)+1)**2

class CommLOSComp(ExplicitComponent):
    Re = 6378.137

    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']

        self.add_input('r_b2g_I', np.zeros((3, num_times)),
                       desc='Position vector from satellite to ground station '
                            'in Earth-centered inertial frame over time')

        self.add_input('r_e2g_I', np.zeros((3, num_times)),
                       desc='Position vector from earth to ground station in '
                            'Earth-centered inertial frame over time')

        self.add_output('CommLOS', np.zeros(num_times), units=None,
                        desc='Satellite to ground station line of sight over time')

        # A = np.arange(0,num_times)
        # B = np.repeat(A,3)
        # rows = B.flatten()

        # E = np.arange(0,3*num_times,num_times)
        # F = np.arange(0,num_times).reshape(num_times,1)
        # cols = (E+F).flatten()

        cols = np.arange(3 * num_times)
        rows = np.outer(
            np.ones(3, int),
            np.arange(num_times),
        ).flatten()


        self.declare_partials('CommLOS','r_b2g_I', rows=rows, cols=cols)
        self.declare_partials('CommLOS','r_e2g_I', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        num_times = self.options['num_times']

        r_b2g_I = inputs['r_b2g_I']
        r_e2g_I = inputs['r_e2g_I']

        Rb = 100.0
        proj = np.sum(r_b2g_I * r_e2g_I, axis=0)/self.Re
        outputs['CommLOS'] = sigmoid(proj+Rb)
        print(outputs['CommLOS'])

    def compute_partials(self, inputs, partials):

        num_times = self.options['num_times']

        r_b2g_I = inputs['r_b2g_I']
        r_e2g_I = inputs['r_e2g_I']

        Rb = 100.0
        proj = np.sum(r_b2g_I * r_e2g_I, axis=0)/self.Re

        grad_proj = sigmoid_grad(proj+Rb)

        dLOS_drb = partials['CommLOS','r_b2g_I'].reshape(3,num_times)
        dLOS_dre = partials['CommLOS','r_e2g_I'].reshape(3,num_times)

        dLOS_drb = grad_proj * r_e2g_I/self.Re
        dLOS_dre = grad_proj * r_b2g_I/self.Re

        # partials['CommLOS','r_b2g_I'] = (grad_proj * r_e2g_I/self.Re).flatten()
        # partials['CommLOS','r_e2g_I'] = (grad_proj * r_b2g_I/self.Re).flatten()

if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    n = 4

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('r_b2g_I', val=10000 * np.random.random((3, n)))
    comp.add_output('r_e2g_I', val=10000 * np.random.random((3, n)))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = CommLOSComp(
        num_times=n,
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)

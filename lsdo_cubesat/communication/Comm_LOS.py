"""
Determine if the Satellite has line of sight with the ground stations
"""

from six.moves import range

import numpy as np

from csdl import CustomExplicitOperation
from lsdo_cubesat.utils.utils import get_array_indices


def LOS(x):
    b = np.argsort(x)
    x[b[:50]] = 1
    x[b[50:]] = 0
    return x


class CommLOSComp(CustomExplicitOperation):
    Re = 6378.137

    def initialize(self):
        self.parameters.declare('num_times', types=int)

    def define(self):
        num_times = self.parameters['num_times']

        self.add_input('r_b2g_I',
                       np.zeros((3, num_times)),
                       units='km',
                       desc='Position vector from satellite to ground station '
                       'in Earth-centered inertial frame over time')

        self.add_input('r_e2g_I',
                       np.zeros((3, num_times)),
                       units='km',
                       desc='Position vector from earth to ground station in '
                       'Earth-centered inertial frame over time')

        self.add_output(
            'CommLOS',
            np.zeros(num_times),
            units=None,
            desc='Satellite to ground station line of sight over time')

        cols = np.arange(3 * num_times)
        rows = np.outer(
            np.ones(3, int),
            np.arange(num_times),
        ).flatten()
        self.declare_derivatives('CommLOS', 'r_b2g_I', rows=rows, cols=cols)
        self.declare_derivatives('CommLOS', 'r_e2g_I', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        num_times = self.parameters['num_times']

        r_b2g_I = inputs['r_b2g_I']
        r_e2g_I = inputs['r_e2g_I']

        proj = np.sum(r_b2g_I * r_e2g_I, axis=0) / self.Re

        # np.savetxt("rundata/optics_r_b2g_I.csv", r_b2g_I, header="r_b2g_I")

        # np.savetxt("rundata/optics_r_e2g_I.csv", r_e2g_I, header="r_e2g_I")

        # np.savetxt("rundata/projection.csv",
        #            proj,
        #            header="proj",
        #            delimiter=',')

        outputs['CommLOS'] = LOS(proj)

    # def compute_partials(self, inputs, partials):

    #     num_times = self.parameters['num_times']

    #     r_b2g_I = inputs['r_b2g_I']
    #     r_e2g_I = inputs['r_e2g_I']

    #     Rb = 7e4
    #     proj = np.sum(r_b2g_I * r_e2g_I, axis=0) / self.Re

    #     grad_proj = sigmoid_grad(proj - Rb)

    #     dLOS_drb = partials['CommLOS', 'r_b2g_I'].reshape(3, num_times)
    #     dLOS_dre = partials['CommLOS', 'r_e2g_I'].reshape(3, num_times)

    #     dLOS_drb = grad_proj * r_e2g_I / self.Re
    #     dLOS_dre = grad_proj * r_b2g_I / self.Re

    # partials['CommLOS','r_b2g_I'] = (grad_proj * r_e2g_I/self.Re).flatten()
    # partials['CommLOS','r_e2g_I'] = (grad_proj * r_b2g_I/self.Re).flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp
    import numpy as np
    n = 4
    Re = 6378.137

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('r_b2g_I', val=1000 * np.random.random((3, n)))
    comp.add_output('r_e2g_I', val=1000 * np.random.random((3, n)))
    prob.model.add('inputs_comp', comp, promotes=['*'])

    comp = CommLOSComp(num_times=n, )
    prob.model.add('comp', comp, promotes=['*'])

    prob.setup()
    print(np.sum(prob['r_b2g_I'] * prob['r_e2g_I'], axis=0) / Re)
    a = np.sum(prob['r_b2g_I'] * prob['r_e2g_I'], axis=0) / Re
    # print(sigmoid(a))

    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # print(prob['CommLOS'])

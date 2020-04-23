"""
Determine if the Satellite has line of sight with the ground stations
"""

from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class CommLOSComp(ExplicitComponent):
    Re = 6378.137

    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        opts = self.options
        num_times = opts['num_times']

        self.add_input('r_b2g_I', np.zeros((3, num_times)),
                       desc='Position vector from satellite to ground station '
                            'in Earth-centered inertial frame over time')

        self.add_input('r_e2g_I', np.zeros((3, num_times)),
                       desc='Position vector from earth to ground station in '
                            'Earth-centered inertial frame over time')

        # Outputs
        self.add_output('CommLOS', np.zeros(num_times), units=None,
                        desc='Satellite to ground station line of sight over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        num_times = self.options['num_times']

        r_b2g_I = inputs['r_b2g_I']
        r_e2g_I = inputs['r_e2g_I']
        CommLOS = outputs['CommLOS']

        Rb = 100.0
        for i in range(0, num_times):
            proj = np.dot(r_b2g_I[:, i], r_e2g_I[:, i]) / self.Re

            if proj > 0:
                CommLOS[i] = 0.
            elif proj < -Rb:
                CommLOS[i] = 1.
            else:
                x = (proj - 0) / (-Rb - 0)
                CommLOS[i] = 3 * x ** 2 - 2 * x ** 3

    def compute_partials(self, inputs, partials):

        num_times = self.options['num_times']

        r_b2g_I = inputs['r_b2g_I']
        r_e2g_I = inputs['r_e2g_I']

        self.dLOS_drb = np.zeros((num_times, 3))
        self.dLOS_dre = np.zeros((num_times, 3))

        Rb = 10.0
        for i in range(0, num_times):

            proj = np.dot(r_b2g_I[:, i], r_e2g_I[:, i]) / self.Re

            if proj > 0:
                self.dLOS_drb[i, :] = 0.
                self.dLOS_dre[i, :] = 0.
            elif proj < -Rb:
                self.dLOS_drb[i, :] = 0.
                self.dLOS_dre[i, :] = 0.
            else:
                x = (proj - 0) / (-Rb - 0)
                dx_dproj = -1. / Rb
                dLOS_dx = 6 * x - 6 * x ** 2
                dproj_drb = r_e2g_I[:, i]
                dproj_dre = r_b2g_I[:, i]

                self.dLOS_drb[i, :] = dLOS_dx * dx_dproj * dproj_drb
                self.dLOS_dre[i, :] = dLOS_dx * dx_dproj * dproj_dre


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    n = 3

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('r_b2g_I', val=10 * np.random.random((3, n)))
    comp.add_output('r_e2g_I', val=10 * np.random.random((3, n)))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = CommLOSComp(
        num_times=n,
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)

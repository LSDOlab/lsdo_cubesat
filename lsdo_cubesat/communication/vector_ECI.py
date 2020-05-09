import numpy as np

from openmdao.api import ExplicitComponent


class Comm_VectorECI(ExplicitComponent):
    """
    Determine vector between satellite and ground station.
    """
    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']

        self.add_input('r_e2g_I', np.zeros((3, num_times)), units='km',
                       desc='Position vector from earth to ground station in '
                            'Earth-centered inertial frame over time')

        self.add_input('r_e2b_I', np.zeros((6, num_times)), units=None,
                       desc='Position and velocity vector from earth to satellite '
                            'in Earth-centered inertial frame over time')

        self.add_output('r_b2g_I', np.zeros((3, num_times)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in Earth-centered inertial frame over time')

    def compute(self, inputs, outputs):

        outputs['r_b2g_I'] = inputs['r_e2g_I'] - inputs['r_e2b_I'][:3, :]
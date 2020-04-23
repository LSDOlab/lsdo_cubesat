import numpy as np

from openmdao.api import ExplicitComponent

class VectorBodyComp(ExplicitComponent):
    """
    Transform from body to inertial frame.
    """

    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']

        # Inputs
        self.add_input('r_b2g_I', np.zeros((3, num_times)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in Earth-centered inertial frame over time')

        self.add_input('Rot_b_i', np.zeros((3, 3, num_times)), units=None,
                       desc='Rotation matrix from body-fixed frame to Earth-centered'
                            'inertial frame over time')

        # Outputs
        self.add_output('r_b2g_B', np.zeros((3, num_times)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in body-fixed frame over time')

    def compute(self, inputs, outputs):
        num_times = self.options['num_times']

        r_b2g_I = inputs['r_b2g_I']
        Rot_b_i = inputs['Rot_b_i']
        r_b2g_B = outputs['r_b2g_B']

        for i in range(0, num_times):
            r_b2g_B[:, i] = np.dot(Rot_b_i[:, :, i], r_b2g_I[:, i])

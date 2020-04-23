import numpy as np
from openmdao.api import ExplicitComponent

class AntennaInertialComp(ExplicitComponent):
    """
    Transform from body to inertial frame.
    """
    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        opts = self.options
        num_times = opts['num_times']

        # Inputs
        self.add_input('r_b2g_I', np.zeros((3, num_times)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in Earth-centered inertial frame over time')

        self.add_input('O_BI', np.zeros((3, 3, num_times)), units=None,
                       desc='Rotation matrix from body-fixed frame to Earth-centered'
                            'inertial frame over time')

        # Outputs
        self.add_output('r_b2g_B', np.zeros((3, num_times)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in body-fixed frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        num_times = self.options['num_times']
        r_b2g_I = inputs['r_b2g_I']
        O_BI = inputs['O_BI']
        r_b2g_B = outputs['r_b2g_B']

        for i in range(0, num_times):
            r_b2g_B[:, i] = np.dot(O_BI[:, :, i], r_b2g_I[:, i])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        num_times = self.options['num_times']
        r_b2g_I = inputs['r_b2g_I']
        O_BI = inputs['O_BI']

        self.J1 = np.zeros((num_times, 3, 3, 3))

        for k in range(0, 3):
            for v in range(0, 3):
                self.J1[:, k, k, v] = r_b2g_I[v, :]

        self.J2 = np.transpose(O_BI, (2, 0, 1))

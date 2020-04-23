import numpy as np
from openmdao.api import ExplicitComponent

class AntennaBodyComp(ExplicitComponent):
    """
    Transform from antenna to body frame
    """
    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        opts = self.options
        num_times = opts['num_times']

        # Inputs
        self.add_input('r_b2g_B', np.zeros((3, num_times)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in body-fixed frame over time')

        self.add_input('O_AB', np.zeros((3, 3, num_times)), units=None,
                       desc='Rotation matrix from antenna angle to body-fixed '
                            'frame over time')

        # Outputs
        self.add_output('r_b2g_A', np.zeros((3, num_times)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in antenna angle frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        num_times = self.options['num_times']
        outputs['r_b2g_A'] = computepositionrotd(num_times, inputs['r_b2g_B'],
                                                 inputs['O_AB'])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        num_times = self.options['num_times']
        self.J1, self.J2 = computepositionrotdjacobian(num_times, inputs['r_b2g_B'],
                                                       inputs['O_AB'])

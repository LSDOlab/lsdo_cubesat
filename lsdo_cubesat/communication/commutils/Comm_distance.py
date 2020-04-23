import numpy as numpy
from openmdao.api import ExplicitComponent

class StationSatelliteDistance(ExplicitComponent):
    """
    Calculates distance from ground station to satellite.
    """

    def initialize(self):
        self.options.declare('num_times',types=int)

    def setup(self):
        num_times = self.options['num_times']
        # Inputs
        self.add_input('r_b2g_A', np.zeros((3, num_times)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in antenna angle frame over time')

        # Outputs
        self.add_output('GSdist', np.zeros(num_times), units='km',
                        desc='Distance from ground station to satellite over time')
        self.J = np.zeros((num_times, 3))
        
    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        num_times = self.options['num_times']
        r_b2g_A = inputs['r_b2g_A']
        GSdist = outputs['GSdist']

        for i in range(0, num_times):
            GSdist[i] = np.dot(r_b2g_A[:, i], r_b2g_A[:, i])**0.5

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        num_times = self.options['num_times']
        r_b2g_A = inputs['r_b2g_A']

        for i in range(0, num_times):
            norm = np.dot(r_b2g_A[:, i], r_b2g_A[:, i])**0.5
            if norm > 1e-10:
                self.J[i, :] = r_b2g_A[:, i] / norm
            else:
                self.J[i, :] = 0.


if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem, IndepVarComp, Group

    group = Group()
    comp = IndepVarComp()
    num_times = 3
    comp.add_output('r_b2g_A', val= np.random.random((3, num_times)))

    group.add_subsystem('Inputcomp', comp, promotes=['*'])
    group.add_subsystem('distance',
                        StationSatelliteDistance(num_times=num_times),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)

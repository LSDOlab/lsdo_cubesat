import numpy as np
from openmdao.api import ExplicitComponent

class GSposECIComp(ExplicitComponent):
    """
    Convert time history of ground station position from earth frame
    to inertial frame.
    """
    def initialize(self):
        self.options.declare('num_times',types=int)

    def setup(self):
        num_times = self.options['num_times']
        # Inputs
        self.add_input('O_IE', np.zeros((3, 3, num_times)), units=None,
                       desc='Rotation matrix from Earth-centered inertial '
                            'frame to Earth-fixed frame over time')

        self.add_input('r_e2g_E', np.zeros((3, num_times)), units='km',
                       desc='Position vector from earth to ground station in '
                            'Earth-fixed frame over time')

        # Outputs
        self.add_output('r_e2g_I', np.zeros((3, num_times)), units='km',
                        desc='Position vector from earth to ground station in '
                             'Earth-centered inertial frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        num_times = self.options['num_times']
        O_IE = inputs['O_IE']
        r_e2g_E = inputs['r_e2g_E']
        r_e2g_I = outputs['r_e2g_I']

        for i in range(0, num_times):
            r_e2g_I[:, i] = np.dot(O_IE[:, :, i], r_e2g_E[:, i])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        num_times = self.options['num_times']
        O_IE = inputs['O_IE']
        r_e2g_E = inputs['r_e2g_E']

        self.J1 = np.zeros((num_times, 3, 3, 3))

        for k in range(0, 3):
            for v in range(0, 3):
                self.J1[:, k, k, v] = r_e2g_E[v, :]

        self.J2 = np.transpose(O_IE, (2, 0, 1))

if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem, IndepVarComp, Group

    group = Group()
    comp = IndepVarComp()
    num_times = 3
    
    comp.add_output('O_IE', val= np.random.random((3, 3, num_times)))
    comp.add_output('r_e2g_E', val= np.random.random((3, num_times)))

    group.add_subsystem('Inputcomp', comp, promotes=['*'])
    group.add_subsystem('ground_station_position',
                        GSposECIComp(num_times=num_times),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)

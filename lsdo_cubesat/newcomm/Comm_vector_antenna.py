import numpy as np
from openmdao.api import ExplicitComponent
from lsdo_cubesat.new_comm.kinematics import computepositionrotd,computepositionrotdjacobian

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

        self.add_input('Rot_AB', np.zeros((3, 3, num_times)), units=None,
                       desc='Rotation matrix from antenna angle to body-fixed '
                            'frame over time')

        # Outputs
        self.add_output('r_b2g_A', np.zeros((3, num_times)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in antenna angle frame over time')

    def compute(self, inputs, outputs):

        num_times = self.options['num_times']
        outputs['r_b2g_A'] = computepositionrotd(num_times, inputs['r_b2g_B'],
                                                 inputs['Rot_AB'])

    def compute_partials(self, inputs, partials):

        num_times = self.options['num_times']
        self.J1, self.J2 = computepositionrotdjacobian(num_times, inputs['r_b2g_B'],
                                                       inputs['Rot_AB'])


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    num_times = 3

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('r_b2g_B', val=10 * np.random.random((3, num_times)))
    comp.add_output('Rot_AB', val=10 * np.random.random((3, 3, num_times)))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = AntennaBodyComp(
        num_times=num_times,
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)
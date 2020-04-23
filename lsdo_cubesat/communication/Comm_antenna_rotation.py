import numpy as np
from openmdao.api import ExplicitComponent

class AntennaRotationComp(ExplicitComponent):
    """
    Fixed antenna angle to time history of the quaternion.
    """

    def initialize(self):
        self.options.declare('num_times',types=int)
        self.options.declare('antAngle_name',types=int)

    def setup(self):
        num_times = self.options['num_times']
        antAngle_name

        # Inputs
        self.add_input('antAngle', 0.0, units='rad')

        # Outputs
        self.add_output('q_A', np.zeros((4, num_times)), units=None,
                        desc='Quarternion matrix in antenna angle frame over time')

        self.dq_dt = np.zeros(4)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        antAngle = inputs['antAngle']
        q_A = outputs['q_A']

        rt2 = np.sqrt(2)
        q_A[0, :] = np.cos(inputs[antAngle_name]/2.)
        q_A[1, :] = np.sin(inputs[antAngle_name]/2.) / rt2
        q_A[2, :] = - np.sin(inputs[antAngle_name]/2.) / rt2
        q_A[3, :] = 0.0

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        antAngle = inputs['antAngle']

        rt2 = np.sqrt(2)
        self.dq_dt[0] = - np.sin(antAngle / 2.) / 2.
        self.dq_dt[1] = np.cos(antAngle / 2.) / rt2 / 2.
        self.dq_dt[2] = - np.cos(antAngle / 2.) / rt2 / 2.
        self.dq_dt[3] = 0.0


if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem, IndepVarComp, Group

    group = Group()
    comp = IndepVarComp()
    num_times = 3
    antAngle_name = 2.2
    comp.add_output('antAngle_name', val= 2.2)

    group.add_subsystem('Inputcomp', comp, promotes=['*'])
    group.add_subsystem('antenna_angle',
                        AntennaRotationComp(num_times=num_times),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)

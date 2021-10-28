import numpy as np

from openmdao.api import ExplicitComponent


class AntRotationComp(ExplicitComponent):
    """
    Fixed antenna angle to time history of the quaternion.
    """
    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']

        self.add_input('antAngle',
                       0.0,
                       units='rad',
                       desc='Fiexed antenna angle')

        self.add_output(
            'q_A',
            np.zeros((4, num_times)),
            units=None,
            desc='Quarternion matrix in antenna angle frame over time')

        self.declare_partials('q_A', 'antAngle')

    def compute(self, inputs, outputs):
        antAngle = inputs['antAngle']

        rt2 = np.sqrt(2)
        outputs['q_A'][0, :] = np.cos(antAngle / 2.)
        outputs['q_A'][1, :] = np.sin(antAngle / 2.) / rt2
        outputs['q_A'][2, :] = -np.sin(antAngle / 2.) / rt2
        outputs['q_A'][3, :] = 0.0

    def compute_partials(self, inputs, partials):
        num_times = self.options['num_times']
        antAngle = inputs['antAngle']

        rt2 = np.sqrt(2)
        dq_dt = np.zeros((4, num_times))
        dq_dt[0, :] = -np.sin(antAngle / 2.) / 2.
        dq_dt[1, :] = np.cos(antAngle / 2.) / rt2 / 2.
        dq_dt[2, :] = -np.cos(antAngle / 2.) / rt2 / 2.
        dq_dt[3, :] = 0.0

        partials['q_A', 'antAngle'] = dq_dt.flatten()


if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem, IndepVarComp, Group

    group = Group()
    comp = IndepVarComp()
    num_times = 3

    comp.add_output('antAngle', val=1.0, units='rad')

    group.add('Inputcomp', comp, promotes=['*'])
    group.add('antenna_angle',
              AntRotationComp(num_times=num_times),
              promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)

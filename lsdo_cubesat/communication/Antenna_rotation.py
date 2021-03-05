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

        self.add_input('antenna_angle',
                       0.0,
                       units='rad',
                       desc='Fiexed antenna angle')

        self.add_output(
            'antenna_orientation',
            np.zeros((4, num_times)),
            units=None,
            desc='Quarternion matrix in antenna angle frame over time')

        self.declare_partials('antenna_orientation', 'antenna_angle')

    def compute(self, inputs, outputs):
        antenna_angle = inputs['antenna_angle']

        rt2 = np.sqrt(2)
        outputs['antenna_orientation'][0, :] = np.cos(antenna_angle / 2.)
        outputs['antenna_orientation'][1, :] = np.sin(antenna_angle / 2.) / rt2
        outputs['antenna_orientation'][
            2, :] = -np.sin(antenna_angle / 2.) / rt2
        outputs['antenna_orientation'][3, :] = 0.0

    def compute_partials(self, inputs, partials):
        num_times = self.options['num_times']
        antenna_angle = inputs['antenna_angle']

        rt2 = np.sqrt(2)
        dq_dt = np.zeros((4, num_times))
        dq_dt[0, :] = -np.sin(antenna_angle / 2.) / 2.
        dq_dt[1, :] = np.cos(antenna_angle / 2.) / rt2 / 2.
        dq_dt[2, :] = -np.cos(antenna_angle / 2.) / rt2 / 2.
        dq_dt[3, :] = 0.0

        partials['antenna_orientation', 'antenna_angle'] = dq_dt.flatten()


if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem, IndepVarComp, Group

    group = Group()
    comp = IndepVarComp()
    num_times = 3

    comp.add_output('antenna_angle', val=1.0, units='rad')

    group.add_subsystem('Inputcomp', comp, promotes=['*'])
    group.add_subsystem('antenna_angle',
                        AntRotationComp(num_times=num_times),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)

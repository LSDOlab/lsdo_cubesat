import numpy as np
from openmdao.api import ExplicitComponent

class EarthSpinComp(ExplicitComponent):
    """
    Returns earth quaternion matrix over time
    """

    def initialize(self):
        self.options.declare('num_times', types=int)
        # self.options.declare('launch_date', types=float)

    def setup(self):
        num_times = self.options['num_times']
        # launch_date = self.options['launch_date']

        self.add_input('times', shape=num_times, units='s')
        
        self.add_output('q_E', shape=(4,num_times), units=None,
                desc = 'Quaternion matrix in Earth-fixed frame over time')

        cols = np.arange(0,num_times)
        cols = np.tile(cols,4)

        rows = np.arange(0,4*num_times)
        self.declare_partials('q_E','times', rows=rows,cols=cols)

    def compute(self, inputs, outputs):
        num_times = self.options['num_times']

        t = inputs['times']

        fact = np.pi / 3600.0 / 24.0
        theta = fact * t

        outputs['q_E'][0, :] = np.cos(theta)
        outputs['q_E'][3, :] = -np.sin(theta)

    def compute_partials(self, inputs, partials):
        num_times = self.options['num_times']

        t = inputs['times']
        print(partials['q_E','times'].shape)

        fact = np.pi / 3600.0 / 24.0
        theta = fact * t

        dq_dt = np.zeros((4,num_times))
        dq_dt[0,:] = -np.sin(theta) * fact
        dq_dt[3,:] = -np.cos(theta) * fact

        partials['q_E','times'] = dq_dt.flatten()


if __name__ == '__main__':

    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp

    num_times = 30

    group = Group()

    comp = IndepVarComp()
    comp.add_output('times', val=np.arange(num_times))

    group.add_subsystem('Inputcomp', comp, promotes=['*'])

    group.add_subsystem('EarthSpinComp',
                        EarthSpinComp(num_times=num_times),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)







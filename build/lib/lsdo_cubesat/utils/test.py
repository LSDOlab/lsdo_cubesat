import numpy as numpy
from openmdao.api import ExplicitComponent


class Comm_EarthsSpin(ExplicitComponent):
    """
    Returns the Earth quaternion as a function of time.
    """

    def __init__(self, n):
        super(Comm_EarthsSpin, self).__init__()

        self.n = n

    def setup(self):
        # Inputs
        self.add_input('t', np.zeros(self.n), units='s',
                       desc='Time')

        # Outputs
        self.add_output('q_E', np.zeros((4, self.n)), units=None,
                        desc='Quarternion matrix in Earth-fixed frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        t = inputs['t']
        q_E = outputs['q_E']

        fact = np.pi / 3600.0 / 24.0
        theta = fact * t

        q_E[0, :] = np.cos(theta)
        q_E[3, :] = -np.sin(theta)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        t = inputs['t']

        ntime = self.n
        self.dq_dt = np.zeros((ntime, 4))

        fact = np.pi / 3600.0 / 24.0
        theta = fact * t

        self.dq_dt[:, 0] = -np.sin(theta) * fact
        self.dq_dt[:, 3] = -np.cos(theta) * fact

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        if mode == 'fwd':
            if 't' in d_inputs:
                for k in range(4):
                    d_outputs['q_E'][k, :] += self.dq_dt[:, k] * d_inputs['t']
        else:
            if 't' in d_inputs:
                for k in range(4):
                    d_inputs['t'] += self.dq_dt[:, k] * d_outputs['q_E'][k, :]



if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem, IndepVarComp, Group

    group = Group()
    comp = IndepVarComp()
    n = 3
    comp.add_output('t', val=np.array([3600*2,3600*4,3600*6]))

    group.add_subsystem('Inputcomp', comp, promotes=['*'])
    group.add_subsystem('frameconvert',
                        Comm_EarthsSpin(n=n),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()
    print(prob['t'])
    print(prob['q_E'])

    prob.check_partials(compact_print=True)

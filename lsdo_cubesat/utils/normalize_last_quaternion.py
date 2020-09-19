from openmdao.api import ExplicitComponent
import numpy as np


class NormalizeLastQuaternion(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_times', default=1, types=int)

    def setup(self):
        num_times = self.options['num_times']
        self.add_input('quaternions', shape=(4, num_times))
        self.add_output('normalized_quaternions', shape=(4, num_times))
        inds = np.arange(4 * num_times - 4, 4 * num_times)
        rows = np.einsum('i,j->ij', inds, np.ones(4)).flatten()
        cols = np.einsum('i,j->ji', inds, np.ones(4)).flatten()
        self.declare_partials(
            'normalized_quaternions',
            'quaternions',
            rows=rows,
            cols=cols,
        )

    def compute(self, inputs, outputs):
        q = inputs['quaternions']
        outputs['normalized_quaternions'] = q
        outputs['normalized_quaternions'][:, -1] = q[:, -1] / np.linalg.norm(
            q[:, -1])

    def compute_partials(self, inputs, partials):
        q = inputs['quaternions']
        qnorm = np.linalg.norm(q[:, -1])

        dqdq = qnorm * np.eye(4) - np.einsum('i,j->ij', q[:, -1],
                                             q[:, -1]) / qnorm
        dqdq /= qnorm**2

        partials['normalized_quaternions', 'quaternions'] = dqdq.flatten()


if __name__ == '__main__':

    from openmdao.api import Problem, IndepVarComp
    import matplotlib.pyplot as plt
    np.random.seed(0)

    num_times = 1

    q = np.random.rand(4, num_times) * 100
    # qnorm = np.linalg.norm(q, axis=0)
    # qnorm = np.einsum('j,i->ij', qnorm, np.ones(4))
    # q /= qnorm

    prob = Problem()
    inputs = IndepVarComp()
    inputs.add_output(
        'quaternions',
        val=q,
    )
    prob.model.add_subsystem(
        'inputs',
        inputs,
        promotes=['*'],
    )
    prob.model.add_subsystem(
        'normalize',
        NormalizeLastQuaternion(num_times=num_times),
        promotes=['*'],
    )
    prob.setup(force_alloc_complex=True)
    prob.check_partials(compact_print=True)

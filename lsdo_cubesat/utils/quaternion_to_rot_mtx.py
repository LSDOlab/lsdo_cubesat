import numpy as np

from openmdao.api import ExplicitComponent
from lsdo_cubesat.utils.utils import get_array_indices


class QuaternionToRotMtx(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']

        self.add_input('quaternions', shape=(4, num_times))
        self.add_output('rot_mtx_i_b_3x3xn', shape=(3, 3, num_times))

        mtx_indices = get_array_indices(*(3, 3, num_times))
        A = mtx_indices.flatten()
        rows = np.repeat(A, 4).flatten()

        B = np.arange(0, 4 * num_times, num_times)
        C = np.arange(0, num_times).reshape(num_times, 1)
        D = (B + C).flatten()
        E = np.tile(D, 9)
        cols = E.flatten()

        self.declare_partials('rot_mtx_i_b_3x3xn',
                              'quaternions',
                              rows=rows,
                              cols=cols)

    def compute(self, inputs, outputs):
        q = inputs['quaternions']
        outputs['rot_mtx_i_b_3x3xn'][0,
                                     0, :] = 1 - 2 * (q[2, :]**2 + q[3, :]**2)
        outputs['rot_mtx_i_b_3x3xn'][0, 1, :] = 2 * (q[1, :] * q[2, :] +
                                                     q[0, :] * q[3, :])
        outputs['rot_mtx_i_b_3x3xn'][0, 2, :] = 2 * (q[1, :] * q[3, :] -
                                                     q[0, :] * q[2, :])

        outputs['rot_mtx_i_b_3x3xn'][1, 0, :] = 2 * (q[1, :] * q[2, :] -
                                                     q[0, :] * q[3, :])
        outputs['rot_mtx_i_b_3x3xn'][1,
                                     1, :] = 1 - 2 * (q[1, :]**2 + q[3, :]**2)
        outputs['rot_mtx_i_b_3x3xn'][1, 2, :] = 2 * (q[2, :] * q[3, :] +
                                                     q[0, :] * q[1, :])

        outputs['rot_mtx_i_b_3x3xn'][2, 0, :] = 2 * (q[1, :] * q[3, :] +
                                                     q[0, :] * q[2, :])
        outputs['rot_mtx_i_b_3x3xn'][2, 1, :] = 2 * (q[2, :] * q[3, :] -
                                                     q[0, :] * q[1, :])
        outputs['rot_mtx_i_b_3x3xn'][2,
                                     2, :] = 1 - 2 * (q[1, :]**2 + q[2, :]**2)

    def compute_partials(self, inputs, partials):
        num_times = self.options['num_times']
        q = inputs['quaternions']
        dRdq = np.zeros(4 * 9 * num_times).reshape((9, num_times, 4))
        # R[0,0]
        dRdq[0, :, 2] = -4 * q[2, :]
        dRdq[0, :, 3] = -4 * q[3, :]

        # R[0,1]
        dRdq[1, :, 0] = 2 * q[3, :]
        dRdq[1, :, 1] = 2 * q[2, :]
        dRdq[1, :, 2] = 2 * q[1, :]
        dRdq[1, :, 3] = 2 * q[0, :]

        # R[0,2]
        dRdq[2, :, 0] = -2 * q[2, :]
        dRdq[2, :, 1] = 2 * q[3, :]
        dRdq[2, :, 2] = -2 * q[0, :]
        dRdq[2, :, 3] = 2 * q[1, :]

        # R[1,0]
        dRdq[3, :, 0] = -2 * q[3, :]
        dRdq[3, :, 1] = 2 * q[2, :]
        dRdq[3, :, 2] = 2 * q[1, :]
        dRdq[3, :, 3] = -2 * q[0, :]

        # R[1,1]
        dRdq[4, :, 1] = -4 * q[1, :]
        dRdq[4, :, 3] = -4 * q[3, :]

        # R[1,2]
        dRdq[5, :, 0] = 2 * q[1, :]
        dRdq[5, :, 1] = 2 * q[0, :]
        dRdq[5, :, 2] = 2 * q[3, :]
        dRdq[5, :, 3] = 2 * q[2, :]

        # R[2,0]
        dRdq[6, :, 0] = 2 * q[2, :]
        dRdq[6, :, 1] = 2 * q[3, :]
        dRdq[6, :, 2] = 2 * q[0, :]
        dRdq[6, :, 3] = 2 * q[1, :]

        # R[2,1]
        dRdq[7, :, 0] = -2 * q[1, :]
        dRdq[7, :, 1] = -2 * q[0, :]
        dRdq[7, :, 2] = 2 * q[3, :]
        dRdq[7, :, 3] = 2 * q[2, :]

        # R[2,2]
        dRdq[8, :, 1] = -4 * q[1, :]
        dRdq[8, :, 2] = -4 * q[2, :]

        partials['rot_mtx_i_b_3x3xn', 'quaternions'] = dRdq.flatten()


if __name__ == '__main__':

    from openmdao.api import Problem, Group, IndepVarComp
    np.random.seed(0)

    class TestGroup(Group):
        def setup(self):
            num_times = 50
            q = np.random.rand(4, num_times) * 100
            # q /= np.einsum(
            #     'i,j->ij', np.ones(4), np.linalg.norm(q, axis=0, ord=2))
            inputs = IndepVarComp()
            inputs.add_output('quaternions', val=q, shape=(4, num_times))
            self.add_subsystem('inputs', inputs, promotes=['*'])
            self.add_subsystem('q2RotMtx',
                               QuaternionToRotMtx(num_times=num_times),
                               promotes=['*'])

    prob = Problem()
    prob.model = TestGroup()
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
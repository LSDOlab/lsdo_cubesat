import numpy as np

from openmdao.api import ExplicitComponent

# https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf


def get_mtx_norms(mtx):
    norm0 = np.linalg.norm(mtx, axis=0)
    norm1 = np.linalg.norm(mtx, axis=1)
    mtx


class RotMtxToRollPitchYaw(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('mtx_name', types=str)

    def setup(self):
        num_times = self.options['num_times']
        mtx_name = self.options['mtx_name']
        self.add_input(mtx_name, shape=(3, 3, num_times))
        self.add_output('roll', shape=(num_times))
        self.add_output('pitch', shape=(num_times))
        self.add_output('yaw', shape=(num_times))

        rows = np.arange(num_times)
        rows = np.concatenate((rows, rows))
        cols1 = np.arange(num_times) + 5 * num_times
        cols2 = np.arange(num_times) + 8 * num_times
        cols = np.concatenate((cols1, cols2))
        self.declare_partials(
            'roll',
            mtx_name,
            rows=rows,
            cols=cols,
        )

        self.declare_partials(
            'pitch',
            mtx_name,
            rows=np.arange(num_times),
            cols=np.arange(num_times) + 2 * num_times,
        )

        cols1 = np.arange(num_times) + num_times
        cols2 = np.arange(num_times)
        cols = np.concatenate((cols1, cols2))
        self.declare_partials(
            'yaw',
            mtx_name,
            rows=rows,
            cols=cols,
        )

    def compute(self, inputs, outputs):
        mtx_name = self.options['mtx_name']
        R = inputs[mtx_name]

        # https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf
        # (67)
        outputs['roll'] = np.arctan2(R[1, 2, :], R[2, 2, :])

        if np.any(R[0, 2, :] > 1) or np.any(R[0, 2, :] < -1):
            print('R[0, 2, :]')
            print(R[0, 2, :])
        outputs['pitch'] = -np.arcsin(R[0, 2, :])
        outputs['yaw'] = np.arctan2(R[0, 1, :], R[0, 0, :])

    def compute_partials(self, inputs, partials):
        mtx_name = self.options['mtx_name']
        R = inputs[mtx_name]
        norm2 = R[2, 2, :]**2 + R[1, 2, :]**2
        dx = -R[1, 2, :] / norm2
        dy = R[2, 2, :] / norm2
        partials['roll', 'rot_mtx_b_i_3x3xn'] = np.concatenate(
            (dy.flatten(), dx.flatten()))
        partials['pitch', mtx_name] = -1.0 / np.sqrt(1.0 - R[0, 2, :]**2)

        norm2 = R[0, 0, :]**2 + R[0, 1, :]**2
        dx = -R[0, 1, :] / norm2
        dy = R[0, 0, :] / norm2
        partials['yaw', 'rot_mtx_b_i_3x3xn'] = np.concatenate(
            (dy.flatten(), dx.flatten()))


if __name__ == '__main__':

    from openmdao.api import Problem, Group, IndepVarComp
    from lsdo_cubesat.utils.quaternion_to_rot_mtx import QuaternionToRotMtx
    import matplotlib.pyplot as plt
    # np.random.seed(0)

    num_times = 500

    q = np.random.rand(4, num_times)
    qnorm = np.linalg.norm(q, axis=0)
    qnorm = np.einsum('j,i->ij', qnorm, np.ones(4))
    q /= qnorm

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
        'q2RotMtx',
        QuaternionToRotMtx(num_times=num_times),
        promotes=['*'],
    )
    prob.model.add_subsystem(
        'comp',
        RotMtxToRollPitchYaw(
            mtx_name='rot_mtx_b_i_3x3xn',
            num_times=num_times,
        ),
        promotes=['*'],
    )
    prob.setup()
    prob.run_model()

    print(
        np.sum(np.linalg.norm(prob['rot_mtx_b_i_3x3xn'], axis=0)) -
        3 * num_times)
    print(
        np.sum(np.linalg.norm(prob['rot_mtx_b_i_3x3xn'], axis=1)) -
        3 * num_times)

    if num_times < 1001:
        prob.check_partials(compact_print=True)
    else:
        t = np.arange(num_times)
        plt.plot(t, prob['roll'])
        plt.plot(t, prob['pitch'])
        plt.plot(t, prob['yaw'])
        plt.show()

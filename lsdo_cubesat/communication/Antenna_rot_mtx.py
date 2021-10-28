import numpy as np
from csdl import CustomExplicitOperation
from lsdo_cubesat.utils.utils import get_array_indices


class AntennaRotationMtx(CustomExplicitOperation):
    """
    Translate antenna angle into the body frame.
    """
    def initialize(self):
        self.parameters.declare('num_times', types=int)

    def define(self):
        num_times = self.parameters['num_times']

        self.add_input(
            'q_A',
            np.zeros((4, num_times)),
            units=None,
            desc='Quarternion matrix in antenna angle frame over time')

        self.add_output(
            'Rot_AB',
            np.zeros((3, 3, num_times)),
            units=None,
            desc='Rotation matrix from antenna angle to body-fixed '
            'frame over time')

        mtx_indices = get_array_indices(*(3, 3, num_times))
        A = mtx_indices.flatten()
        rows = np.repeat(A, 4).flatten()

        B = np.arange(0, 4 * num_times, num_times)
        C = np.arange(0, num_times).reshape(num_times, 1)
        D = (B + C).flatten()
        E = np.tile(D, 9)
        cols = E.flatten()

        self.declare_derivatives('Rot_AB', 'q_A', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        num_times = self.parameters['num_times']

        q_A = inputs['q_A']

        outputs['Rot_AB'][0, 0, :] = 1 - 2 * q_A[2, :]**2 - 2 * q_A[3, :]**2
        outputs['Rot_AB'][
            0, 1, :] = 2 * q_A[1, :] * q_A[2, :] - 2 * q_A[3, :] * q_A[0, :]
        outputs['Rot_AB'][
            0, 2, :] = 2 * q_A[1, :] * q_A[3, :] + 2 * q_A[2, :] * q_A[0, :]
        outputs['Rot_AB'][
            1, 0, :] = 2 * q_A[1, :] * q_A[2, :] + 2 * q_A[3, :] * q_A[0, :]
        outputs['Rot_AB'][1, 1, :] = 1 - 2 * q_A[1, :]**2 - 2 * q_A[3, :]**2
        outputs['Rot_AB'][
            1, 2, :] = 2 * q_A[2, :] * q_A[3, :] - 2 * q_A[1, :] * q_A[0, :]
        outputs['Rot_AB'][
            2, 0, :] = 2 * q_A[1, :] * q_A[3, :] - 2 * q_A[2, :] * q_A[0, :]
        outputs['Rot_AB'][
            2, 1, :] = 2 * q_A[2, :] * q_A[3, :] + 2 * q_A[1, :] * q_A[0, :]
        outputs['Rot_AB'][2, 2, :] = 1 - 2 * q_A[1, :]**2 - 2 * q_A[2, :]**2

    def compute_derivatives(self, inputs, partials):
        num_times = self.parameters['num_times']
        q_A = inputs['q_A']

        dRot_dq = partials['Rot_AB', 'q_A'].reshape(9, num_times, 4)

        dRot_dq[0, :, 0] = 0
        dRot_dq[0, :, 2] = -4 * q_A[2, :]
        dRot_dq[0, :, 3] = -4 * q_A[3, :]
        dRot_dq[1, :, 0] = -2 * q_A[3, :]
        dRot_dq[1, :, 1] = 2 * q_A[2, :]
        dRot_dq[1, :, 2] = 2 * q_A[1, :]
        dRot_dq[1, :, 3] = -2 * q_A[0, :]
        dRot_dq[2, :, 0] = 2 * q_A[2, :]
        dRot_dq[2, :, 1] = 2 * q_A[3, :]
        dRot_dq[2, :, 2] = 2 * q_A[0, :]
        dRot_dq[2, :, 3] = 2 * q_A[1, :]
        dRot_dq[3, :, 0] = 2 * q_A[3, :]
        dRot_dq[3, :, 1] = 2 * q_A[2, :]
        dRot_dq[3, :, 2] = 2 * q_A[1, :]
        dRot_dq[3, :, 3] = 2 * q_A[0, :]
        dRot_dq[4, :, 0] = 0
        dRot_dq[4, :, 1] = -4 * q_A[1, :]
        dRot_dq[4, :, 2] = 0
        dRot_dq[4, :, 3] = -4 * q_A[3, :]
        dRot_dq[5, :, 0] = -2 * q_A[1, :]
        dRot_dq[5, :, 1] = -2 * q_A[0, :]
        dRot_dq[5, :, 2] = 2 * q_A[3, :]
        dRot_dq[5, :, 3] = 2 * q_A[2, :]
        dRot_dq[6, :, 0] = -2 * q_A[2, :]
        dRot_dq[6, :, 1] = 2 * q_A[3, :]
        dRot_dq[6, :, 2] = -2 * q_A[0, :]
        dRot_dq[6, :, 3] = 2 * q_A[1, :]
        dRot_dq[7, :, 0] = 2 * q_A[1, :]
        dRot_dq[7, :, 1] = 2 * q_A[0, :]
        dRot_dq[7, :, 2] = 2 * q_A[3, :]
        dRot_dq[7, :, 3] = 2 * q_A[2, :]
        dRot_dq[8, :, 0] = 0
        dRot_dq[8, :, 1] = -4 * q_A[1, :]
        dRot_dq[8, :, 2] = -4 * q_A[2, :]
        dRot_dq[8, :, 3] = 0


if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem, IndepVarComp, Group

    group = Group()
    comp = IndepVarComp()
    num_times = 3
    comp.add_output('q_A', val=np.ones((4, num_times)))

    group.add('Inputcomp', comp, promotes=['*'])
    group.add('antenna_angle',
              AntennaRotationMtx(num_times=num_times),
              promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)

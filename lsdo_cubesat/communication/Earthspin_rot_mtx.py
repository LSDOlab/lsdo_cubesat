import numpy as np

from openmdao.api import ExplicitComponent
from lsdo_cubesat.utils.utils import get_array_indices


class EarthspinRotationMtx(ExplicitComponent):
    """
    Earth spin rotation matrix from Earth-centered to Earth-fixed frame
    """
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('q_E', types=np.ndarray)

    def setup(self):
        num_times = self.options['num_times']

        self.add_output(
            'Rot_ECI_EF',
            np.zeros((3, 3, num_times)),
            units=None,
            desc='Rotation matrix from Earth-centered inertial frame'
            'to Earth-fixed frame over time')

        mtx_indices = get_array_indices(*(3, 3, num_times))
        A = mtx_indices.flatten()
        rows = np.repeat(A, 4).flatten()

        B = np.arange(0, 4 * num_times, num_times)
        C = np.arange(0, num_times).reshape(num_times, 1)
        D = (B + C).flatten()
        E = np.tile(D, 9)
        cols = E.flatten()

        self.declare_partials('Rot_ECI_EF', 'q_E', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        num_times = self.options['num_times']
        q_E = self.options['q_E']

        outputs['Rot_ECI_EF'][0,
                              0, :] = 1 - 2 * q_E[2, :]**2 - 2 * q_E[3, :]**2
        outputs['Rot_ECI_EF'][
            0, 1, :] = 2 * q_E[1, :] * q_E[2, :] - 2 * q_E[3, :] * q_E[0, :]
        outputs['Rot_ECI_EF'][
            0, 2, :] = 2 * q_E[1, :] * q_E[3, :] + 2 * q_E[2, :] * q_E[0, :]
        outputs['Rot_ECI_EF'][
            1, 0, :] = 2 * q_E[1, :] * q_E[2, :] + 2 * q_E[3, :] * q_E[0, :]
        outputs['Rot_ECI_EF'][1,
                              1, :] = 1 - 2 * q_E[1, :]**2 - 2 * q_E[3, :]**2
        outputs['Rot_ECI_EF'][
            1, 2, :] = 2 * q_E[2, :] * q_E[3, :] - 2 * q_E[1, :] * q_E[0, :]
        outputs['Rot_ECI_EF'][
            2, 0, :] = 2 * q_E[1, :] * q_E[3, :] - 2 * q_E[2, :] * q_E[0, :]
        outputs['Rot_ECI_EF'][
            2, 1, :] = 2 * q_E[2, :] * q_E[3, :] + 2 * q_E[1, :] * q_E[0, :]
        outputs['Rot_ECI_EF'][2,
                              2, :] = 1 - 2 * q_E[1, :]**2 - 2 * q_E[2, :]**2

    def compute_partials(self, inputs, partials):
        num_times = self.options['num_times']
        q_E = inputs['q_E']

        dRot_dq = partials['Rot_ECI_EF', 'q_E'].reshape(9, num_times, 4)

        dRot_dq[0, :, 0] = 0
        dRot_dq[0, :, 2] = -4 * q_E[2, :]
        dRot_dq[0, :, 3] = -4 * q_E[3, :]
        dRot_dq[1, :, 0] = -2 * q_E[3, :]
        dRot_dq[1, :, 1] = 2 * q_E[2, :]
        dRot_dq[1, :, 2] = 2 * q_E[1, :]
        dRot_dq[1, :, 3] = -2 * q_E[0, :]
        dRot_dq[2, :, 0] = 2 * q_E[2, :]
        dRot_dq[2, :, 1] = 2 * q_E[3, :]
        dRot_dq[2, :, 2] = 2 * q_E[0, :]
        dRot_dq[2, :, 3] = 2 * q_E[1, :]
        dRot_dq[3, :, 0] = 2 * q_E[3, :]
        dRot_dq[3, :, 1] = 2 * q_E[2, :]
        dRot_dq[3, :, 2] = 2 * q_E[1, :]
        dRot_dq[3, :, 3] = 2 * q_E[0, :]
        dRot_dq[4, :, 0] = 0
        dRot_dq[4, :, 1] = -4 * q_E[1, :]
        dRot_dq[4, :, 2] = 0
        dRot_dq[4, :, 3] = -4 * q_E[3, :]
        dRot_dq[5, :, 0] = -2 * q_E[1, :]
        dRot_dq[5, :, 1] = -2 * q_E[0, :]
        dRot_dq[5, :, 2] = 2 * q_E[3, :]
        dRot_dq[5, :, 3] = 2 * q_E[2, :]
        dRot_dq[6, :, 0] = -2 * q_E[2, :]
        dRot_dq[6, :, 1] = 2 * q_E[3, :]
        dRot_dq[6, :, 2] = -2 * q_E[0, :]
        dRot_dq[6, :, 3] = 2 * q_E[1, :]
        dRot_dq[7, :, 0] = 2 * q_E[1, :]
        dRot_dq[7, :, 1] = 2 * q_E[0, :]
        dRot_dq[7, :, 2] = 2 * q_E[3, :]
        dRot_dq[7, :, 3] = 2 * q_E[2, :]
        dRot_dq[8, :, 0] = 0
        dRot_dq[8, :, 1] = -4 * q_E[1, :]
        dRot_dq[8, :, 2] = -4 * q_E[2, :]
        dRot_dq[8, :, 3] = 0


if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem, IndepVarComp, Group

    group = Group()
    comp = IndepVarComp()
    num_times = 3
    comp.add_output('q_E', val=np.ones((4, num_times)))

    group.add_subsystem('Inputcomp', comp, promotes=['*'])
    group.add_subsystem('Rot_ECI_EF',
                        EarthspinRotationMtx(num_times=num_times),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)

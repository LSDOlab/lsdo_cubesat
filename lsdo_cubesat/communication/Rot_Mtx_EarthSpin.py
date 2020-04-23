import numpy as numpy
from openmdao.api import ExplicitComponent


class EarthsSpinMtxComp(ExplicitComponent):
    """
    Quaternion to rotation matrix for the earth spin.
    """

    def initialize(self):
        self.options.declare('num_times',types=int)

    def setup(self):
        opts = self.options
        num_times = opts['num_times']
        
        # Inputs
        self.add_input('q_E', np.zeros((4, num_times)), units=None,
                       desc='Quarternion matrix in Earth-fixed frame over time')

        # Outputs
        self.add_output('O_IE', np.zeros((3, 3, num_times)), units=None,
                        desc='Rotation matrix from Earth-centered inertial frame to '
                             'Earth-fixed frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        num_times = self.options['num_times']
        
        q_E = inputs['q_E']
        O_IE = outputs['O_IE']

        A = np.zeros((4, 3))
        B = np.zeros((4, 3))

        for i in range(0, self.n):
            A[0, :] = ( q_E[0, i], -q_E[3, i],  q_E[2, i])  # noqa: E201
            A[1, :] = ( q_E[3, i],  q_E[0, i], -q_E[1, i])  # noqa: E201
            A[2, :] = (-q_E[2, i],  q_E[1, i],  q_E[0, i])  # noqa: E201
            A[3, :] = ( q_E[1, i],  q_E[2, i],  q_E[3, i])  # noqa: E201

            B[0, :] = ( q_E[0, i],  q_E[3, i], -q_E[2, i])  # noqa: E201
            B[1, :] = (-q_E[3, i],  q_E[0, i],  q_E[1, i])  # noqa: E201
            B[2, :] = ( q_E[2, i], -q_E[1, i],  q_E[0, i])  # noqa: E201
            B[3, :] = ( q_E[1, i],  q_E[2, i],  q_E[3, i])  # noqa: E201

            O_IE[:, :, i] = np.dot(A.T, B)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        num_times = self.options['num_times']
        q_E = inputs['q_E']

        A = np.zeros((4, 3))
        B = np.zeros((4, 3))

        dA_dq = np.zeros((4, 3, 4))
        dB_dq = np.zeros((4, 3, 4))

        self.J = np.zeros((self.n, 3, 3, 4))

        # dA_dq
        dA_dq[0, :, 0] = (1, 0, 0)
        dA_dq[1, :, 0] = (0, 1, 0)
        dA_dq[2, :, 0] = (0, 0, 1)
        dA_dq[3, :, 0] = (0, 0, 0)

        dA_dq[0, :, 1] = (0, 0, 0)
        dA_dq[1, :, 1] = (0, 0, -1)
        dA_dq[2, :, 1] = (0, 1, 0)
        dA_dq[3, :, 1] = (1, 0, 0)

        dA_dq[0, :, 2] = (0, 0, 1)
        dA_dq[1, :, 2] = (0, 0, 0)
        dA_dq[2, :, 2] = (-1, 0, 0)
        dA_dq[3, :, 2] = (0, 1, 0)

        dA_dq[0, :, 3] = (0, -1, 0)
        dA_dq[1, :, 3] = (1, 0, 0)
        dA_dq[2, :, 3] = (0, 0, 0)
        dA_dq[3, :, 3] = (0, 0, 1)

        # dB_dq
        dB_dq[0, :, 0] = (1, 0, 0)
        dB_dq[1, :, 0] = (0, 1, 0)
        dB_dq[2, :, 0] = (0, 0, 1)
        dB_dq[3, :, 0] = (0, 0, 0)

        dB_dq[0, :, 1] = (0, 0, 0)
        dB_dq[1, :, 1] = (0, 0, 1)
        dB_dq[2, :, 1] = (0, -1, 0)
        dB_dq[3, :, 1] = (1, 0, 0)

        dB_dq[0, :, 2] = (0, 0, -1)
        dB_dq[1, :, 2] = (0, 0, 0)
        dB_dq[2, :, 2] = (1, 0, 0)
        dB_dq[3, :, 2] = (0, 1, 0)

        dB_dq[0, :, 3] = (0, 1, 0)
        dB_dq[1, :, 3] = (-1, 0, 0)
        dB_dq[2, :, 3] = (0, 0, 0)
        dB_dq[3, :, 3] = (0, 0, 1)

        for i in range(0, self.n):
            A[0, :] = ( q_E[0, i], -q_E[3, i],  q_E[2, i])  # noqa: E201
            A[1, :] = ( q_E[3, i],  q_E[0, i], -q_E[1, i])  # noqa: E201
            A[2, :] = (-q_E[2, i],  q_E[1, i],  q_E[0, i])  # noqa: E201
            A[3, :] = ( q_E[1, i],  q_E[2, i],  q_E[3, i])  # noqa: E201

            B[0, :] = ( q_E[0, i],  q_E[3, i], -q_E[2, i])  # noqa: E201
            B[1, :] = (-q_E[3, i],  q_E[0, i],  q_E[1, i])  # noqa: E201
            B[2, :] = ( q_E[2, i], -q_E[1, i],  q_E[0, i])  # noqa: E201
            B[3, :] = ( q_E[1, i],  q_E[2, i],  q_E[3, i])  # noqa: E201

            for k in range(0, 4):
                self.J[i, :, :, k] = np.dot(dA_dq[:, :, k].T, B) + \
                    np.dot(A.T, dB_dq[:, :, k])


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

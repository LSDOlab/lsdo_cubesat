"""
Coordinate transformation from the interial plane to the rolled, pitched
(forward facing) plane.
"""
import numpy as np
from openmdao.api import ExplicitComponent


class RotMtxTIComp(ExplicitComponent):

    dvx_dv = np.zeros((3, 3, 3))
    dvx_dv[0, :, 0] = (0., 0., 0.)
    dvx_dv[1, :, 0] = (0., 0., -1.)
    dvx_dv[2, :, 0] = (0., 1., 0.)

    dvx_dv[0, :, 1] = (0., 0., 1.)
    dvx_dv[1, :, 1] = (0., 0., 0.)
    dvx_dv[2, :, 1] = (-1., 0., 0.)

    dvx_dv[0, :, 2] = (0., -1., 0.)
    dvx_dv[1, :, 2] = (1., 0., 0.)
    dvx_dv[2, :, 2] = (0., 0., 0.)

    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        n = self.options['num_times']
        self.dO_dr = np.zeros((n, 3, 3, 6))

        # Inputs
        self.add_input(
            'orbit_state_km',
            np.zeros((6, n)),
            desc='Position and velocity vector from earth to satellite in '
            'Earth-centered inertial frame over time')

        # Outputs
        self.add_output('rot_mtx_t_i_3x3xn',
                        np.zeros((3, 3, n)),
                        desc='Rotation matrix from rolled body-fixed frame to '
                        'Earth-centered inertial frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        n = self.options['num_times']

        r_e2b_I = inputs['orbit_state_km']
        rot_mtx_t_i_3x3xn = outputs['rot_mtx_t_i_3x3xn']

        rot_mtx_t_i_3x3xn[:] = np.zeros(rot_mtx_t_i_3x3xn.shape)
        for i in range(0, n):

            r = r_e2b_I[0:3, i]
            v = r_e2b_I[3:, i]

            normr = np.sqrt(np.dot(r, r))
            normv = np.sqrt(np.dot(v, v))

            # Prevent overflow
            if normr < 1e-10:
                normr = 1e-10
            if normv < 1e-10:
                normv = 1e-10

            r = r / normr
            v = v / normv

            vx = np.zeros((3, 3))
            vx[0, :] = (0., -v[2], v[1])
            vx[1, :] = (v[2], 0., -v[0])
            vx[2, :] = (-v[1], v[0], 0.)

            iB = np.dot(vx, r)
            jB = -np.dot(vx, iB)

            rot_mtx_t_i_3x3xn[0, :, i] = iB
            rot_mtx_t_i_3x3xn[1, :, i] = jB
            rot_mtx_t_i_3x3xn[2, :, i] = -v

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        n = self.options['num_times']

        r_e2b_I = inputs['orbit_state_km']

        diB_dv = np.zeros((3, 3))
        djB_dv = np.zeros((3, 3))

        for i in range(0, n):

            r = r_e2b_I[0:3, i]
            v = r_e2b_I[3:, i]

            normr = np.sqrt(np.dot(r, r))
            normv = np.sqrt(np.dot(v, v))

            # Prevent overflow
            if normr < 1e-10:
                normr = 1e-10
            if normv < 1e-10:
                normv = 1e-10

            r = r / normr
            v = v / normv

            dr_dr = np.zeros((3, 3))
            dv_dv = np.zeros((3, 3))

            for k in range(0, 3):
                dr_dr[k, k] += 1.0 / normr
                dv_dv[k, k] += 1.0 / normv
                dr_dr[:, k] -= r_e2b_I[0:3, i] * r_e2b_I[k, i] / normr**3
                dv_dv[:, k] -= r_e2b_I[3:, i] * r_e2b_I[3 + k, i] / normv**3

            vx = np.zeros((3, 3))
            vx[0, :] = (0., -v[2], v[1])
            vx[1, :] = (v[2], 0., -v[0])
            vx[2, :] = (-v[1], v[0], 0.)

            iB = np.dot(vx, r)

            diB_dr = vx
            diB_dv[:, 0] = np.dot(self.dvx_dv[:, :, 0], r)
            diB_dv[:, 1] = np.dot(self.dvx_dv[:, :, 1], r)
            diB_dv[:, 2] = np.dot(self.dvx_dv[:, :, 2], r)

            djB_diB = -vx
            djB_dv[:, 0] = -np.dot(self.dvx_dv[:, :, 0], iB)
            djB_dv[:, 1] = -np.dot(self.dvx_dv[:, :, 1], iB)
            djB_dv[:, 2] = -np.dot(self.dvx_dv[:, :, 2], iB)

            self.dO_dr[i, 0, :, 0:3] = np.dot(diB_dr, dr_dr)
            self.dO_dr[i, 0, :, 3:] = np.dot(diB_dv, dv_dv)

            self.dO_dr[i, 1, :, 0:3] = np.dot(np.dot(djB_diB, diB_dr), dr_dr)
            self.dO_dr[i, 1, :,
                       3:] = np.dot(np.dot(djB_diB, diB_dv) + djB_dv, dv_dv)

            self.dO_dr[i, 2, :, 3:] = -dv_dv

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        drot_mtx_t_i_3x3xn = d_outputs['rot_mtx_t_i_3x3xn']

        if mode == 'fwd':
            if 'orbit_state_km' in d_inputs:
                for k in range(3):
                    for j in range(3):
                        for i in range(6):
                            drot_mtx_t_i_3x3xn[k, j, :] += self.dO_dr[:, k, j, i] * \
                                d_inputs['orbit_state_km'][i, :]
        else:
            if 'orbit_state_km' in d_inputs:
                for k in range(3):
                    for j in range(3):
                        for i in range(6):
                            d_inputs['orbit_state_km'][i, :] += self.dO_dr[:, k, j, i] * \
                                drot_mtx_t_i_3x3xn[k, j, :]


if __name__ == '__main__':

    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp
    import numpy as np

    group = Group()

    comp = IndepVarComp()
    n = 2

    r_e2b_I = np.random.random((6, n))
    comp.add_output('orbit_state_km', val=r_e2b_I)
    group.add('Inputcomp', comp, promotes=['*'])
    group.add('att', RotMtxTIComp(num_times=n), promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)

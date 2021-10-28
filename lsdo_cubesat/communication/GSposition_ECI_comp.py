import numpy as np

from openmdao.api import ExplicitComponent
from lsdo_cubesat.utils.utils import get_array_indices


class GS_ECI_Comp(ExplicitComponent):
    """
    Convert time history of ground station position from ECEF frame to ECI frame
    """
    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']

        self.add_input('Rot_ECI_EF',
                       np.zeros((3, 3, num_times)),
                       units=None,
                       desc='Rotation matrix from Earth-centered inertial '
                       'frame to Earth-fixed frame over time')

        self.add_input('r_e2g_E',
                       np.zeros((3, num_times)),
                       units='km',
                       desc='Position vector from earth to ground station in '
                       'Earth-fixed frame over time')

        self.add_output('r_e2g_I',
                        np.zeros((3, num_times)),
                        units='km',
                        desc='Position vector from earth to ground station in '
                        'Earth-centered inertial frame over time')

        ones_3 = np.ones(3, int)
        mtx_indices = get_array_indices(*(3, 3, num_times))
        vec_indices = get_array_indices(*(3, num_times))

        rows = np.einsum('in,j->ijn', vec_indices, ones_3).flatten()
        cols = mtx_indices.flatten()
        self.declare_partials('r_e2g_I', 'Rot_ECI_EF', rows=rows, cols=cols)

        rows = np.einsum('in,j->ijn', vec_indices, ones_3).flatten()
        cols = np.einsum('jn,i->ijn', vec_indices, ones_3).flatten()
        self.declare_partials('r_e2g_I', 'r_e2g_E', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        num_times = self.options['num_times']
        Rot_ECI_EF = inputs['Rot_ECI_EF']

        outputs['r_e2g_I'] = np.einsum('ijn,jn->in', inputs['Rot_ECI_EF'],
                                       inputs['r_e2g_E'])

        # np.savetxt("rundata/Rot_ECI_EF.csv", Rot_ECI_EF, header="Rot_ECI_EF")

    def compute_partials(self, inputs, partials):
        num_times = self.options['num_times']

        partials['r_e2g_I', 'Rot_ECI_EF'] = np.einsum('jn,i->ijn',
                                                      inputs['r_e2g_E'],
                                                      np.ones(3)).flatten()
        partials['r_e2g_I', 'r_e2g_E'] = inputs['Rot_ECI_EF'].flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    num_times = 3

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('r_e2g_E', val=10 * np.random.random((3, num_times)))
    comp.add_output('Rot_ECI_EF', val=10 * np.random.random((3, 3, num_times)))
    prob.model.add('inputs_comp', comp, promotes=['*'])

    comp = GS_ECI_Comp(num_times=num_times, )
    prob.model.add('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)

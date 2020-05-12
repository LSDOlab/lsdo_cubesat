import numpy as np

from openmdao.api import ExplicitComponent
from lsdo_cubesat.utils.utils import get_array_indices

class VectorBodyComp(ExplicitComponent):
    """
    Transform from body to inertial frame.
    """

    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']

        self.add_input('r_b2g_I', np.zeros((3, num_times)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in Earth-centered inertial frame over time')

        self.add_input('rot_mtx_i_b_3x3xn', np.zeros((3, 3, num_times)), units=None,
                       desc='Rotation matrix from body-fixed frame to Earth-centered'
                            'inertial frame over time')
                            
        self.add_output('r_b2g_B', np.zeros((3, num_times)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in body-fixed frame over time')

        ones_3 = np.ones(3, int)
        mtx_indices = get_array_indices(*(3, 3, num_times))
        vec_indices = get_array_indices(*(3, num_times))

        rows = np.einsum('in,j->ijn', vec_indices, ones_3).flatten()
        cols = mtx_indices.flatten()
        self.declare_partials('r_b2g_B', 'rot_mtx_i_b_3x3xn', rows=rows, cols=cols)

        rows = np.einsum('in,j->ijn', vec_indices, ones_3).flatten()
        cols = np.einsum('jn,i->ijn', vec_indices, ones_3).flatten()
        self.declare_partials('r_b2g_B', 'r_b2g_I', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        num_times = self.options['num_times']

        outputs['r_b2g_B'] = np.einsum('ijn,jn->in', inputs['rot_mtx_i_b_3x3xn'], inputs['r_b2g_I'])

    def compute_partials(self, inputs, partials):
        num_times = self.options['num_times']

        partials['r_b2g_B', 'rot_mtx_i_b_3x3xn'] = np.einsum('jn,i->ijn', inputs['r_b2g_I'], np.ones(3)).flatten()
        partials['r_b2g_B', 'r_b2g_I'] = inputs['rot_mtx_i_b_3x3xn'].flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    num_times = 4

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('r_b2g_I', val=10 * np.random.random((3, num_times)))
    comp.add_output('rot_mtx_i_b_3x3xn', val=10 * np.random.random((3, 3, num_times)))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = VectorBodyComp(
        num_times=num_times,
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)
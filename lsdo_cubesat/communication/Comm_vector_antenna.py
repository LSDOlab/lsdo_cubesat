import numpy as np

from openmdao.api import ExplicitComponent
from lsdo_cubesat.utils.utils import get_array_indices


class AntennaBodyComp(ExplicitComponent):
    """
    Transform from antenna to body frame
    """
    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']

        self.add_input('r_b2g_B',
                       np.zeros((3, num_times)),
                       units='km',
                       desc='Position vector from satellite to ground station '
                       'in body-fixed frame over time')

        self.add_input('Rot_AB',
                       np.zeros((3, 3, num_times)),
                       units=None,
                       desc='Rotation matrix from antenna angle to body-fixed '
                       'frame over time')

        self.add_output(
            'r_b2g_A',
            np.zeros((3, num_times)),
            units='km',
            desc='Position vector from satellite to ground station '
            'in antenna angle frame over time')

        ones_3 = np.ones(3, int)
        mtx_indices = get_array_indices(*(3, 3, num_times))
        vec_indices = get_array_indices(*(3, num_times))

        rows = np.einsum('in,j->ijn', vec_indices, ones_3).flatten()
        cols = mtx_indices.flatten()
        self.declare_partials('r_b2g_A', 'Rot_AB', rows=rows, cols=cols)

        rows = np.einsum('in,j->ijn', vec_indices, ones_3).flatten()
        cols = np.einsum('jn,i->ijn', vec_indices, ones_3).flatten()
        self.declare_partials('r_b2g_A', 'r_b2g_B', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        num_times = self.options['num_times']

        outputs['r_b2g_A'] = np.einsum('ijn,jn->in', inputs['Rot_AB'],
                                       inputs['r_b2g_B'])
        # print(outputs['r_b2g_A'].shape)

    def compute_partials(self, inputs, partials):
        num_times = self.options['num_times']

        # print(partials['r_b2g_A', 'Rot_AB'].shape)
        partials['r_b2g_A',
                 'Rot_AB'] = np.einsum('jn,i->ijn', inputs['r_b2g_B'],
                                       np.ones(3)).flatten()
        # print(partials['r_b2g_A', 'Rot_AB'].shape)
        partials['r_b2g_A', 'r_b2g_B'] = inputs['Rot_AB'].flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    num_times = 3

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('r_b2g_B', val=10 * np.random.random((3, num_times)))
    comp.add_output('Rot_AB', val=10 * np.random.random((3, 3, num_times)))
    prob.model.add('inputs_comp', comp, promotes=['*'])

    comp = AntennaBodyComp(num_times=num_times, )
    prob.model.add('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)

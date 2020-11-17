import numpy as np

from openmdao.api import ExplicitComponent

from lsdo_cubesat.utils.miscellaneous_functions.get_array_indices import get_array_indices
from lsdo_cubesat.utils.miscellaneous_functions.decompose_shape_tuple import decompose_shape_tuple


class ArrayExpansionComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('expand_indices', types=list)
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)

    def setup(self):
        shape = self.options['shape']
        expand_indices = self.options['expand_indices']
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        (
            in_string,
            ones_string,
            out_string,
            in_shape,
            ones_shape,
            out_shape,
        ) = decompose_shape_tuple(shape, expand_indices)

        einsum_string = '{},{}->{}'.format(in_string, ones_string, out_string)

        self.add_input(in_name, shape=in_shape)
        self.add_output(out_name, shape=out_shape)

        in_indices = get_array_indices(*in_shape)
        out_indices = get_array_indices(*out_shape)

        self.einsum_string = einsum_string
        self.ones_shape = ones_shape

        rows = out_indices.flatten()
        cols = np.einsum(einsum_string, in_indices, np.ones(ones_shape,
                                                            int)).flatten()
        self.declare_partials(out_name, in_name, val=1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        outputs[out_name] = np.einsum(self.einsum_string, inputs[in_name],
                                      np.ones(self.ones_shape))


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    shape = (3, 2, 4)
    expand_indices = [0, 1]

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('in_name', np.random.rand(4))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = ArrayExpansionComp(
        shape=shape,
        expand_indices=expand_indices,
        out_name='out_name',
        in_name='in_name',
    )
    prob.model.add_subsystem('array_expansion_comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

    print(prob['in_name'])
    print(prob['out_name'])

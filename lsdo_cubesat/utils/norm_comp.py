import numpy as np

from openmdao.api import ExplicitComponent


class NormComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('axis', types=int)
        self.options.declare('order', default=2, types=int)
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)

    def setup(self):
        shape = self.options['shape']
        axis = self.options['axis']
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        out_shape = shape[:axis] + shape[axis + 1:]

        self.add_input(in_name, shape=shape)
        self.add_output(out_name, shape=out_shape)
        self.declare_partials(out_name, in_name, method='cs')

    def compute(self, inputs, outputs):
        axis = self.options['axis']
        order = self.options['order']
        array = inputs[self.options['in_name']]

        outputs[self.options['out_name']] = np.linalg.norm(
            array,
            axis=axis,
            ord=order,
        )

    def compute_partials(self, inputs, partials):
        pass

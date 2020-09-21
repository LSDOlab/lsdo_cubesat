import numpy as np
from openmdao.api import ExplicitComponent
from math import ceil
from lsdo_utils.miscellaneous_functions.get_array_indices import get_array_indices
from lsdo_utils.miscellaneous_functions.decompose_shape_tuple import decompose_shape_tuple


class SliceComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('slice_axis', default=0, types=int)
        self.options.declare('step', types=int)
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)

    def setup(self):
        shape = self.options['shape']
        step = self.options['step']
        slice_axis = self.options['slice_axis']
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        self.add_input(in_name, shape=shape)

        out_shape = list(shape)
        out_shape[slice_axis] = ceil(shape[slice_axis] / step)
        out_shape = tuple(out_shape)
        self.add_output(out_name, shape=out_shape)

        self.slices = []
        axis = 0
        for dim in shape:
            if axis == slice_axis:
                self.slices.append(slice(0, dim, step))
            else:
                self.slices.append(slice(0, dim, None))
            axis += 1
        self.slices = tuple(self.slices)

        cols = np.arange(np.prod(shape)).reshape(shape)
        cols = cols[self.slices].flatten()

        rows = np.arange(np.prod(out_shape))

        self.declare_partials(
            out_name,
            in_name,
            rows=rows,
            cols=cols,
            val=1,
        )

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        outputs[out_name] = inputs[in_name][self.slices]


if __name__ == '__main__':

    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp

    np.random.seed(0)

    num_times = 100
    step = 1
    shape = (3, 3, num_times)
    slice_axis = 2

    comp = IndepVarComp()
    comp.add_output(
        'rng',
        val=np.random.rand(np.prod(shape)).reshape(shape),
        shape=shape,
    )
    prob = Problem()
    prob.model.add_subsystem('inputs', comp, promotes=['*'])
    prob.model.add_subsystem(
        'slice',
        SliceComp(
            shape=shape,
            step=step,
            slice_axis=slice_axis,
            in_name='rng',
            out_name='out',
        ),
        promotes=['*'],
    )
    prob.setup(check=True, force_alloc_complex=True)
    prob.check_partials(compact_print=True)

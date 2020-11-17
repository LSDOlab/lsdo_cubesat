from lsdo_cubesat.utils.api import LinearCombinationComp, LinearPowerCombinationComp, PowerCombinationComp, ArrayExpansionComp, CrossProductComp, ArrayContractionComp
from openmdao.api import Group


class NormGroup(Group):
    def initialize(self):
        self.options.declare('shape')
        self.options.declare('in_name')
        self.options.declare('out_name')
        self.options.declare('axis')

    def setup(self):
        shape = self.options['shape']
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        axis = self.options['axis']

        self.add_subsystem(
            'compute_square',
            PowerCombinationComp(
                shape=shape,
                out_name='{}_squared'.format(in_name),
                powers_dict={
                    in_name: 2.,
                },
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'compute_sum',
            ArrayContractionComp(
                shape=shape,
                contract_indices=[axis],
                in_name='{}_squared'.format(in_name),
                out_name='sum_{}_squared'.format(in_name),
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'compute_square_root',
            PowerCombinationComp(
                shape=shape[:axis] + shape[axis + 1:],
                out_name=out_name,
                powers_dict={
                    'sum_{}_squared'.format(in_name): 0.5,
                },
            ),
            promotes=['*'],
        )


if __name__ == '__main__':

    from openmdao.api import Problem, IndepVarComp
    import numpy as np

    np.random.seed(0)
    shape = (3, 4, 5)

    comp = IndepVarComp()
    comp.add_output('vec',
                    val=np.random.rand(np.prod(shape)).reshape(shape),
                    shape=shape)

    prob = Problem()
    prob.model.add_subsystem(
        'indeps',
        comp,
        promotes=['*'],
    )
    prob.model.add_subsystem(
        'norm',
        NormGroup(
            shape=shape,
            in_name='vec',
            out_name='vec_norm',
            axis=2,
        ),
        promotes=['*'],
    )
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)

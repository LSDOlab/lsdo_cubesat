from lsdo_utils.api import LinearCombinationComp, LinearPowerCombinationComp, PowerCombinationComp, ArrayExpansionComp, CrossProductComp, ArrayContractionComp
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
                contract_indices=[0],
                in_name='{}_squared'.format(in_name),
                out_name='sum_{}_squared'.format(in_name),
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'compute_square_root',
            PowerCombinationComp(
                shape=(shape[1], ),
                out_name=out_name,
                powers_dict={
                    'sum_{}_squared'.format(in_name): 0.5,
                },
            ),
            promotes=['*'],
        )

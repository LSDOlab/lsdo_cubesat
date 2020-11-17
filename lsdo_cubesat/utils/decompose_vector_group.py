from openmdao.api import Group

from lsdo_cubesat.utils.api import ArrayExpansionComp, ArrayContractionComp, PowerCombinationComp


class DecomposeVectorGroup(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('vec_name', types=str)
        self.options.declare('norm_name', types=str)
        self.options.declare('unit_vec_name', types=str)

    def setup(self):
        num_times = self.options['num_times']
        vec_name = self.options['vec_name']
        norm_name = self.options['norm_name']
        unit_vec_name = self.options['unit_vec_name']

        comp = PowerCombinationComp(
            shape=(3, num_times),
            out_name='tmp_{}_2'.format(vec_name),
            powers_dict={vec_name: 2.},
        )
        self.add_subsystem('tmp_{}_2_comp'.format(vec_name),
                           comp,
                           promotes=['*'])

        comp = ArrayContractionComp(
            shape=(3, num_times),
            contract_indices=[0],
            out_name='tmp_{}_2'.format(norm_name),
            in_name='tmp_{}_2'.format(vec_name),
        )
        self.add_subsystem('tmp_{}_2_comp'.format(norm_name),
                           comp,
                           promotes=['*'])

        comp = PowerCombinationComp(
            shape=(num_times, ),
            out_name=norm_name,
            powers_dict={'tmp_{}_2'.format(norm_name): 0.5},
        )
        self.add_subsystem('{}_comp'.format(norm_name), comp, promotes=['*'])

        comp = ArrayExpansionComp(
            shape=(3, num_times),
            expand_indices=[0],
            out_name='tmp_{}_expanded'.format(norm_name),
            in_name=norm_name,
        )
        self.add_subsystem('tmp_{}_expanded_comp'.format(norm_name),
                           comp,
                           promotes=['*'])

        comp = PowerCombinationComp(shape=(3, num_times),
                                    out_name=unit_vec_name,
                                    powers_dict={
                                        vec_name: 1.,
                                        'tmp_{}_expanded'.format(norm_name):
                                        -1.,
                                    })
        self.add_subsystem('{}_comp'.format(unit_vec_name),
                           comp,
                           promotes=['*'])

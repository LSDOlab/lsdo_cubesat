import numpy as np

from openmdao.api import Group, IndepVarComp

from lsdo_cubesat.utils.api import CrossProductComp, LinearCombinationComp, PowerCombinationComp, ScalarContractionComp

from lsdo_cubesat.utils.decompose_vector_group import DecomposeVectorGroup
from lsdo_cubesat.utils.projection_group import ProjectionGroup
from lsdo_cubesat.utils.ks_comp import KSComp
from lsdo_cubesat.orbit.constant_orbit_group import ConstantOrbitGroup
from lsdo_cubesat.alignment.sun_direction_comp import SunDirectionComp
from lsdo_cubesat.alignment.mask_vec_comp import MaskVecComp
from lsdo_cubesat.utils.dot_product_comp import DotProductComp


class AlignmentGroup(Group):
    def initialize(self):
        self.options.declare('swarm')
        self.options.declare('mtx')

    def setup(self):
        swarm = self.options['swarm']
        mtx = self.options['mtx']

        num_times = swarm['num_times']
        num_cp = swarm['num_cp']
        step_size = swarm['step_size']

        shape = (3, num_times)

        times = np.linspace(0., step_size * (num_times - 1), num_times)

        comp = IndepVarComp()
        comp.add_output('times', val=times)
        self.add_subsystem('inputs_comp', comp, promotes=['*'])

        comp = SunDirectionComp(
            num_times=num_times,
            launch_date=swarm['launch_date'],
        )
        self.add_subsystem('sun_direction_comp', comp, promotes=['*'])

        # group = ConstantOrbitGroup(
        #     num_times=num_times,
        #     num_cp=num_cp,
        #     step_size=step_size,
        #     cubesat=swarm.children[0],
        # )
        # self.add_subsystem('constant_orbit_group', group, promotes=['*'])
        comp = CrossProductComp(
            shape_no_3=(num_times, ),
            out_index=0,
            in1_index=0,
            in2_index=0,
            out_name='normal_cross_vec',
            in1_name='velocity_unit_vec',
            in2_name='position_unit_vec',
        )
        self.add_subsystem('normal_cross_vec_comp', comp, promotes=['*'])

        comp = CrossProductComp(
            shape_no_3=(num_times, ),
            out_index=0,
            in1_index=0,
            in2_index=0,
            out_name='observation_cross_vec',
            in1_name='position_unit_vec',
            in2_name='sun_unit_vec',
        )
        self.add_subsystem('observation_cross_vec_comp', comp, promotes=['*'])

        group = DecomposeVectorGroup(
            num_times=num_times,
            vec_name='normal_cross_vec',
            norm_name='normal_cross_norm',
            unit_vec_name='normal_cross_unit_vec',
        )
        self.add_subsystem('normal_cross_decomposition_group',
                           group,
                           promotes=['*'])

        group = DecomposeVectorGroup(
            num_times=num_times,
            vec_name='observation_cross_vec',
            norm_name='observation_cross_norm',
            unit_vec_name='observation_cross_unit_vec',
        )
        self.add_subsystem('observation_cross_decomposition_group',
                           group,
                           promotes=['*'])

        comp = DotProductComp(vec_size=3,
                              length=num_times,
                              a_name='observation_cross_unit_vec',
                              b_name='normal_cross_unit_vec',
                              c_name='observation_dot',
                              a_units=None,
                              b_units=None,
                              c_units=None)
        self.add_subsystem('observation_dot_comp', comp, promotes=['*'])

        comp = MaskVecComp(
            num_times=num_times,
            swarm=swarm,
        )
        self.add_subsystem('mask_vec_comp', comp, promotes=['*'])

        # Separation

        separation_constraint_names = [
            # ('sunshade', 'optics'),
            ('optics', 'detector'),
        ]

        for name1, name2 in [
                # ('sunshade', 'optics'),
                # ('sunshade', 'detector'),
            ('optics', 'detector'),
        ]:
            position_name = 'position_{}_{}_km'.format(name1, name2)
            distance_name = 'distance_{}_{}_km'.format(name1, name2)
            unit_vec_name = 'unit_vec_{}_{}_km'.format(name1, name2)

            comp = LinearCombinationComp(
                shape=(3, num_times),
                out_name=position_name,
                coeffs_dict={
                    '{}_cubesat_group_position_km'.format(name1): 1.,
                    '{}_cubesat_group_position_km'.format(name2): -1.,
                },
            )
            self.add_subsystem('{}_comp'.format(position_name),
                               comp,
                               promotes=['*'])

            group = DecomposeVectorGroup(
                num_times=num_times,
                vec_name=position_name,
                norm_name=distance_name,
                unit_vec_name=unit_vec_name,
            )
            self.add_subsystem('{}_{}_decomposition_group'.format(
                name1, name2),
                               group,
                               promotes=['*'])

        # Transverse displacement

        transverse_constraint_names = [
            # ('sunshade', 'detector'),
            ('optics', 'detector'),
        ]

        for name1, name2 in transverse_constraint_names:
            position_name = 'position_{}_{}_km'.format(name1, name2)
            projected_position_name = 'projected_position_{}_{}_km'.format(
                name1, name2)
            normal_position_name = 'normal_position_{}_{}_km'.format(
                name1, name2)
            normal_distance_name = 'normal_distance_{}_{}_km'.format(
                name1, name2)
            normal_unit_vec_name = 'normal_unit_vec_{}_{}_km'.format(
                name1, name2)

            group = ProjectionGroup(
                num_times=num_times,
                in1_name='sun_unit_vec',
                in2_name=position_name,
                out_name=projected_position_name,
            )
            self.add_subsystem('{}_group'.format(projected_position_name),
                               group,
                               promotes=['*'])

            comp = LinearCombinationComp(
                shape=(3, num_times),
                out_name=normal_position_name,
                coeffs_dict={
                    position_name: 1.,
                    projected_position_name: -1.,
                },
            )
            self.add_subsystem('{}_comp'.format(normal_position_name),
                               comp,
                               promotes=['*'])

            group = DecomposeVectorGroup(
                num_times=num_times,
                vec_name=normal_position_name,
                norm_name=normal_distance_name,
                unit_vec_name=normal_unit_vec_name,
            )
            self.add_subsystem(
                '{}_decomposition_group'.format(normal_position_name),
                group,
                promotes=['*'])

        for constraint_name in [
                'normal_distance_{}_{}'.format(name1, name2)
                for name1, name2 in transverse_constraint_names
        ] + [
                'distance_{}_{}'.format(name1, name2)
                for name1, name2 in separation_constraint_names
        ]:
            comp = PowerCombinationComp(
                shape=(num_times, ),
                out_name='{}_mm'.format(constraint_name),
                coeff=1.e6,
                powers_dict={
                    '{}_km'.format(constraint_name): 1.,
                })
            self.add_subsystem('{}_mm_comp'.format(constraint_name),
                               comp,
                               promotes=['*'])

            comp = PowerCombinationComp(
                shape=(num_times, ),
                out_name='masked_{}_mm'.format(constraint_name),
                powers_dict={
                    'mask_vec': 1.,
                    '{}_mm'.format(constraint_name): 1.,
                })
            self.add_subsystem('masked_{}_mm_comp'.format(constraint_name),
                               comp,
                               promotes=['*'])

            comp = KSComp(
                in_name='masked_{}_mm'.format(constraint_name),
                out_name='ks_masked_{}_mm'.format(constraint_name),
                shape=(1, ),
                constraint_size=num_times,
                rho=100.,
            )
            self.add_subsystem('ks_masked_{}_mm_comp'.format(constraint_name),
                               comp,
                               promotes=['*'])

            comp = PowerCombinationComp(
                shape=(num_times, ),
                out_name='masked_{}_mm_sq'.format(constraint_name),
                powers_dict={
                    'masked_{}_mm'.format(constraint_name): 2.,
                })
            self.add_subsystem('masked_{}_mm_sq_comp'.format(constraint_name),
                               comp,
                               promotes=['*'])

            comp = ScalarContractionComp(
                shape=(num_times, ),
                out_name='masked_{}_mm_sq_sum'.format(constraint_name),
                in_name='masked_{}_mm_sq'.format(constraint_name),
            )
            self.add_subsystem(
                'masked_{}_mm_sq_sum_comp'.format(constraint_name),
                comp,
                promotes=['*'])

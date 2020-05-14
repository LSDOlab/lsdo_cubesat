import numpy as np

from openmdao.api import Group
from lsdo_utils.api import LinearCombinationComp, LinearPowerCombinationComp, PowerCombinationComp, ArrayExpansionComp, CrossProductComp, ArrayContractionComp
from lsdo_cubesat.utils.norm_comp import NormComp


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
                shape=(num_times, ),
                out_name=out_name,
                powers_dict={
                    'sum_{}_squared'.format(in_name): 0.5,
                },
            ),
            promotes=['*'],
        )


class MeanMotionGroup(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']
        self.add_subsystem(
            'compute_sp_ang_momentum',
            CrossProductComp(
                shape_no_3=(num_times, ),
                in1_index=0,
                in2_index=0,
                out_index=0,
                in1_name='position_km',
                in2_name='velocity_km_s',
                out_name='sp_ang_momentum_vec',
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'compute_sp_ang_momentum_mag',
            NormGroup(
                shape=(
                    3,
                    num_times,
                ),
                in_name='sp_ang_momentum_vec',
                out_name='sp_ang_momentum_mag',
                axis=0,
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'compute_semi_latus_rectum',
            PowerCombinationComp(
                shape=(num_times, ),
                out_name='semi_latus_rectum',
                powers_dict=dict(
                    sp_ang_momentum_mag=2.,
                    mu=-1.,
                ),
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'compute_position_mag',
            NormGroup(
                shape=(
                    3,
                    num_times,
                ),
                in_name='position_km',
                out_name='position_mag',
                axis=0,
            ),
            promotes=['*'],
        )

        self.add_subsystem('expand_position_mag',
                           ArrayExpansionComp(
                               shape=(
                                   3,
                                   num_times,
                               ),
                               expand_indices=[0],
                               in_name='position_mag',
                               out_name='position_mag_3xn',
                           ),
                           promotes=['*'])

        self.add_subsystem(
            'normalize_position',
            PowerCombinationComp(
                shape=(3, num_times),
                out_name='position_unit_vector',
                powers_dict=dict(
                    position_km=1.,
                    position_mag_3xn=-1.,
                ),
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'compute_v_cross_h',
            CrossProductComp(
                shape_no_3=(num_times, ),
                in1_index=0,
                in2_index=0,
                out_index=0,
                in1_name='velocity_km_s',
                in2_name='sp_ang_momentum_vec',
                out_name='vel_cross_sp_ang_momentum',
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'expand_mu',
            ArrayExpansionComp(
                shape=(
                    3,
                    num_times,
                ),
                expand_indices=[0],
                in_name='mu',
                out_name='mu_3xn',
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'compute_v_cross_h__mu',
            PowerCombinationComp(
                shape=(3, num_times),
                out_name='v_cross_h__mu',
                powers_dict=dict(
                    vel_cross_sp_ang_momentum=1.,
                    mu_3xn=-1.,
                ),
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'compute_eccentricity_vector',
            LinearCombinationComp(
                shape=(3, num_times),
                in_names=['v_cross_h__mu', 'position_unit_vector'],
                out_name='eccentricity_vec',
                coeffs=[1., -1.],
                constant=1,
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'compute_eccentricity',
            NormGroup(
                shape=(
                    3,
                    num_times,
                ),
                in_name='eccentricity_vec',
                out_name='eccentricity',
                axis=0,
            ),
            promotes=['*'],
        )

        # Compute (1-e**2)
        self.add_subsystem(
            'compute_semimajor_axis_denominator',
            LinearPowerCombinationComp(
                shape=(num_times, ),
                out_name='semimajor_axis_denominator',
                terms_list=[
                    (
                        -1.0,
                        dict(eccentricity=2., ),
                    ),
                ],
                constant=1,
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'compute_semimajor_axis',
            PowerCombinationComp(
                shape=(num_times, ),
                out_name='semimajor_axis',
                powers_dict=dict(
                    semi_latus_rectum=1.,
                    semimajor_axis_denominator=-1.,
                ),
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'compute_mean_motion_squared',
            PowerCombinationComp(
                shape=(num_times, ),
                out_name='mean_motion_squared',
                powers_dict=dict(
                    mu=1.,
                    semimajor_axis=-3.,
                ),
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'compute_osculating_mean_motion',
            PowerCombinationComp(
                shape=(num_times, ),
                out_name='osculating_mean_motion',
                powers_dict=dict(mean_motion_squared=0.5, ),
            ),
            promotes=['*'],
        )


if __name__ == '__main__':

    from openmdao.api import Problem, IndepVarComp

    np.random.seed(0)

    num_times = 10
    comp = IndepVarComp()
    comp.add_output('mu', val=398600.44 * np.ones(num_times))
    comp.add_output('position_km', val=np.abs(np.random.rand(3, num_times)))
    comp.add_output('velocity_km_s', val=np.random.rand(3, num_times))
    # comp.add_output('eccentricity', val=np.ones(num_times))

    prob = Problem()
    prob.model.add_subsystem(
        'indeps',
        comp,
        promotes=['*'],
    )
    prob.model.add_subsystem(
        'mean_motion_group',
        MeanMotionGroup(num_times=num_times),
        promotes=['*'],
    )
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)

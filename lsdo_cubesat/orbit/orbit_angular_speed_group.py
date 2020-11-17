import numpy as np

from openmdao.api import Group, IndepVarComp
from lsdo_cubesat.utils.api import LinearCombinationComp, LinearPowerCombinationComp, PowerCombinationComp, ArrayExpansionComp, CrossProductComp, ArrayContractionComp
from lsdo_cubesat.utils.norm import NormGroup
from lsdo_cubesat.utils.random_arrays import make_random_signed_array, make_random_bounded_array


class OrbitAngularSpeedGroup(Group):
    """
    Compute orbit angular speed from position and velocity vectors. For
    position and velocity vectors that change over time, compute
    osculating orbit angular velocity. Osculating orbit angular velocity
    can be used to integrate attitude dynamics.

    Options
    -------
    num_times : int
        Number of time steps over which orbit is integrated

    Parameters
    ----------
    position_km : shape=(3, num_times)
    velocity_km_s : shape=(3, num_times)

    Returns
    -------
    osculating_orbit_angular_speed : shape=(3, num_times)
    """
    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']

        comp = IndepVarComp()
        comp.add_output('mu', val=398600.4418 * np.ones(num_times))
        self.add_subsystem(
            'indeps',
            comp,
            promotes=['*'],
        )

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

        self.add_subsystem(
            'expand_position_mag',
            ArrayExpansionComp(
                shape=(
                    3,
                    num_times,
                ),
                expand_indices=[0],
                in_name='position_mag',
                out_name='position_mag_3xn',
            ),
            promotes=['*'],
        )

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
                    (-1.0, dict(eccentricity=2., )),
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
            'compute_orbit_angular_speed_sq',
            PowerCombinationComp(
                shape=(num_times, ),
                out_name='orbit_angular_speed_sq',
                powers_dict=dict(
                    mu=1.,
                    semimajor_axis=-3.,
                ),
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'compute_osculating_orbit_angular_speed',
            PowerCombinationComp(
                shape=(num_times, ),
                out_name='osculating_orbit_angular_speed',
                powers_dict=dict(orbit_angular_speed_sq=0.5, ),
            ),
            promotes=['*'],
        )

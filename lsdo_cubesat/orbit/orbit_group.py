import numpy as np

from openmdao.api import Group, IndepVarComp, NonlinearBlockGS, LinearBlockGS

from lsdo_cubesat.utils.api import ArrayReorderComp, LinearCombinationComp, PowerCombinationComp, ScalarContractionComp

from lsdo_cubesat.utils.decompose_vector_group import DecomposeVectorGroup
from lsdo_cubesat.utils.mtx_vec_comp import MtxVecComp
from lsdo_cubesat.utils.ks_comp import KSComp
from lsdo_cubesat.orbit.initial_orbit_comp import InitialOrbitComp
from lsdo_cubesat.orbit.relative_orbit_rk4_comp import RelativeOrbitRK4Comp
from lsdo_cubesat.orbit.orbit_state_decomposition_comp import OrbitStateDecompositionComp
from lsdo_cubesat.orbit.rot_mtx_t_i_comp import RotMtxTIComp
from lsdo_cubesat.utils.api import ArrayExpansionComp


class OrbitGroup(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('cubesat')
        self.options.declare('mtx')

    def setup(self):
        num_times = self.options['num_times']
        num_cp = self.options['num_cp']
        step_size = self.options['step_size']
        cubesat = self.options['cubesat']
        mtx = self.options['mtx']

        shape = (3, num_times)

        drag_unit_vec = np.outer(
            np.array([0., 0., 1.]),
            np.ones(num_times),
        )

        comp = IndepVarComp()
        comp.add_output('drag_unit_vec_t_3xn', val=drag_unit_vec)
        comp.add_output('dry_mass', val=cubesat['dry_mass'], shape=num_times)
        comp.add_output('radius_earth_km',
                        val=cubesat['radius_earth_km'],
                        shape=num_times)
        for var_name in ['initial_orbit_state']:
            comp.add_output(var_name, val=cubesat[var_name])
        self.add_subsystem('input_comp', comp, promotes=['*'])

        comp = LinearCombinationComp(
            shape=(num_times, ),
            out_name='mass',
            coeffs_dict=dict(
                dry_mass=1.,
                propellant_mass=1.,
                battery_mass_exp=1.,
            ),
        )
        self.add_subsystem('mass_comp', comp, promotes=['*'])

        if 1:

            coupled_group = Group()

            comp = LinearCombinationComp(
                shape=shape,
                out_name='force_3xn',
                coeffs_dict=dict(thrust_3xn=1., drag_3xn=1.),
            )
            coupled_group.add_subsystem('force_3xn_comp', comp, promotes=['*'])

            comp = RelativeOrbitRK4Comp(
                num_times=num_times,
                step_size=step_size,
            )
            coupled_group.add_subsystem('relative_orbit_rk4_comp',
                                        comp,
                                        promotes=['*'])

            comp = LinearCombinationComp(
                shape=(6, num_times),
                out_name='orbit_state',
                coeffs_dict=dict(
                    relative_orbit_state=1.,
                    reference_orbit_state=1.,
                ),
            )
            coupled_group.add_subsystem('orbit_state_comp',
                                        comp,
                                        promotes=['*'])

            comp = LinearCombinationComp(
                shape=(6, num_times),
                out_name='orbit_state_km',
                coeffs_dict=dict(orbit_state=1.e-3),
            )
            coupled_group.add_subsystem('orbit_state_km_comp',
                                        comp,
                                        promotes=['*'])

            comp = RotMtxTIComp(num_times=num_times)
            coupled_group.add_subsystem('rot_mtx_t_i_3x3xn_comp',
                                        comp,
                                        promotes=['*'])

            comp = ArrayReorderComp(
                in_shape=(3, 3, num_times),
                out_shape=(3, 3, num_times),
                in_subscripts='ijn',
                out_subscripts='jin',
                in_name='rot_mtx_t_i_3x3xn',
                out_name='rot_mtx_i_t_3x3xn',
            )
            coupled_group.add_subsystem('rot_mtx_i_t_3x3xn_comp',
                                        comp,
                                        promotes=['*'])

            comp = MtxVecComp(
                num_times=num_times,
                mtx_name='rot_mtx_i_t_3x3xn',
                vec_name='drag_unit_vec_t_3xn',
                out_name='drag_unit_vec_3xn',
            )
            coupled_group.add_subsystem('drag_unit_vec_3xn_comp',
                                        comp,
                                        promotes=['*'])

            comp = PowerCombinationComp(shape=shape,
                                        out_name='drag_3xn',
                                        powers_dict=dict(
                                            drag_unit_vec_3xn=1.,
                                            drag_scalar_3xn=1.,
                                        ))
            coupled_group.add_subsystem('drag_3xn_comp', comp, promotes=['*'])

            coupled_group.nonlinear_solver = NonlinearBlockGS(iprint=0,
                                                              maxiter=100,
                                                              atol=1e-14,
                                                              rtol=1e-12)
            coupled_group.linear_solver = LinearBlockGS(iprint=0,
                                                        maxiter=100,
                                                        atol=1e-14,
                                                        rtol=1e-12)

            self.add_subsystem('coupled_group', coupled_group, promotes=['*'])

        comp = OrbitStateDecompositionComp(
            num_times=num_times,
            position_name='position_km',
            velocity_name='velocity_km_s',
            orbit_state_name='orbit_state_km',
        )
        self.add_subsystem('orbit_state_decomposition_comp',
                           comp,
                           promotes=['*'])

        comp = LinearCombinationComp(
            shape=shape,
            out_name='position',
            coeffs_dict=dict(position_km=1.e3),
        )
        self.add_subsystem('position_comp', comp, promotes=['*'])

        comp = LinearCombinationComp(
            shape=shape,
            out_name='velocity',
            coeffs_dict=dict(velocity_km_s=1.e3),
        )
        self.add_subsystem('velocity_comp', comp, promotes=['*'])

        #

        group = DecomposeVectorGroup(
            num_times=num_times,
            vec_name='position_km',
            norm_name='radius_km',
            unit_vec_name='position_unit_vec',
        )
        self.add_subsystem('position_decomposition_group',
                           group,
                           promotes=['*'])

        group = DecomposeVectorGroup(
            num_times=num_times,
            vec_name='velocity_km_s',
            norm_name='speed_km_s',
            unit_vec_name='velocity_unit_vec',
        )
        self.add_subsystem('velocity_decomposition_group',
                           group,
                           promotes=['*'])

        #

        comp = LinearCombinationComp(
            shape=(num_times, ),
            out_name='altitude_km',
            coeffs_dict=dict(radius_km=1., radius_earth_km=-1.),
        )
        self.add_subsystem('altitude_km_comp', comp, promotes=['*'])

        comp = KSComp(
            in_name='altitude_km',
            out_name='ks_altitude_km',
            shape=(1, ),
            constraint_size=num_times,
            rho=100.,
            lower_flag=True,
        )
        comp.add_constraint('ks_altitude_km', lower=450.)
        self.add_subsystem('ks_altitude_km_comp', comp, promotes=['*'])

        comp = PowerCombinationComp(shape=(
            6,
            num_times,
        ),
                                    out_name='relative_orbit_state_sq',
                                    powers_dict={
                                        'relative_orbit_state': 2.,
                                    })
        self.add_subsystem('relative_orbit_state_sq_comp',
                           comp,
                           promotes=['*'])

        comp = ScalarContractionComp(
            shape=(
                6,
                num_times,
            ),
            out_name='relative_orbit_state_sq_sum',
            in_name='relative_orbit_state_sq',
        )
        self.add_subsystem('relative_orbit_state_sq_sum_comp',
                           comp,
                           promotes=['*'])

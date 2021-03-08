import numpy as np
from openmdao.api import LinearBlockGS, NonlinearBlockGS

import omtools.api as ot
from lsdo_cubesat.orbit.initial_orbit_comp import InitialOrbitComp
from lsdo_cubesat.orbit.orbit_state_decomposition_comp import \
    OrbitStateDecompositionComp
from lsdo_cubesat.orbit.relative_orbit_rk4_comp import RelativeOrbitRK4Comp
from lsdo_cubesat.orbit.rot_mtx_t_i_comp import RotMtxTIComp
from lsdo_cubesat.utils.api import (ArrayReorderComp, LinearCombinationComp,
                                    PowerCombinationComp,
                                    ScalarContractionComp)
from lsdo_cubesat.utils.comps.array_comps.array_expansion_comp import \
    ArrayExpansionComp
from lsdo_cubesat.utils.decompose_vector_group import DecomposeVectorGroup
from lsdo_cubesat.utils.ks_comp import KSComp
from lsdo_cubesat.utils.mtx_vec_comp import MtxVecComp
from omtools.api import Group


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

        drag_unit_vec_t_3xn = self.create_indep_var(
            'drag_unit_vec_t_3xn',
            val=drag_unit_vec,
        )
        dry_mass = self.create_indep_var(
            'dry_mass',
            val=cubesat['dry_mass'],
            shape=(num_times, ),
        )
        initial_orbit_state = self.create_indep_var(
            'initial_orbit_state',
            val=cubesat['initial_orbit_state'],
        )

        battery_mass = self.declare_input('battery_mass')
        battery_mass_exp = ot.expand(battery_mass, (num_times, ))
        propellant_mass = self.declare_input(
            'propellant_mass',
            shape=(num_times, ),
        )

        mass = self.register_output(
            'mass',
            dry_mass + propellant_mass + battery_mass_exp,
        )

        with self.create_group('drag_group') as drag_group:
            thrust_3xn = drag_group.declare_input(
                'thrust_3xn',
                shape=(3, num_times),
            )
            drag_3xn = drag_group.create_output(
                'drag_3xn',
                shape=(3, num_times),
            )

            force_3xn = drag_group.register_output(
                'force_3xn',
                thrust_3xn + drag_3xn,
            )
            drag_group.add_subsystem(
                'relative_orbit_rk4_comp',
                RelativeOrbitRK4Comp(
                    num_times=num_times,
                    step_size=step_size,
                ),
                promotes=['*'],
            )
            relative_orbit_state = drag_group.declare_input(
                'relative_orbit_state',
                shape=(6, num_times),
            )

            reference_orbit_state = drag_group.declare_input(
                'reference_orbit_state',
                shape=(6, num_times),
            )
            orbit_state = drag_group.register_output(
                'orbit_state',
                relative_orbit_state + reference_orbit_state,
            )
            orbit_state_km = drag_group.register_output(
                'orbit_state_km',
                1e-3 * orbit_state,
            )
            drag_group.add_subsystem(
                'rot_mtx_t_i_3x3xn_comp',
                RotMtxTIComp(num_times=num_times),
                promotes=['*'],
            )
            rot_mtx_t_i_3x3xn = drag_group.declare_input(
                'rot_mtx_t_i_3x3xn',
                shape=(3, 3, num_times),
            )

            rot_mtx_i_t_3x3xn = drag_group.register_output(
                'rot_mtx_i_t_3x3xn',
                ot.einsum(
                    rot_mtx_t_i_3x3xn,
                    subscripts='ijk->jik',
                ),
            )

            drag_unit_vec_t_3xn = drag_group.declare_input(
                'drag_unit_vec_t_3xn',
                shape=(3, num_times),
            )

            drag_unit_vec_3xn = ot.einsum(
                rot_mtx_i_t_3x3xn,
                drag_unit_vec_t_3xn,
                subscripts='ijk,jk->ik',
            )
            drag_scalar_3xn = drag_group.declare_input(
                'drag_scalar_3xn',
                shape=(3, num_times),
            )
            drag_3xn.define(drag_unit_vec_3xn * drag_scalar_3xn)

            drag_group.nonlinear_solver = NonlinearBlockGS(
                iprint=0,
                maxiter=100,
                atol=1e-14,
                rtol=1e-12,
            )
            drag_group.linear_solver = LinearBlockGS(
                iprint=0,
                maxiter=100,
                atol=1e-14,
                rtol=1e-12,
            )

        relative_orbit_state = self.declare_input(
            'relative_orbit_state',
            shape=(6, num_times),
        )
        orbit_state_km = self.declare_input('orbit_state_km',
                                            shape=(6, num_times))
        position_km = orbit_state_km[:3, :]
        velocity_km_s = orbit_state_km[3:, :]
        position = 1e3 * position_km
        velocity = 1e3 * velocity_km_s
        ones = self.declare_input(
            'ones',
            shape=(3, ),
            val=1,
        )
        radius_km = self.register_output(
            'radius_km',
            ot.einsum(
                ones,
                ot.pnorm(position_km, axis=0),
                subscripts='i,k->ik',
                partial_format='sparse',
            ),
        )
        position_unit_vec = position_km / radius_km

        speed_km_s = self.register_output(
            'speed_km_s',
            ot.einsum(
                ones,
                ot.pnorm(velocity_km_s, axis=0),
                subscripts='i,k->ik',
                partial_format='sparse',
            ),
        )
        velocity_unit_vec = velocity_km_s / speed_km_s

        altitude_km = self.register_output(
            'altitude_km',
            radius_km + cubesat['radius_earth_km'],
        )
        ks_altitude_km = self.register_output(
            'ks_altitude_km',
            ot.min(
                altitude_km,
                rho=100,
            ),
        )
        self.add_constraint('ks_altitude_km', lower=450.)

        relative_orbit_state_sq_sum = self.register_output(
            'relative_orbit_state_sq_sum',
            ot.pnorm(relative_orbit_state, axis=0)**2,
        )


if __name__ == "__main__":
    from openmdao.api import Problem
    from openmdao.api import n2
    from lsdo_cubesat.utils.api import get_bspline_mtx
    from lsdo_cubesat.options.cubesat import Cubesat
    initial_orbit_state_magnitude = np.array([1e-3] * 3 + [1e-3] * 3)
    num_times = 30
    num_cp = 10
    prob = Problem()
    prob.model = OrbitGroup(
        num_times=num_times,
        num_cp=num_cp,
        step_size=0.1,
        cubesat=Cubesat(
            name='sunshade',
            dry_mass=1.3,
            initial_orbit_state=initial_orbit_state_magnitude *
            np.random.rand(6),
            approx_altitude_km=500.,
            specific_impulse=47.,
            apogee_altitude=500.001,
            perigee_altitude=499.99,
        ),
        mtx=get_bspline_mtx(num_cp, num_times, order=4),
    )
    prob.setup()
    # prob.run_model()
    n2(prob)

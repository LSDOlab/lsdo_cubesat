import numpy as np
from openmdao.api import ExecComp, IndepVarComp

import omtools.api as ot
from lsdo_cubesat.propulsion.propellant_mass_rk4_comp import \
    PropellantMassRK4Comp
from lsdo_cubesat.utils.api import (ArrayExpansionComp, BsplineComp,
                                    LinearCombinationComp,
                                    PowerCombinationComp)
from lsdo_cubesat.utils.mtx_vec_comp import MtxVecComp
from omtools.api import Group


class PropulsionGroup(Group):
    """
    This Goup computes the mass and volume of the total propellant
    consumed based on thrust profile.

    Options
    ----------
    num_times : int
        Number of time steps over which to integrate dynamics
    num_cp : int
        Dimension of design variables/number of control points for
        BSpline components.
    step_size : float
        Constant time step size to use for integration
    cubesat : Cubesat
        Cubesat OptionsDictionary with initial orbital elements,
        specific impulse
    mtx : array
        Matrix that translates control points (num_cp) to actual points
        (num_times)
    """
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

        # Define the direction of thrust in the body frame; constant
        # during optimization (not a design variable)
        thrust_unit_vec = np.outer(
            np.array([1., 0., 0.]),
            np.ones(num_times),
        )
        thrust_unit_vec_b_3xn = self.create_indep_var(
            'thrust_unit_vec_b_3xn',
            val=thrust_unit_vec,
        )
        # Omnidirectional thrust
        self.add_design_var('thrust_unit_vec_b_3xn')

        # Magnitude of thrust set by optimizer (design variable)
        # TODO: get upper and lower bounds
        # TODO: Do the upper and lower bounds differ for thrusters along
        # different axes (e.g. different cant angles)? Yes. Change later.
        thrust_scalar_mN_cp = self.create_indep_var(
            'thrust_scalar_mN_cp',
            val=1.e-3 * np.ones(num_cp),
            # lower=0.,
            # upper=20000,
        )
        self.add_design_var('thrust_scalar_mN_cp')

        initial_propellant_mass = self.create_indep_var(
            'initial_propellant_mass',
            val=0.17,
        )

        rot_mtx_i_b_3x3xn = self.declare_input(
            'rot_mtx_i_b_3x3xn',
            (3, 3, num_times),
        )

        thrust_unit_vec_3xn = self.register_output(
            'thrust_unit_vec_3xn',
            ot.einsum(
                rot_mtx_i_b_3x3xn,
                thrust_unit_vec_b_3xn,
                subscripts='ijk,jk->jk',
            ),
        )

        thrust_scalar_cp = self.register_output(
            'thrust_scalar_cp',
            1e-3 * thrust_scalar_mN_cp,
        )

        self.add_subsystem(
            'thrust_scalar_comp',
            BsplineComp(
                num_pt=num_times,
                num_cp=num_cp,
                jac=mtx,
                in_name='thrust_scalar_cp',
                out_name='thrust_scalar',
            ),
            promotes=['*'],
        )
        thrust_scalar = self.declare_input('thrust_scalar',
                                           shape=(1, num_times))

        ones = self.declare_input('ones', shape=(3, ), val=1)
        thrust_scalar_3xn = self.register_output(
            'thrust_scalar_3xn',
            ot.einsum(
                ones,
                thrust_scalar,
                subscripts='i,jk->ik',
            ),
        )

        # Full time history of thrust in all three axes (inertial frame)
        thrust_3xn = self.register_output(
            'thrust_3xn',
            thrust_scalar_3xn * thrust_unit_vec_3xn,
        )

        mass_flow_rate = self.register_output(
            'mass_flow_rate',
            -1. / (cubesat['acceleration_due_to_gravity'] *
                   cubesat['specific_impulse']) * thrust_scalar,
        )

        # NOTE: cannot convert integrators to omtools.Group
        self.add_subsystem(
            'propellant_mass_rk4_comp',
            PropellantMassRK4Comp(
                num_times=num_times,
                step_size=step_size,
            ),
            promotes=['*'],
        )
        propellant_mass = self.declare_input(
            'propellant_mass',
            shape=(1, num_times),
        )

        total_propellant_used = self.register_output(
            'total_propellant_used',
            propellant_mass[0, 0] - propellant_mass[0, num_times - 1],
        )

        # NOTE: Using Ideal Gas Law
        # boltzmann = 1.380649e-23
        # avogadro = 6.02214076e23
        boltzmann_avogadro = 1.380649 * 6.02214076
        # https://advancedspecialtygases.com/pdf/R-236FA_MSDS.pdf
        r236fa_molecular_mass_kg = 152.05 / 1000
        pressure = 100 * 6895
        temperature = 273.15 + 56
        # (273.15+25)*1.380649*6.02214076/(152.05/1000)/(100*6895)

        # TODO: constraint?
        total_propellant_volume = self.register_output(
            'total_propellant_volume',
            (temperature * boltzmann_avogadro / r236fa_molecular_mass_kg /
             pressure) * total_propellant_used,
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
    prob.model = PropulsionGroup(
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
    prob.run_model()
    n2(prob)

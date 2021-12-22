import numpy as np

from csdl import Model
import csdl

from csdl.utils.get_bspline_mtx import get_bspline_mtx

from lsdo_cubesat.propulsion.propellant_mass_rk4_integrator import PropellantMassRK4Integrator


class Propulsion(Model):
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
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('num_cp', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('omnidirectional', types=bool, default=True)
        self.parameters.declare('cubesat')
        self.parameters.declare('max_thrust', types=float, default=20000.)

    def define(self):
        num_times = self.parameters['num_times']
        num_cp = self.parameters['num_cp']
        step_size = self.parameters['step_size']
        cubesat = self.parameters['cubesat']
        omnidirectional = self.parameters['omnidirectional']
        max_thrust = self.parameters['max_thrust']

        # kg
        initial_propellant_mass = self.create_input(
            'initial_propellant_mass',
            val=0.17,
        )

        self.add_design_variable(
            'initial_propellant_mass',
            lower=0.,
        )

        if omnidirectional is True:
            thrust_cp = self.create_input(
                'thrust_cp',
                val=0.0001,
                shape=(3, num_cp),
            )
            self.add_design_variable('thrust_cp',
                                     # lower=-max_thrust,
                                     # upper=max_thrust,
                                     )
            v = get_bspline_mtx(num_cp, num_times).toarray()
            bspline_mtx = self.declare_variable(
                'bspline_mtx',
                val=v,
                shape=v.shape,
            )
            thrust_3xn = csdl.einsum(thrust_cp,
                                     bspline_mtx,
                                     subscripts='ij,kj->ik')
            thrust_scalar = csdl.sum(thrust_3xn, axes=(0, ))

        else:
            thrust_unit_vec = np.outer(
                np.array([1., 0., 0.]),
                np.ones(num_times),
            )

            thrust_unit_vec_b_3xn = self.declare_variable(
                'thrust_unit_vec_b_3xn',
                val=thrust_unit_vec,
            )

            thrust_scalar_mN_cp = self.create_input(
                'thrust_scalar_mN_cp',
                val=1.e-3 * np.ones(num_cp),
            )
            self.add_design_variable('thrust_scalar_mN_cp',
                                     lower=0.,
                                     upper=max_thrust)

            rot_mtx_i_b_3x3xn = self.declare_variable(
                'rot_mtx_i_b_3x3xn',
                shape=(3, 3, num_times),
            )
            thrust_unit_vec_3xn = csdl.einsum(
                rot_mtx_i_b_3x3xn,
                thrust_unit_vec_b_3xn,
                subscripts='ijk, jk->ik',
            )
            self.register_output('thrust_unit_vec_3xn', thrust_unit_vec_3xn)

            thrust_scalar_cp = 1.e-3 * thrust_scalar_mN_cp
            thrust_scalar = csdl.matvec(get_bspline_mtx(num_cp, num_times),
                                        thrust_scalar_cp)
            thrust_scalar_3xn = csdl.expand(thrust_scalar,
                                            shape=(3, num_times),
                                            indices='i->ji')
            thrust_3xn = thrust_unit_vec_3xn * thrust_scalar_3xn

        self.register_output('thrust_3xn', thrust_3xn)

        mass_flow_rate = csdl.reshape(
            -1. / (cubesat['acceleration_due_to_gravity'] *
                   cubesat['specific_impulse']) * thrust_scalar,
            (1, num_times),
        )

        self.register_output('mass_flow_rate', mass_flow_rate)

        propellant_mass = csdl.custom(
            mass_flow_rate,
            initial_propellant_mass,
            op=PropellantMassRK4Integrator(
                num_times=num_times,
                step_size=step_size,
            ),
        )
        self.register_output(
            'propellant_mass',
            propellant_mass,
        )

        total_propellant_used = propellant_mass[0, 0] - propellant_mass[0, -1]
        self.register_output('total_propellant_used', total_propellant_used)

        # NOTE: Use Ideal Gas Law
        # boltzmann = 1.380649e-23
        # avogadro = 6.02214076e23
        boltzmann_avogadro = 1.380649 * 6.02214076
        # https://advancedspecialtygases.com/pdf/R-236FA_MSDS.pdf
        r236fa_molecular_mass_kg = 152.05 / 1000
        pressure = 100 * 6895
        temperature = 273.15 + 56
        # (273.15+25)*1.380649*6.02214076/(152.05/1000)/(100*6895)
        total_propellant_volume = temperature * boltzmann_avogadro / r236fa_molecular_mass_kg / pressure * total_propellant_used

        self.register_output(
            'total_propellant_volume',
            total_propellant_volume,
        )

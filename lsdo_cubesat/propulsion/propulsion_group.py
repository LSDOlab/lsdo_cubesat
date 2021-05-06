import numpy as np

from omtools.api import Group
import omtools.api as ot

from lsdo_cubesat.utils.api import ArrayExpansionComp, BsplineComp, PowerCombinationComp, LinearCombinationComp

from lsdo_cubesat.utils.mtx_vec_comp import MtxVecComp
from lsdo_cubesat.propulsion.propellant_mass_rk4_comp import PropellantMassRK4Comp


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

        shape = (3, num_times)

        thrust_unit_vec = np.outer(
            np.array([1., 0., 0.]),
            np.ones(num_times),
        )

        #comp = IndepVarComp()
        #comp.add_output('thrust_unit_vec_b_3xn', val=thrust_unit_vec)
        #comp.add_output('thrust_scalar_mN_cp', val=1.e-3 * np.ones(num_cp))
        #comp.add_output('initial_propellant_mass', 0.17)
        #comp.add_design_var('thrust_scalar_mN_cp', lower=0., upper=20000)
        #self.add_subsystem('inputs_comp', comp, promotes=['*'])

        thrust_unit_vec_b_3xn = self.create_output('thrust_unit_vec_b_3xn',
                                                   val=thrust_unit_vec,
                                                   shape=shape)

        thrust_scalar_mN_cp = self.create_output('thrust_scalar_mN_cp',
                                                 val=1.e-3 * np.ones(num_cp),
                                                 shape=shape)

        initial_propellant_mass = self.create_output('initial_propellant_mass',
                                                     val=0.17,
                                                     shape=shape)

        comp = MtxVecComp(
            num_times=num_times,
            mtx_name='rot_mtx_i_b_3x3xn',
            vec_name='thrust_unit_vec_b_3xn',
            out_name='thrust_unit_vec_3xn',
        )
        self.add_subsystem('thrust_unit_vec_3xn_comp', comp, promotes=['*'])

        comp = LinearCombinationComp(
            shape=(num_cp, ),
            out_name='thrust_scalar_cp',
            coeffs_dict=dict(thrust_scalar_mN_cp=1.e-3),
        )
        self.add_subsystem('thrust_scalar_cp_comp', comp, promotes=['*'])

        comp = BsplineComp(
            num_pt=num_times,
            num_cp=num_cp,
            jac=mtx,
            in_name='thrust_scalar_cp',
            out_name='thrust_scalar',
        )
        self.add_subsystem('thrust_scalar_comp', comp, promotes=['*'])

        comp = ArrayExpansionComp(
            shape=shape,
            expand_indices=[0],
            in_name='thrust_scalar',
            out_name='thrust_scalar_3xn',
        )
        self.add_subsystem('thrust_scalar_3xn_comp', comp, promotes=['*'])

        comp = PowerCombinationComp(
            shape=shape,
            out_name='thrust_3xn',
            powers_dict=dict(
                thrust_unit_vec_3xn=1.,
                thrust_scalar_3xn=1.,
            ),
        )
        self.add_subsystem('thrust_3xn_comp', comp, promotes=['*'])

        comp = LinearCombinationComp(
            shape=(num_times, ),
            out_name='mass_flow_rate',
            coeffs_dict=dict(thrust_scalar=-1. /
                             (cubesat['acceleration_due_to_gravity'] *
                              cubesat['specific_impulse']), ),
        )
        self.add_subsystem('mass_flow_rate_comp', comp, promotes=['*'])

        comp = PropellantMassRK4Comp(
            num_times=num_times,
            step_size=step_size,
        )
        self.add_subsystem('propellant_mass_rk4_comp', comp, promotes=['*'])

        comp = ExecComp(
            'total_propellant_used=propellant_mass[0] - propellant_mass[-1]',
            propellant_mass=np.empty(num_times),
        )
        self.add_subsystem('total_propellant_used_comp', comp, promotes=['*'])

        # NOTE: Use Ideal Gas Law
        # boltzmann = 1.380649e-23
        # avogadro = 6.02214076e23
        boltzmann_avogadro = 1.380649 * 6.02214076
        # https://advancedspecialtygases.com/pdf/R-236FA_MSDS.pdf
        r236fa_molecular_mass_kg = 152.05 / 1000
        pressure = 100 * 6895
        temperature = 273.15 + 56
        # (273.15+25)*1.380649*6.02214076/(152.05/1000)/(100*6895)
        self.add_subsystem(
            'compute_propellant_volume',
            PowerCombinationComp(
                shape=(1, ),
                out_name='total_propellant_volume',
                coeff=temperature * boltzmann_avogadro /
                r236fa_molecular_mass_kg / pressure,
                powers_dict=dict(total_propellant_used=1.0, ),
            ),
            promotes=['*'],
        )

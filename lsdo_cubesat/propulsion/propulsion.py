from ozone.api import ODEProblem
from csdl import Model
from csdl.utils.get_bspline_mtx import get_bspline_mtx
import csdl
import numpy as np


class PropellantMassDynamics(Model):

    def initialize(self):
        self.parameters.declare('num_nodes', types=int)

    def define(self):
        n = self.parameters['num_nodes']

        mass_flow_rate = self.declare_variable('mass_flow_rate', shape=(n, 1))
        # unused, so we need to create an input
        propellant_mass = self.create_input('propellant_mass', shape=(n, 1))

        dm_dt = -mass_flow_rate
        self.register_output('dm_dt', dm_dt)


class ODEProblemTest(ODEProblem):

    def setup(self):
        self.add_parameter('mass_flow_rate',
                           dynamic=True,
                           shape=(self.num_times, 1))
        self.add_state(
            'propellant_mass',
            'dm_dt',
            shape=(1, ),
            initial_condition_name='initial_propellant_mass',
            output='propellant_mass',
        )
        self.add_times(step_vector='h')
        self.set_ode_system(PropellantMassDynamics)


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
        self.parameters.declare('cubesat')
        self.parameters.declare('max_thrust', types=float, default=20000.)

    def define(self):
        num_times = self.parameters['num_times']
        num_cp = self.parameters['num_cp']
        cubesat = self.parameters['cubesat']
        max_thrust = self.parameters['max_thrust']

        # kg
        initial_propellant_mass = self.create_input(
            'initial_propellant_mass',
            val=0.17,
        )

        # self.add_design_variable(
        #     'initial_propellant_mass',
        #     lower=0.,
        # )

        # NOTE: need to initialize nonzero values because approximation
        # to absolute value has derivative 0 at 0
        thrust_cp = self.create_input(
            'thrust_cp',
            # val=0.,
            # val=1e-5*(np.random.rand(num_cp*3).reshape((num_cp, 3))-0.5),
            # val=(np.random.rand(num_cp*3).reshape((num_cp, 3))-0.5),
            val=1e-11 * np.random.rand(num_cp * 3).reshape((num_cp, 3)) *
            np.einsum('i,j->ij', np.exp(-np.arange(num_cp)), np.ones(3)),
            shape=(num_cp, 3),
        )
        self.add_design_variable(
            'thrust_cp',
            # lower=-max_thrust,
            # upper=max_thrust,
            scaler=1e5,
        )
        v = get_bspline_mtx(num_cp, num_times).toarray()
        bspline_mtx = self.declare_variable(
            'bspline_mtx',
            val=v,
            shape=v.shape,
        )
        thrust = csdl.einsum(bspline_mtx, thrust_cp, subscripts='kj,ji->ki')
        self.register_output('thrust', thrust)

        # continuous and differentiable approximation to absolute
        # value, used to compute mass flow rate
        # reg = 1e-1
        # total_thrust = csdl.sum((thrust**2+reg)**0.5, axes=(1,))
        # total_thrust = csdl.sum(thrust * csdl.tanh(5 * thrust), axes=(1, ))
        total_thrust = csdl.sum(thrust * csdl.tanh(1e5 * thrust), axes=(1, ))

        mass_flow_rate = total_thrust / (cubesat['acceleration_due_to_gravity']
                                         * cubesat['specific_impulse'])

        self.register_output('mass_flow_rate', mass_flow_rate)

        self.add(
            ODEProblemTest('RK4', 'time-marching',
                           num_times).create_solver_model(),
            name='propellant_mass_integrator',
        )
        propellant_mass = self.declare_variable('propellant_mass',
                                                shape=(num_times, 1))

        total_mass = csdl.expand(
            csdl.reshape(cubesat['dry_mass'] + propellant_mass, (num_times, )),
            shape=(num_times, 3),
            indices='i->ij',
        )
        self.register_output('total_mass', total_mass)

        final_propellant_mass = propellant_mass[-1, 0]
        self.register_output('final_propellant_mass', final_propellant_mass)
        # self.add_constraint('final_propellant_mass', lower=0)
        total_propellant_used = propellant_mass[0, 0] - final_propellant_mass
        self.register_output('total_propellant_used', total_propellant_used)

        # NOTE: Use Ideal Gas Law
        # boltzmann = 1.380649e-23
        # avogadro = 6.02214076e23
        boltzmann_avogadro = 1.380649 * 6.02214076
        # https://advancedspecialtygases.com/pdf/R-236FA_MSDS.pdf
        r236fa_molecular_mass_kg = 152.05 / 1000
        # 100 psi to Pa
        pressure = 100 * 6.894757e5
        temperature = 273.15 + 56
        # (273.15+25)*1.380649*6.02214076/(152.05/1000)/(100*6.894757e5)
        # total_propellant_volume = temperature * boltzmann_avogadro / r236fa_molecular_mass_kg / pressure * initial_propellant_mass

        # self.register_output(
        #     'total_propellant_volume',
        #     total_propellant_volume,
        # )


if __name__ == "__main__":
    from csdl_om import Simulator
    from csdl import GraphRepresentation
    from lsdo_cubesat.specifications.cubesat_spec import CubesatSpec

    rep = GraphRepresentation(
        Propulsion(
            num_times=100,
            num_cp=20,
            cubesat=CubesatSpec(
                name='optics',
                dry_mass=1.3,
                initial_orbit_state=np.array([
                    40,
                    0,
                    0,
                    0.,
                    0.,
                    0.,
                    # 1.76002146e+03,
                    # 6.19179823e+03,
                    # 6.31576531e+03,
                    # 4.73422022e-05,
                    # 1.26425269e-04,
                    # 5.39731211e-05,
                ]),
                specific_impulse=47.,
                perigee_altitude=500.,
                apogee_altitude=500.,
            )))
    sim = Simulator(rep)
    sim.visualize_implementation()

from csdl import Model
import csdl

from lsdo_cubesat.operations.relative_orbit_integrator import RelativeOrbitIntegrator
from lsdo_cubesat.utils.compute_norm_unit_vec import compute_norm_unit_vec
from lsdo_cubesat.constants import RADII

radius_earth = RADII['Earth']


class RelativeOrbit(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('cubesat')

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        cubesat = self.parameters['cubesat']

        dry_mass = self.create_input('dry_mass',
                                     val=cubesat['dry_mass'],
                                     shape=num_times)
        initial_orbit_state = self.create_input(
            'initial_orbit_state',
            val=cubesat['initial_orbit_state'],
        )

        reference_orbit_state_km = self.declare_variable(
            'reference_orbit_state_km', shape=(6, num_times))
        radius_m = csdl.reshape(
            csdl.pnorm(reference_orbit_state_km[:3, :], axis=0) * 1e3,
            (1, num_times),
        )
        self.register_output('radius_m', radius_m)
        propellant_mass = self.declare_variable(
            'propellant_mass',
            shape=num_times,
        )

        mass = csdl.expand(dry_mass + propellant_mass, (1, num_times),
                           indices='i->ji')
        self.register_output('mass', mass)

        force_3xn = self.declare_variable('force_3xn', shape=(3, num_times))
        mass = self.declare_variable('mass', shape=(1, num_times))
        initial_orbit_state = self.declare_variable(
            'initial_orbit_state',
            val=cubesat['initial_orbit_state'],
        )

        relative_orbit_state_m = csdl.custom(
            force_3xn,
            mass,
            radius_m,
            initial_orbit_state,
            op=RelativeOrbitIntegrator(
                num_times=num_times,
                step_size=step_size,
            ),
        )
        relative_orbit_state_m = self.register_output('relative_orbit_state_m',
                                                      relative_orbit_state_m)
        orbit_state_km = relative_orbit_state_m / 1e3 + reference_orbit_state_km
        self.register_output('orbit_state_km', orbit_state_km)

        # TODO: Why was Aobo taking the norm of the entire state and not
        # # only the position relative to the reference orbit?
        # relative_orbit_state_sq = relative_orbit_state_m[:3, :]**2

        # relative_orbit_state_sq_sum = csdl.sum(relative_orbit_state_sq)
        # self.register_output('relative_orbit_state_sq_sum',
        #                      relative_orbit_state_sq_sum)

        position_km = orbit_state_km[:3, :]
        velocity_km_s = orbit_state_km[3:, :]

        radius_km, position_unit_vec = compute_norm_unit_vec(
            position_km, num_times=num_times)
        speed_km_s, velocity_unit_vec = compute_norm_unit_vec(
            velocity_km_s, num_times=num_times)

        altitude_km = radius_km - radius_earth
        ks_altitude_km = csdl.min(
            altitude_km,
            rho=100.,
        )

        self.register_output('position_km', position_km)
        self.register_output('velocity_km_s', velocity_km_s)
        self.register_output('ks_altitude_km', ks_altitude_km)
        self.add_constraint('ks_altitude_km', lower=450.)

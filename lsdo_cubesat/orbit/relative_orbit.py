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
        self.parameters.declare('min_alt', default=450., types=float)
        self.parameters.declare('cubesat')

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        min_alt = self.parameters['min_alt']
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
        altitude_m = csdl.pnorm(reference_orbit_state_km * 1000 +
                                relative_orbit_state_m,
                                axis=0)

        self.register_output('altitude_km', altitude_km)
        self.register_output('altitude_m', altitude_m)
        self.register_output('position_km', position_km)
        self.register_output('velocity_km_s', velocity_km_s)
        # self.register_output('min_altitude_m',
        #                      csdl.min(
        #                          altitude_m,
        #                          rho=10. / 1e-7,
        #                      ))
        # self.add_constraint('min_altitude_m', lower=min_alt * 1000)
        self.register_output('min_altitude_km',
                             csdl.min(
                                 altitude_km,
                                 rho=10. / 1e3,
                             ))
        self.add_constraint('min_altitude_km', lower=min_alt)


if __name__ == '__main__':

    from lsdo_cubesat.parameters.cubesat import CubesatParams
    from csdl_om import Simulator
    import numpy as np

    num_times = 40
    step_size = 95 * 60 / (num_times - 1)

    initial_orbit_state = np.array([1e-3] * 3 + [1e-3] * 3)
    sim = Simulator(
        RelativeOrbit(num_times=num_times,
                      step_size=step_size,
                      cubesat=CubesatParams(
                          name='optics',
                          dry_mass=1.3,
                          initial_orbit_state=initial_orbit_state *
                          np.random.rand(6) * 1e-3,
                          specific_impulse=47.,
                          perigee_altitude=500.,
                          apogee_altitude=500.,
                      )))
    sim.check_partials(compact_print=True)

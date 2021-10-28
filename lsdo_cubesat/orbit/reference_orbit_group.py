import numpy as np

from lsdo_cubesat.utils.decompose_vector_group import compute_norm_unit_vec
from lsdo_cubesat.orbit.keplerian_to_cartesian import KeplerianToCartesian
from lsdo_cubesat.orbit.reference_orbit_rk4_comp import ReferenceOrbitRK4Comp
from csdl import Model
import csdl
from lsdo_cubesat.constants import RADII

radius_earth = RADII['Earth']


class ReferenceOrbit(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']

        self.add(
            KeplerianToCartesian(
                periapsis=radius_earth + 500.,
                apoapsis=radius_earth + 500.,
                longitude=66.279 * np.pi / 180.,
                inclination=82.072 * np.pi / 180.,
                argument_of_periapsis=0.,
                true_anomaly=337.987 * np.pi / 180.,
                r_name='initial_radius_km',
                v_name='initial_velocity_km_s',
            ),
            name='initial_orbit',
        )
        r = self.declare_variable('initial_radius_km', shape=(3, ))
        v = self.declare_variable('initial_velocity_km_s', shape=(3, ))

        initial_orbit_state_km = self.create_output('initial_orbit_state_km',
                                                    shape=(6, ))
        initial_orbit_state_km[:3] = r
        initial_orbit_state_km[3:] = v

        # TODO: (nice to have) propagate reference orbit without rk4

        reference_orbit_state_km = csdl.custom(
            initial_orbit_state_km,
            op=ReferenceOrbitRK4Comp(
                num_times=num_times,
                step_size=step_size,
            ),
        )
        self.register_output(
            'reference_orbit_state_km',
            reference_orbit_state_km,
        )

        position_km = reference_orbit_state_km[:3, :]
        velocity_km_s = reference_orbit_state_km[3:, :]
        self.register_output('position_km', position_km)
        self.register_output('velocity_km_s', velocity_km_s)

        reference_orbit_state = 1.e3 * reference_orbit_state_km
        self.register_output('reference_orbit_state', reference_orbit_state)

        radius_km, position_unit_vec = compute_norm_unit_vec(
            position_km,
            num_times=num_times,
        )
        self.register_output('position_unit_vec', position_unit_vec)
        _, velocity_unit_vec = compute_norm_unit_vec(
            velocity_km_s,
            num_times=num_times,
        )

        self.register_output('velocity_unit_vec', velocity_unit_vec)
        radius = 1.e3 * radius_km
        self.register_output('radius', radius)


if __name__ == "__main__":

    from csdl_om import Simulator

    sim = Simulator(ReferenceOrbit(
        num_times=1,
        step_size=0.1,
    ))
    sim.visualize_implementation()

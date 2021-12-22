import numpy as np

from lsdo_cubesat.orbit.keplerian_to_cartesian import KeplerianToCartesian
from lsdo_cubesat.operations.reference_orbit_integrator import ReferenceOrbitIntegrator
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
        # self.add(
        #     KeplerianToCartesian(
        #         central_body='Earth',
        #         periapsis=radius_earth + 500,
        #         apoapsis=radius_earth + 500.,
        #         longitude=0.,
        #         inclination=0.,
        #         argument_of_periapsis=0.,
        #         true_anomaly=0.,
        #         r_name='initial_radius_km',
        #         v_name='initial_velocity_km_s',
        #     ),
        #     name='initial_orbit',
        # )
        r = self.declare_variable('initial_radius_km', shape=(3, ))
        v = self.declare_variable('initial_velocity_km_s', shape=(3, ))

        initial_orbit_state_km = self.create_output('initial_orbit_state_km',
                                                    shape=(6, ))
        initial_orbit_state_km[:3] = r
        initial_orbit_state_km[3:] = v

        reference_orbit_state_km = csdl.custom(
            initial_orbit_state_km,
            op=ReferenceOrbitIntegrator(
                num_times=num_times,
                step_size=step_size,
            ),
        )
        self.register_output(
            'reference_orbit_state_km',
            reference_orbit_state_km,
        )


if __name__ == "__main__":

    from csdl_om import Simulator

    import matplotlib.pyplot as plt
    num_times = 1501
    sim = Simulator(
        ReferenceOrbit(
            num_times=num_times,
            step_size=95 * 60 / (num_times - 1),
        ))
    print(sim['initial_radius_km'])
    sim.run()
    print('eccentricity', sim['eccentricity'])
    print('semimajor_axis', sim['semimajor_axis'])
    print('periapsis', sim['periapsis'])
    print('apoapsis', sim['apoapsis'])
    print('speed', sim['speed'])
    print('initial_orbit_state_km', sim['initial_orbit_state_km'])
    print('initial_radius_km', sim['initial_radius_km'])
    plt.plot(sim['reference_orbit_state_km'][0, 0],
             sim['reference_orbit_state_km'][1, 0], 'o')
    plt.plot(sim['reference_orbit_state_km'][0, :],
             sim['reference_orbit_state_km'][1, :])
    plt.show()
    # plt.plot(sim['reference_orbit_state_km'][2, :],
    #          sim['reference_orbit_state_km'][1, :])
    # plt.show()
    # plt.plot(sim['reference_orbit_state_km'][2, :],
    #          sim['reference_orbit_state_km'][0, :])
    # plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(
        sim['reference_orbit_state_km'][0, :],
        sim['reference_orbit_state_km'][1, :],
        sim['reference_orbit_state_km'][2, :],
    )
    plt.show()

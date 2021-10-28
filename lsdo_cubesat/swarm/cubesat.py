import numpy as np

from lsdo_cubesat.utils.options_dictionary import OptionsDictionary


class CubesatParams(OptionsDictionary):
    def initialize(self):
        self.declare('name', types=str)
        self.declare('dry_mass', types=float)
        self.declare('initial_orbit_state', types=np.ndarray)
        self.declare('approx_altitude_km',
                     default=None,
                     types=float,
                     allow_none=True)
        self.declare('acceleration_due_to_gravity', default=9.81, types=float)
        self.declare('specific_impulse', default=47., types=float)

        self.declare('perigee_altitude', default=500.1, types=float)
        self.declare('apogee_altitude', default=499.9, types=float)
        self.declare('RAAN', default=66.279, types=float)
        self.declare('inclination', default=82.072, types=float)
        # self.declare('inclination', default=97.4, types=float)
        self.declare('argument_of_periapsis', default=0., types=float)
        self.declare('true_anomaly', default=337.987, types=float)

        self.declare('radius_earth_km', default=6371., types=float)

    def pre_setup(self):
        if self['approx_altitude_km'] is not None:
            self['acceleration_due_to_gravity'] = 9.81 * self[
                'radius_earth_km']**2 / (self['radius_earth_km'] +
                                         self['approx_altitude_km'])**2

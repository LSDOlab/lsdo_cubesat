import numpy as np

from omtools.api import Group
import omtools.api as ot

# TODO: omtools should handle complex numbers
# TODO: initialize values with arrays
# TODO: transform function defs into component classes, run omtools
# problems the way pytest runs tests?
# TODO: conditional execution at Group level?

# Constants
mu = 398600.44
Re = 6378.137
J2 = 1.08264e-3
J3 = -2.51e-6
J4 = -1.60e-6

C1 = -mu
C2 = -1.5 * mu * J2 * Re**2
C3 = -2.5 * mu * J3 * Re**3
C4 = 1.875 * mu * J4 * Re**4

# rho = 3.89e-12 # kg/m**3 atmoshperic density at altitude = 400 km with mean solar activity
# C_D = 2.2 # Drag coefficient for cube
# area = 0.1 * 0.1 # m**2 cross sectional area
drag = 1.e-6


class InitialOrbitComp(Group):
    """
    Component for defining an initial orbit using Keplerian elements

    Parameters
    ----------
    perigee_altitude : float
        Perigee Altitude (km); altitude above Earth at perigee; note, this
        is not the distance from the center of the Earth at perigee
    apogee_altitude : float
        Apogee Altitude (km); altitude above Earth at apogee; note, this
        is not the distance from the center of the Earth at apogee
    RAAN : float
        Right Ascension of Ascending Node (degrees)
    inclination : float
        inclination (degrees)
    argument_of_periapsis : float
        Argument of periapsis (degrees)
    true_anomaly : float
        True Anomaly (degrees)
    """
    def setup(self):
        self.declare_input('perigee_altitude', val=500.)
        self.declare_input('apogee_altitude', val=500.)
        self.declare_input('RAAN', val=66.279)
        self.declare_input('inclination', val=82.072)
        self.declare_input('argument_of_periapsis', val=0.0)
        self.declare_input('true_anomaly', val=337.987)

    def compute_rv(self, perigee_altitude, apogee_altitude, RAAN, inclination,
                   argument_of_periapsis, true_anomaly):
        """
        Compute position and velocity from orbital elements
        """
        Re = 6378.137
        mu = 398600.44

        def S(v):
            S = np.zeros((3, 3), complex)
            S[0, :] = [0, -v[2], v[1]]
            S[1, :] = [v[2], 0, -v[0]]
            S[2, :] = [-v[1], v[0], 0]
            return S

        def getRotation(axis, angle):
            R = np.eye(3, dtype=complex) + S(axis)*ot.sin(angle) + \
                (1 - ot.cos(angle)) * (ot.outer(axis, axis) - np.eye(3, dtype=complex))
            return R

        d2r = np.pi / 180.0
        r_perigee = Re + perigee_altitude
        r_apogee = Re + apogee_altitude
        e = (r_apogee - r_perigee) / (r_apogee + r_perigee)
        a = (r_perigee + r_apogee) / 2
        p = a * (1 - e**2)
        # h = ot.sqrt(p*mu)

        rmag0 = p / (1 + e * ot.cos(d2r * true_anomaly))

        # FIXME: this is supposed to be a complex number!
        r0_P = self.create_output('r0_P', shape=(3, ))
        r0_P[0] = rmag0 * ot.cos(d2r * true_anomaly)
        r0_P[1] = rmag0 * ot.sin(d2r * true_anomaly)
        r0_P[2] = 0

        # FIXME: this is supposed to be a complex number!
        v0_P = self.create_output('v0_P', shape=(3, ))
        v0_P[0] = -ot.sqrt(mu / p) * ot.sin(d2r * true_anomaly)
        v0_P[1] = ot.sqrt(mu / p) * (e + ot.cos(d2r * true_anomaly))
        v0_P[2] = 0

        O_IP = self.create_output('O_IP', shape=(3, 3))
        for i in range(3):
            O_IP[i, i] = 1
        O_IP = ot.dot(O_IP, getRotation(np.array([0, 0, 1]), RAAN * d2r))
        O_IP = ot.dot(O_IP, getRotation(np.array([1, 0, 0]),
                                        inclination * d2r))
        O_IP = ot.dot(
            O_IP, getRotation(np.array([0, 0, 1]),
                              argument_of_periapsis * d2r))

        r0_ECI = ot.dot(O_IP, r0_P)
        v0_ECI = ot.dot(O_IP, v0_P)
        initial_orbit_state_km = self.create_output(
            'initial_orbit_state_km',
            val=np.ones((6, )),
        )
        initial_orbit_state_km[:3] = r0_ECI.real
        initial_orbit_state_km[3:] = v0_ECI.real

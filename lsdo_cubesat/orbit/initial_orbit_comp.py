import numpy as np

from csdl import CustomExplicitOperation

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

# def perifocal_to_cartesian(
#     r: Concatenation,
#     v: Concatenation,
#     rp,
#     ra,
#     raan,
#     i,
#     arg_peri,
#     true_anomaly,
# ):
#     mu = 1
#     e = (ra - rp) / (ra + rp)
#     a = (rp + ra) / 2
#     p = a * (1 - e**2)
#     h = csdl.norm(p * mu)

#     # compute radius
#     rmag = p / (1 + e * np.cos(true_anomaly))

#     # compute speed

#     # construct position, velocity vectors in perifocal coordinates
#     r[0] = rmag * np.cos(true_anomaly)
#     r[1] = rmag * np.cos(true_anomaly)
#     r[3] = 0

#     v[0] = -mu / h * csdl.sin(true_anomaly)
#     v[1] = mu / h * (e + np.cos(true_anomaly))
#     v[3] = 0

#     # construct position, velocity in cartesian coordinates
#     R = np.zeros((3, 3))
#     # R[0] =

#     return r, v


class InitialOrbit(Model):
    def define(self):
        perigee_altitude = self.create_input('perigee_altitude',
                                             val=cubesat['perigee_altitude'])
        apogee_altitude = self.create_input('apogee_altitude',
                                            val=cubesat['apogee_altitude'])
        RAAN = self.create_input('RAAN', val=cubesat['RAAN'])
        inclination = self.create_input('inclination',
                                        val=cubesat['inclination'])
        argument_of_periapsis = self.create_input(
            'argument_of_periapsis', val=cubesat['argument_of_periapsis'])
        true_anomaly = self.create_input('perigee_altitude',
                                         val=cubesat['true_anomaly'])


class InitialOrbitComp(CustomExplicitOperation):
    def define(self):
        # Inputs
        self.add_input('perigee_altitude', 500.)
        self.add_input('apogee_altitude', 500.)
        self.add_input('RAAN', 66.279)
        self.add_input('inclination', 82.072)
        self.add_input('argument_of_periapsis', 0.0)
        self.add_input('true_anomaly', 337.987)

        # Outputs
        self.add_output(
            'initial_orbit_state_km',
            np.ones((6, )),
            desc='Initial position and velocity vectors from Earth '
            'to satellite in Earth-centered inertial frame')

        self.declare_derivatives('*', '*')

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
            R = np.eye(3, dtype=complex) + S(axis)*np.sin(angle) + \
                (1 - np.cos(angle)) * (np.outer(axis, axis) - np.eye(3, dtype=complex))
            return R

        d2r = np.pi / 180.0
        r_perigee = Re + perigee_altitude
        r_apogee = Re + apogee_altitude
        e = (r_apogee - r_perigee) / (r_apogee + r_perigee)
        a = (r_perigee + r_apogee) / 2
        p = a * (1 - e**2)
        # h = np.sqrt(p*mu)

        rmag0 = p / (1 + e * np.cos(d2r * true_anomaly))
        r0_P = np.array([
            rmag0 * np.cos(d2r * true_anomaly),
            rmag0 * np.sin(d2r * true_anomaly), 0
        ], complex)
        v0_P = np.array([
            -np.sqrt(mu / p) * np.sin(d2r * true_anomaly),
            np.sqrt(mu / p) * (e + np.cos(d2r * true_anomaly)), 0
        ], complex)

        O_IP = np.eye(3, dtype=complex)
        O_IP = np.dot(O_IP, getRotation(np.array([0, 0, 1]), RAAN * d2r))
        O_IP = np.dot(O_IP, getRotation(np.array([1, 0, 0]),
                                        inclination * d2r))
        O_IP = np.dot(
            O_IP, getRotation(np.array([0, 0, 1]),
                              argument_of_periapsis * d2r))

        r0_ECI = np.dot(O_IP, r0_P)
        v0_ECI = np.dot(O_IP, v0_P)

        return r0_ECI, v0_ECI

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        r0_ECI, v0_ECI = self.compute_rv(inputs['perigee_altitude'],
                                         inputs['apogee_altitude'],
                                         inputs['RAAN'], inputs['inclination'],
                                         inputs['argument_of_periapsis'],
                                         inputs['true_anomaly'])
        outputs['initial_orbit_state_km'][:3] = r0_ECI.real
        outputs['initial_orbit_state_km'][3:] = v0_ECI.real

    def compute_derivatives(self, inputs, J):
        """
        Calculate and save derivatives. (i.e., Jacobian).
        """
        h = 1e-16
        ih = complex(0, h)
        v = np.zeros(6, complex)
        v[:] = [
            inputs['perigee_altitude'], inputs['apogee_altitude'],
            inputs['RAAN'], inputs['inclination'],
            inputs['argument_of_periapsis'], inputs['true_anomaly']
        ]
        jacs = np.zeros((6, 6))

        # Find derivatives by complex step.
        for i in range(6):
            v[i] += ih
            r0_ECI, v0_ECI = self.compute_rv(v[0], v[1], v[2], v[3], v[4],
                                             v[5])
            v[i] -= ih
            jacs[:3, i] = r0_ECI.imag / h
            jacs[3:, i] = v0_ECI.imag / h

        J['initial_orbit_state_km', 'perigee_altitude'] = jacs[:, 0]
        J['initial_orbit_state_km', 'apogee_altitude'] = jacs[:, 1]
        J['initial_orbit_state_km', 'RAAN'] = jacs[:, 2]
        J['initial_orbit_state_km', 'inclination'] = jacs[:, 3]
        J['initial_orbit_state_km', 'argument_of_periapsis'] = jacs[:, 4]
        J['initial_orbit_state_km', 'true_anomaly'] = jacs[:, 5]


if __name__ == '__main__':

    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp

    np.random.seed(0)

    group = Group()

    comp = IndepVarComp()

    perigee_altitude = np.random.rand(1)
    apogee_altitude = np.random.rand(1)
    RAAN = np.random.rand(1)
    inclination = np.random.rand(1)
    argument_of_periapsis = np.random.rand(1)
    true_anomaly = np.random.rand(1)

    comp.add_output('perigee_altitude', val=perigee_altitude)
    comp.add_output('apogee_altitude', val=apogee_altitude)
    comp.add_output('RAAN', val=RAAN)
    comp.add_output('inclination', val=inclination)
    comp.add_output('argument_of_periapsis', val=argument_of_periapsis)
    comp.add_output('true_anomaly', val=true_anomaly)

    group.add('Inputcomp', comp, promotes=['*'])

    group.add('Statecomp_Implicit', InitialOrbitComp(), promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    print(prob['initial_orbit_state_km'].shape)
    # prob.check_partials()

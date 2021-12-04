from csdl import Model
import csdl
import numpy as np

# from lsdo_cubesat.attitude.attitude_group import AttitudeGroup
from lsdo_cubesat.attitude.attitude_cadre_style import Attitude
from lsdo_cubesat.propulsion.propulsion_group import Propulsion
# from lsdo_cubesat.aerodynamics.aerodynamics_group import AerodynamicsGroup
from lsdo_cubesat.orbit.orbit_group import OrbitGroup


class VehicleDynamics(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('num_cp', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('cubesat')

    def define(self):
        num_times = self.parameters['num_times']
        num_cp = self.parameters['num_cp']
        step_size = self.parameters['step_size']
        cubesat = self.parameters['cubesat']

        self.add(
            Propulsion(
                num_times=num_times,
                num_cp=num_cp,
                step_size=step_size,
                cubesat=cubesat,
            ),
            name='propulsion',
        )

        # self.add(
        #     AerodynamicsGroup(
        #         num_times=num_times,
        #         # num_cp=num_cp,
        #         # step_size=step_size,
        #         # cubesat=cubesat,
        #     ),
        #     name='aerodynamics',
        # )

        self.add(
            OrbitGroup(
                num_times=num_times,
                step_size=step_size,
                cubesat=cubesat,
            ),
            name='orbit',
        )

        self.connect('thrust_3xn', 'force_3xn')

        orbit_state_km = self.declare_variable(
            'orbit_state_km',
            shape=(6, num_times),
        )
        radius = orbit_state_km[:3, :]
        velocity = orbit_state_km[3:, :]

        a0 = radius / csdl.expand(
            csdl.pnorm(radius, axis=0),
            (3, num_times),
            indices='i->ji',
        )

        osculating_orbit_angular_velocity = csdl.cross(radius,
                                                       velocity,
                                                       axis=0)

        a2 = osculating_orbit_angular_velocity / csdl.expand(
            csdl.pnorm(osculating_orbit_angular_velocity, axis=0),
            (3, num_times),
            indices='i->ji',
        )
        a1 = csdl.cross(a2, a0, axis=0)

        ECI_to_RTN = self.create_output(
            'ECI_to_RTN',
            shape=(3, 3, num_times),
        )
        ECI_to_RTN[0, :, :] = csdl.expand(
            a0,
            (1, 3, num_times),
            indices='jk->ijk',
        )
        ECI_to_RTN[1, :, :] = csdl.expand(
            a1,
            (1, 3, num_times),
            indices='jk->ijk',
        )
        ECI_to_RTN[2, :, :] = csdl.expand(
            a2,
            (1, 3, num_times),
            indices='jk->ijk',
        )

        osculating_orbit_angular_speed = csdl.reshape(
            csdl.pnorm(osculating_orbit_angular_velocity, axis=0),
            (1, num_times))
        self.register_output(
            'osculating_orbit_angular_speed',
            osculating_orbit_angular_speed,
        )
        self.add(
            Attitude(
                num_times=num_times,
                step_size=step_size,
                sc_mmoi=np.array([18, 18, 6]) * 1e-3,
                rw_mmoi=28 * np.ones(3) * 1e-6,
                gravity_gradient=True,
            ),
            name='attitude',
        )

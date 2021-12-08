from csdl import Model
import csdl
import numpy as np

# from lsdo_cubesat.attitude.attitude_group import AttitudeGroup
from lsdo_cubesat.attitude.attitude_cadre_style import Attitude
from lsdo_cubesat.propulsion.propulsion_group import Propulsion
# from lsdo_cubesat.aerodynamics.aerodynamics_group import AerodynamicsGroup
from lsdo_cubesat.orbit.relative_orbit import RelativeOrbit
from lsdo_cubesat.utils.compute_norm_unit_vec import compute_norm_unit_vec


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
            RelativeOrbit(
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

        RTN_from_ECI = self.create_output(
            'RTN_from_ECI',
            shape=(3, 3, num_times),
        )
        RTN_from_ECI[0, :, :] = csdl.expand(
            a0,
            (1, 3, num_times),
            indices='jk->ijk',
        )
        RTN_from_ECI[1, :, :] = csdl.expand(
            a1,
            (1, 3, num_times),
            indices='jk->ijk',
        )
        RTN_from_ECI[2, :, :] = csdl.expand(
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
                num_cp=num_cp,
                step_size=step_size,
                gravity_gradient=True,
            ),
            name='attitude',
        )

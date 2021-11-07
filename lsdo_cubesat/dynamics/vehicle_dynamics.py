from csdl import Model
import csdl
import numpy as np

from lsdo_cubesat.attitude.attitude_group import AttitudeGroup
from lsdo_cubesat.propulsion.propulsion_group import Propulsion
from lsdo_cubesat.aerodynamics.aerodynamics_group import AerodynamicsGroup
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

        # TODO: change variable of integration to nondimensionalized time
        self.add(
            AttitudeGroup(num_times=num_times, step_size=step_size),
            name='attitude',
        )

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

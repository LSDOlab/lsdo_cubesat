import omtools.api as ot
import numpy as np
import math


class AttitudeActuatorModel(ot.Group):
    def initialize(self):
        self.options.declare('num_times')

    def setup(self):
        self.add_subsystem('reaction_wheel_speed',
                           RWSpeedRK4Comp,
                           promotes=['*'])

        rw_speed = self.declare_input('rw_speed', shape=(n, ))
        power = self.declare_input('power', shape=(n, ))
        torque = power / rw_speed

        self.register_output('torque', torque)
        self.add_constraint('rw_speed', upper=250)
        self.add_constraint('rw_speed', lower=-250)

import omtools.api as ot
from openmdao.api import NonlinearBlockGS
from lsdo_cubesat.attitude.rw_speed_rk4 import RWSpeedRK4


class AttitudeActuator(ot.Group):
    def initialize(self):
        self.options.declare('num_times')
        self.options.declare('step_size')

    def setup(self):
        num_times = self.options['num_times']
        step_size = self.options['step_size']

        # Power supplied by battery to reaction wheels
        power_x = self.declare_input('power_x', shape=(num_times, ))
        power_y = self.declare_input('power_y', shape=(num_times, ))
        power_z = self.declare_input('power_z', shape=(num_times, ))

        # Torque generated by reaction wheels; required to compute
        # reaction wheel speed
        external_torques_x = self.create_output('external_torques_x',
                                                shape=(num_times, ))
        external_torques_y = self.create_output('external_torques_y',
                                                shape=(num_times, ))
        external_torques_z = self.create_output('external_torques_z',
                                                shape=(num_times, ))

        # Integrate reaction wheel acceleration (function of torque)
        # over time to get reaction wheel speed;
        # Need reaction wheel speed to enforce constraint
        self.add_subsystem(
            'reaction_wheel_speed',
            RWSpeedRK4(
                num_times=num_times,
                step_size=step_size,
            ),
            promotes=['*'],
        )
        rw_speed = self.declare_input('rw_speed', shape=(3, num_times))

        # Prevent reaction wheels from saturating
        self.add_constraint('rw_speed', upper=250, lower=-250)

        # Define external torques to use in attitude integrator
        external_torques_x.define(-ot.expand(
            power_x,
            (3, num_times),
            'i->ji',
        ) / rw_speed)
        external_torques_y.define(-ot.expand(
            power_y,
            (3, num_times),
            'i->ji',
        ) / rw_speed)
        external_torques_z.define(-ot.expand(
            power_z,
            (3, num_times),
            'i->ji',
        ) / rw_speed)

        # Since reaction wheel speed depends on torque and vise versa,
        # we need an iterative solver to solve the nonlinear system;
        # Put solver selction at the end of model definition to avoid
        # distracting from the rest of the model definition
        self.nonlinear_solver = NonlinearBlockGS(iprint=0)

from lsdo_cubesat.solar.ivt import IVT
from lsdo_battery.battery_pack import BatteryPack
from csdl import Model
import csdl


class ElectricalPowerSystem(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('comm', types=bool, default=False)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        comm = self.parameters['comm']

        # solar power
        self.add(IVT(num_times=num_times), name='solar_panels')
        load_voltage = self.declare_variable(
            'load_voltage',
            shape=(1, num_times),
        )
        load_current = self.declare_variable(
            'load_current',
            shape=(1, num_times),
        )
        solar_power = load_current * load_voltage
        self.register_output('solar_power', solar_power)

        rw_power = self.declare_variable(
            'rw_power',
            shape=(3, num_times),
        )

        if comm is True:
            comm_power = self.declare_variable(
                'comm_power',
                shape=(num_times, ),
            )
            # TODO: sign?
            battery_power = csdl.reshape(
                csdl.sum(rw_power, axes=(0, )),
                (1, num_times)) + comm_power - solar_power,
        else:
            pass
            # TODO: sign?
            battery_power = csdl.reshape(csdl.sum(rw_power, axes=(0, )),
                                         (1, num_times)) - solar_power
        self.register_output('battery_power', battery_power)

        self.add(
            BatteryPack(num_times=num_times, step_size=step_size),
            name='battery_pack',
        )

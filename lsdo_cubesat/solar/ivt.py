from csdl import Model
import csdl

from lsdo_cubesat.constants import charge_of_electron, boltzman


class IVT(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('diode_voltage', default=-0.6, types=float)
        self.parameters.declare('shunt_resistance', default=40.0, types=float)
        self.parameters.declare('max_short_circuit_current',
                                default=0.453,
                                types=float)
        self.parameters.declare('saturation_current',
                                default=2.809e-12,
                                types=float)
        self.parameters.declare('diode_factor', default=1.35, types=float)

    def define(self):
        num_times = self.parameters['num_times']
        diode_voltage = self.parameters['diode_voltage']
        shunt_resistance = self.parameters['shunt_resistance']
        max_short_circuit_current = self.parameters[
            'max_short_circuit_current']
        saturation_current = self.parameters['saturation_current']
        diode_factor = self.parameters['diode_factor']

        LOS = self.declare_variable('LOS', shape=(num_times, ))
        illuminated_area = self.declare_variable('illuminated_area',
                                                 shape=(num_times, ))
        T = self.declare_variable('temperature', val=25.0, shape=(num_times, ))

        short_circuit_current = max_short_circuit_current * LOS * illuminated_area

        VT = diode_factor * boltzman * T / charge_of_electron

        load_current = self.declare_variable('load_current',
                                             val=saturation_current,
                                             shape=(num_times, ))
        load_voltage = self.declare_variable('load_voltage',
                                             val=0,
                                             shape=(num_times, ))
        r_I = load_current - (short_circuit_current - saturation_current *
                              (csdl.exp(load_voltage / VT) - 1) -
                              load_voltage / shunt_resistance)
        # load_voltage = csdl.tanh(-VT * shunt_resistance)
        Voc = 1.1 * VT * csdl.log(VT / saturation_current)
        s = Voc + diode_voltage
        d = Voc - diode_voltage
        dVdI = -VT / (VT +
                      saturation_current * shunt_resistance) * shunt_resistance
        b = 1 / d * dVdI
        r_V = load_voltage - (s) / 2 + (d) / 2 * csdl.tanh(
            b * (load_current - short_circuit_current) + csdl.artanh(s / d))

        solar_power = load_current * load_voltage
        self.register_output('r_I', r_I)
        self.register_output('r_V', r_V)
        self.register_output('solar_power', solar_power)


if __name__ == "__main__":

    class Example(Model):
        def initialize(self):
            self.parameters.declare('num_times', types=int)

        def define(self):
            num_times = self.parameters['num_times']
            ivt = self.create_implicit_operation(IVT(num_times=num_times))
            ivt.declare_state('load_current', residual='r_I')
            ivt.declare_state('load_voltage', residual='r_V')

            LOS = self.declare_variable('LOS', shape=(num_times, ), val=0)
            illuminated_area = self.declare_variable('illuminated_area',
                                                     shape=(num_times, ))
            load_current, load_voltage, solar_power = ivt(
                LOS,
                illuminated_area,
                expose=['solar_power'],
            )

    from csdl_om import Simulator
    sim = Simulator(Example(num_times=1))
    sim.visualize_implementation()
    sim.check_partials(compact_print=True)

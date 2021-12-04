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

        sun_LOS = self.declare_variable('sun_LOS', shape=(1, num_times))
        percent_exposed_area = self.declare_variable('percent_exposed_area',
                                                     shape=(1, num_times))
        T = self.declare_variable('temperature',
                                  val=25.0,
                                  shape=(1, num_times))

        short_circuit_current = max_short_circuit_current * sun_LOS * percent_exposed_area

        VT = diode_factor * boltzman * T / charge_of_electron

        load_voltage = self.declare_variable('load_voltage',
                                             val=0,
                                             shape=(1, num_times))

        load_current = self.create_input('load_current',
                                         val=saturation_current,
                                         shape=(1, num_times))
        self.add_design_variable('load_current',
                                 lower=0,
                                 upper=max_short_circuit_current)
        # r_I = load_current - (short_circuit_current - saturation_current *
        #   (csdl.exp(load_voltage / VT) - 1) -
        #   load_voltage / shunt_resistance)
        # self.register_output('r_I', r_I)

        # load_current = (short_circuit_current - saturation_current *
        #                       (csdl.exp(load_voltage / VT) - 1) -
        #                       load_voltage / shunt_resistance)
        # self.register_output('load_current', load_current)

        # load_voltage = csdl.tanh(-VT * shunt_resistance)
        Voc = 1.1 * VT * csdl.log(VT / saturation_current)
        s = Voc + diode_voltage
        d = Voc - diode_voltage
        dVdI = -VT / (VT +
                      saturation_current * shunt_resistance) * shunt_resistance
        b = 1 / d * dVdI
        load_voltage = s / 2 + d / 2 * csdl.tanh(
            b * (load_current - short_circuit_current) + csdl.artanh(s / d))

        self.register_output('load_voltage', load_voltage)


if __name__ == "__main__":

    # class Example(Model):
    #     def initialize(self):
    #         self.parameters.declare('num_times', types=int)

    #     def define(self):
    #         num_times = self.parameters['num_times']
    #         ivt = self.create_implicit_operation(IVT(num_times=num_times))
    #         ivt.declare_state('load_voltage', residual='r_V')

    #         load_current = self.declare_variable('load_current',
    #                                              shape=(num_times, ),
    #                                              val=0)
    #         sun_LOS = self.declare_variable('sun_LOS', shape=(num_times, ), val=0)
    #         illuminated_area = self.declare_variable('illuminated_area',
    #                                                  shape=(num_times, ))
    #         load_voltage = ivt(
    #             load_current,
    #             sun_LOS,
    #             illuminated_area,
    #         )
    #         solar_power = load_current * load_voltage
    #         self.register_output('solar_power', solar_power)

    from csdl_om import Simulator
    sim = Simulator(IVT(num_times=10))
    sim.visualize_implementation()
    # sim.check_partials(compact_print=True)

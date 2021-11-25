from csdl import Model
import csdl

from lsdo_cubesat.constants import charge_of_electron, boltzman


class IVT(Model):
    def initialize(self):
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
        shunt_resistance = self.parameters['shunt_resistance']
        max_short_circuit_current = self.parameters[
            'max_short_circuit_current']
        saturation_current = self.parameters['saturation_current']
        diode_factor = self.parameters['diode_factor']

        LOS = self.declare_variable('LOS')
        illuminated_area = self.declare_variable('illuminated_area')

        short_circuit_current = max_short_circuit_current * LOS * illuminated_area

        VT = diode_factor * boltzman * T / charge_of_electron

        load_current = self.declare_variable('load_current',
                                             val=saturation_current)
        load_voltage = self.declare_variable('load_voltage ', val=0)
        r_I = load_current - (short_circuit_current - saturation_current *
                              (csdl.exp(load_voltage / VT) - 1) -
                              load_voltage / shunt_resistance)
        load_voltage = csdl.tanh(-VT * shunt_resistance)

        solar_power = load_current * load_voltage

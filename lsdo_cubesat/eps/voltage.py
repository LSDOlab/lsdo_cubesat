import numpy as np
from csdl import Model
import csdl
from lsdo_cubesat.operations.soc_integrator import SOC_Integrator


class Voltage(Model):

    def initialize(self):
        self.parameters.declare('num_times', default=1, types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('temp_decay_coeff',
                                types=float,
                                default=np.log(1 / 1.1**5))
        self.parameters.declare('reference_temperature',
                                types=float,
                                default=293.)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        temp_decay_coeff = self.parameters['temp_decay_coeff']
        reference_temperature = self.parameters['reference_temperature']

        power = self.declare_variable('power',
                                      val=-10 *
                                      (np.random.rand(num_times).reshape(
                                          (1, num_times)) - 0.5),
                                      shape=(1, num_times))
        capacity = self.declare_variable(
            'capacity',
            shape=(1, num_times),
            val=2.9,
        )
        voltage = self.declare_variable('voltage',
                                        shape=(1, num_times),
                                        val=3.3)
        # temperature = self.declare_variable('temperature',
        #                                     shape=(1, num_times))
        coulombic_efficiency = self.declare_variable('coulombic_efficiency',
                                                     shape=(1, num_times),
                                                     val=1)

        initial_soc = self.declare_variable('initial_soc', val=0.95)
        soc = csdl.custom(
            power,
            voltage,
            coulombic_efficiency,
            capacity,
            initial_soc,
            op=SOC_Integrator(
                num_times=num_times,
                step_size=step_size,
            ),
        )
        self.register_output('soc', soc)

        # T = temperature + 273.15
        # self.register_output(
        #     'r_V', voltage - (3 + (csdl.exp(soc) - 1) / (np.exp(1) - 1)) *
        #     (2 - csdl.exp(temp_decay_coeff * (T - reference_temperature / T))))
        self.register_output(
            'r_V', voltage - (3 + (csdl.exp(soc) - 1) / (np.exp(1) - 1)))


if __name__ == '__main__':
    from csdl_om import Simulator
    import matplotlib.pyplot as plt

    num_times = 40
    step_size = 95 * 60 / (num_times - 1)
    time_scale = 1

    print('STEP SIZE', step_size)
    print('TIME SCALE', time_scale)
    print('STEP SIZE (fast)', step_size / float(time_scale))

    np.random.seed(0)

    # NOTE: make power negative to avoid overflow
    # NOTE: power can't be constant over time because it will cause overflow


    class Example(Model):

        def define(self):
            power = self.create_input(
                'power',
                val=-10 * (np.random.rand(num_times * time_scale).reshape(
                    (1, num_times * time_scale)) - 0.5),
                shape=(1, num_times * time_scale),
            )
            voltage = self.create_input('voltage',
                                        val=3.3,
                                        shape=(1, num_times * time_scale))

            initial_soc = self.create_input(
                'initial_soc',
                val=0.95,
            )
            self.add_design_variable('power')
            self.add_design_variable('voltage')
            self.add_design_variable('initial_soc')
            self.add(
                Voltage(num_times=num_times * time_scale,
                        step_size=step_size / float(time_scale)), )
            self.add_constraint('r_V', equals=0)

    sim = Simulator(Example())
    # sim = Simulator(
    #     Voltage(num_times=num_times * time_scale,
    #             step_size=step_size / float(time_scale)), )
    sim.run()
    sim.prob.check_totals(compact_print=True, method='fd')
    plt.plot(sim['soc'].flatten())
    plt.show()

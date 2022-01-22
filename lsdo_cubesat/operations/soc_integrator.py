import numpy as np

from lsdo_cubesat.operations.rk4_op import RK4


class SOC_Integrator(RK4):

    def initialize(self):
        super().initialize()
        self.parameters.declare('num_times', types=int)

        self.parameters['state_var'] = 'soc'
        self.parameters['init_state_var'] = 'initial_soc'
        self.parameters['external_vars'] = [
            'power',
            'voltage',
            'coulombic_efficiency',
            'capacity',
        ]

    def define(self):
        n = self.parameters['num_times']

        self.add_input(
            'power',
            shape=(1, n),
            val=1,
        )
        self.add_input(
            'voltage',
            shape=(1, n),
            val=1,
        )
        self.add_input(
            'coulombic_efficiency',
            shape=(1, n),
            val=1,
        )
        # Standard Capacity in Ah
        # http://www.datasheetcafe.com/icr18650-datasheet-battery-samsung/
        self.add_input(
            'capacity',
            shape=(1, n),
            val=2.6,
        )
        self.add_input(
            'initial_soc',
            shape=(1, ),
        )
        self.add_output(
            'soc',
            shape=(1, n),
        )
        self.dfdy = np.zeros(1)

    def f_dot(self, external, state):
        power = external[0]
        voltage = external[1]
        efficiency = external[2]
        capacity = external[3]

        dz = efficiency * power / (voltage * capacity)
        return dz

    def df_dy(self, external, state):
        return self.dfdy

    def df_dx(self, external, state):
        power = external[0]
        voltage = external[1]
        efficiency = external[2]
        capacity = external[3]

        dfdx = np.zeros((1, 4))
        vc = voltage * capacity
        dfdx[0, 0] = efficiency / vc
        dfdx[0, 1] = -efficiency * power / (voltage**2 * capacity)
        dfdx[0, 2] = power / vc
        dfdx[0, 3] = -efficiency * power / (voltage * capacity**2)
        return dfdx


if __name__ == '__main__':
    from csdl import Model
    import csdl

    # num_times = 200
    # step_size = 0.1
    num_times = 40 * 10
    step_size = 95 * 60 / (num_times - 1)
    np.random.seed(0)

    class Example(Model):

        def define(self):
            power = self.create_input(
                'power',
                shape=(1, num_times),
            )
            voltage = self.create_input(
                'voltage',
                shape=(1, num_times),
                val=3.3,
            )
            coulombic_efficiency = self.create_input(
                'coulombic_efficiency',
                shape=(1, num_times),
                val=1,
            )
            # Standard Capacity in Ah
            # http://www.datasheetcafe.com/icr18650-datasheet-battery-samsung/
            capacity = self.create_input(
                'capacity',
                shape=(1, num_times),
                val=2.6,
            )
            initial_soc = self.create_input(
                'initial_soc',
                shape=(1, ),
                val=0.5,
            )
            self.add_design_variable('power')
            self.add_design_variable('voltage')
            self.add_design_variable('coulombic_efficiency')
            self.add_design_variable('capacity')
            self.add_design_variable('initial_soc')
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
            self.add_constraint('soc')

    from csdl_om import Simulator
    sim = Simulator(Example())
    sim['power'] = 10 * (np.random.rand(num_times).reshape(
        (1, num_times)) - 0.5)
    sim['voltage'] = 3.3 * np.random.rand(num_times).reshape((1, num_times))
    sim['coulombic_efficiency'] = np.random.rand(num_times).reshape(
        (1, num_times))
    sim['capacity'] = 2.6 * np.random.rand(num_times).reshape((1, num_times))
    # sim.check_partials(compact_print=True, method='fd')
    sim.run()
    sim.prob.check_totals(compact_print=True, method='fd')

    # sim.run()
    # import matplotlib.pyplot as plt
    # t = np.arange(num_times) * step_size
    # plt.plot(t, sim['soc'].flatten())
    # plt.show()
    # # plt.plot(t, sim['power'].flatten())
    # plt.plot(sim['power'].flatten(), sim['soc'].flatten())
    # plt.show()

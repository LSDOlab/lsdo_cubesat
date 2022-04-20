import numpy as np
from csdl import Model, NewtonSolver, ScipyKrylov, NonlinearBlockGS
import csdl
from lsdo_cubesat.eps.voltage import Voltage


class Cell(Model):

    def initialize(self):
        self.parameters.declare('num_times', default=1, types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('min_soc', default=0.05, types=float)
        self.parameters.declare('max_soc', default=0.95, types=float)
        self.parameters.declare('periodic_soc', default=False, types=bool)
        self.parameters.declare('SAND_MDF', values=('SAND', 'MDF'))
        self.parameters.declare('time_scale', types=int, default=1)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        min_soc = self.parameters['min_soc']
        max_soc = self.parameters['max_soc']
        periodic_soc = self.parameters['periodic_soc']
        SAND_MDF = self.parameters['SAND_MDF']
        time_scale = self.parameters['time_scale']

        initial_soc = self.create_input(
            'initial_soc',
            shape=(1, ),
            val=0.95,
        )
        self.add_design_variable(
            'initial_soc',
            lower=min_soc,
            upper=max_soc,
        )
        power = self.declare_variable(
            'power',
            val=-10 * (np.random.rand(num_times * time_scale).reshape(
                (1, num_times * time_scale)) - 0.5),
            shape=(1, num_times * time_scale),
        )
        # if SAND_MDF == 'MDF':
        compute_voltage = self.create_implicit_operation(
            Voltage(
                num_times=num_times * time_scale,
                step_size=step_size / float(time_scale),
            ))
        compute_voltage.declare_state('voltage', residual='r_V', val=3.3)
        compute_voltage.linear_solver = ScipyKrylov()
        compute_voltage.nonlinear_solver = NonlinearBlockGS(iprint=0)
        # compute_voltage.nonlinear_solver = NewtonSolver(
        #     solve_subsystems=False,
        #     iprint=0,
        # )
        voltage, soc = compute_voltage(power, initial_soc, expose=['soc'])

        # elif SAND_MDF == 'SAND':
        #     voltage = self.create_input('voltage',
        #                                 val=3.3,
        #                                 shape=(1, num_times * time_scale))
        #     self.add_design_variable('voltage')
        #     self.add(
        #         Voltage(
        #             num_times=num_times * time_scale,
        #             step_size=step_size / float(time_scale),
        #         ))
        #     # self.add_constraint('r_V', equals=0)
        #     soc = self.declare_variable('soc',
        #                                 val=2,
        #                                 shape=(1, num_times * time_scale))
        soc_sliced = soc[0, ::time_scale]

        # enforce soc constraint
        mmin_soc = self.register_output(
            'mmin_soc',
            csdl.exp(soc_sliced),
        )
        self.register_output(
            'min_soc',
            csdl.min(mmin_soc, axis=1, rho=20. / 1e0),
            # csdl.min(soc, axis=1, rho=10.),
        )
        # self.register_output(
        #     'max_soc',
        #     csdl.max(soc_sliced, axis=1, rho=20.),
        # )
        # self.add_constraint(
        #     'min_soc',
        #     lower=min_soc,
        # )
        # self.add_constraint(
        #     'max_soc',
        #     upper=max_soc,
        # )
        if periodic_soc is True:
            delta_soc = soc[0, -1] - soc[0, 0]
            self.register_output('delta_soc', delta_soc)
            self.add_constraint('delta_soc', equals=0)


if __name__ == '__main__':
    from csdl_om import Simulator

    num_times = 40
    step_size = 95 * 60 / (num_times - 1)
    time_scale = 1

    print('STEP SIZE', step_size)
    print('TIME SCALE', time_scale)
    print('STEP SIZE (fast)', step_size / float(time_scale))

    np.random.seed(0)
    v = -10 * (np.random.rand(num_times * time_scale).reshape(
        (1, num_times * time_scale)) - 0.5)

    def a():
        sim = Simulator(
            Cell(
                num_times=num_times,
                step_size=step_size,
                periodic_soc=False,
                SAND_MDF='SAND',
                time_scale=time_scale,
            ))

        # sim['power'] = v
        sim.run()
        import matplotlib.pyplot as plt

        plt.plot(sim['soc'].flatten())
        plt.show()

        # sim.prob.check_totals(compact_print=True)

    def b():
        sim = Simulator(
            Cell(
                num_times=num_times,
                step_size=step_size,
                periodic_soc=False,
                SAND_MDF='MDF',
                time_scale=time_scale,
            ))

        sim['power'] = v
        sim.run()
        sim.visualize_implementation()
        exit()
        sim.prob.check_totals(compact_print=True)

        # import matplotlib.pyplot as plt

        # plt.plot(sim['soc'].flatten())
        # plt.show()

    # a()
    b()

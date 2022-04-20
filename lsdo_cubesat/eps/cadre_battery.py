import numpy as np
from csdl import Model, NonlinearBlockGS, NewtonSolver
import csdl
from lsdo_cubesat.operations.soc_integrator import SOC_Integrator
from csdl.solvers.linear.direct import DirectSolver
from csdl.solvers.linear.linear_block_gs import LinearBlockGS
from csdl.solvers.linear.linear_block_jac import LinearBlockJac
from csdl.solvers.linear.linear_runonce import LinearRunOnce
from csdl.solvers.linear.petsc_ksp import PETScKrylov
from csdl.solvers.linear.scipy_iter_solver import ScipyKrylov
# from csdl.solvers.linear.user_defined import LinearUserDefined


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

        power = self.declare_variable('power', shape=(1, num_times), val=-1)
        capacity = self.declare_variable(
            'capacity',
            shape=(1, num_times),
            val=2.9,
        )
        voltage = self.declare_variable('voltage', shape=(1, num_times))
        # temperature = self.declare_variable('temperature',
        #                                     shape=(1, num_times))
        coulombic_efficiency = self.declare_variable('coulombic_efficiency',
                                                     shape=(1, num_times),
                                                     val=1)

        initial_soc = self.declare_variable('initial_soc')
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
        self.add_constraint('r_V', equals=0)


class Cell(Model):

    def initialize(self):
        self.parameters.declare('num_times', default=1, types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('min_soc', default=0.05, types=float)
        self.parameters.declare('max_soc', default=0.95, types=float)
        self.parameters.declare('periodic_soc', default=False, types=bool)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        min_soc = self.parameters['min_soc']
        max_soc = self.parameters['max_soc']
        periodic_soc = self.parameters['periodic_soc']

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
        ## Use nonlinear solver to solve for voltage
        compute_voltage = self.create_implicit_operation(
            Voltage(
                num_times=num_times,
                step_size=step_size,
            ))
        compute_voltage.declare_state('voltage', residual='r_V', val=3.3)
        compute_voltage.linear_solver = ScipyKrylov()
        compute_voltage.nonlinear_solver = NewtonSolver(
            solve_subsystems=True,
            iprint=0,
        )

        power = self.declare_variable('power', shape=(1, num_times))
        voltage, soc = compute_voltage(power, initial_soc, expose=['soc'])

        ## Use optimizer to solve for voltage
        # self.create_input('voltage', val=3.3, shape=(1, num_times))
        # self.add_design_variable('voltage')
        # self.add(Voltage(
        #     num_times=num_times,
        #     step_size=step_size,
        # ))
        # soc = self.declare_variable('soc', shape=(1, num_times))

        # enforce soc constraint
        mmin_soc = self.register_output(
            'mmin_soc',
            csdl.exp(soc),
        )
        self.register_output(
            'min_soc',
            csdl.min(mmin_soc, axis=1, rho=20. / 1e0),
            # csdl.min(soc, axis=1, rho=10.),
        )
        self.register_output(
            'max_soc',
            csdl.max(soc, axis=1, rho=20.),
        )
        self.add_constraint(
            'min_soc',
            lower=np.exp(min_soc),
        )
        self.add_constraint(
            'max_soc',
            upper=max_soc,
        )
        if periodic_soc is True:
            delta_soc = soc[0, -1] - soc[0, 0]
            self.register_output('delta_soc', delta_soc)
            self.add_constraint('delta_soc', equals=0)


class BatteryPack(Model):

    def initialize(self):
        self.parameters.declare('num_times', default=1, types=int)
        self.parameters.declare('step_size', types=float)

        # 18650
        # http://www.datasheetcafe.com/icr18650-datasheet-battery-samsung/
        dia_mm = 18.4
        rad_mm = dia_mm / 2
        rad_m = rad_mm / 1000
        h_mm = 65
        h_m = h_mm / 1000
        cell_volume = np.pi * rad_m**2 * h_m
        cell_mass = 47 / 1000
        cell_density = cell_mass / cell_volume

        self.parameters.declare('cell_density',
                                default=cell_density,
                                types=float)
        self.parameters.declare('cell_mass', default=cell_mass, types=float)
        self.parameters.declare('cell_volume',
                                default=cell_volume,
                                types=float)
        self.parameters.declare('max_discharge_rate',
                                default=-10.,
                                types=float)
        self.parameters.declare('max_charge_rate', default=5., types=float)

        self.parameters.declare('optimize_plant', default=True, types=bool)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        cell_density = self.parameters['cell_density']
        cell_mass = self.parameters['cell_mass']
        cell_volume = self.parameters['cell_volume']
        max_discharge_rate = self.parameters['max_discharge_rate']
        max_charge_rate = self.parameters['max_charge_rate']

        optimize_plant = self.parameters['optimize_plant']

        # total power demanded by other subsystems
        battery_power = self.declare_variable(
            'battery_power',
            shape=(1, num_times),
        )

        # NOTE: Set default values high to avoid division by zero
        # when computing cell power during optimization
        num_series = self.create_input(
            'num_series',
            val=2,
            shape=(1, ),
        )
        num_parallel = self.create_input(
            'num_parallel',
            val=3,
            shape=(1, ),
        )
        if optimize_plant is True:
            self.add_design_variable('num_series', lower=1)
            self.add_design_variable('num_parallel', lower=1)

        num_cells = num_series * num_parallel
        battery_mass = cell_mass * num_cells
        self.register_output('battery_mass', battery_mass)
        battery_volume = cell_volume * num_cells
        self.register_output('battery_volume', battery_volume)
        self.add_constraint('battery_volume')
        num_series_exp = csdl.expand(num_series, (1, num_times))
        num_parallel_exp = csdl.expand(num_parallel, (1, num_times))

        # NOTE: Only divide by num_parallel to use battery capacity in
        # SOC; not part of computing cell power for general models
        power = self.register_output(
            'power',
            battery_power / (num_parallel_exp * num_series_exp),
        )
        self.add(
            Cell(
                num_times=num_times,
                step_size=step_size,
                periodic_soc=False,
            ),
            name='battery_cell',
        )
        voltage = self.declare_variable('voltage', shape=(1, num_times))

        # enforce charge/discharge constraint
        current = power / voltage
        self.register_output('current', current)
        self.register_output('min_current', csdl.min(current, rho=10.))
        self.register_output('max_current', csdl.max(current, rho=10.))
        # self.add_constraint('min_current', lower=max_discharge_rate)
        # self.add_constraint('max_current', upper=max_charge_rate)


if __name__ == '__main__':
    from csdl_om import Simulator

    num_times = 40
    step_size = 95 * 60 / (num_times - 1)

    np.random.seed(0)
    v = -10 * (np.random.rand(num_times).reshape((1, num_times)) - 0.5)

    sim = Simulator(BatteryPack(num_times=num_times, step_size=step_size))
    sim['battery_power'] = v * 6
    sim['power'] = v
    sim.run()
    sim.prob.check_totals(compact_print=True)
    sim.visualize_implementation()

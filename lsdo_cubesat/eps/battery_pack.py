import numpy as np
from csdl import Model
import csdl
from lsdo_cubesat.eps.cell import Cell


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
        self.parameters.declare('time_scale', types=int, default=1)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        cell_density = self.parameters['cell_density']
        cell_mass = self.parameters['cell_mass']
        cell_volume = self.parameters['cell_volume']
        max_discharge_rate = self.parameters['max_discharge_rate']
        max_charge_rate = self.parameters['max_charge_rate']

        optimize_plant = self.parameters['optimize_plant']
        time_scale = self.parameters['time_scale']

        # total power demanded by other subsystems
        battery_power = self.declare_variable(
            'battery_power',
            val=-6 * 10 * (np.random.rand(num_times * time_scale).reshape(
                (1, num_times * time_scale)) - 0.5),
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
        num_series_exp = csdl.expand(num_series, (1, num_times))
        num_parallel_exp = csdl.expand(num_parallel, (1, num_times))

        # NOTE: Only divide by num_parallel to use battery capacity in
        # SOC; not part of computing cell power for general models
        from csdl.utils.get_bspline_mtx import get_bspline_mtx
        power = self.register_output(
            'power',
            csdl.reshape(
                csdl.matvec(
                    get_bspline_mtx(num_times, num_times * time_scale),
                    csdl.reshape(
                        battery_power / (num_parallel_exp * num_series_exp),
                        (num_times, )),
                ), (1, num_times * time_scale)),
        )
        self.add(
            Cell(
                num_times=num_times,
                step_size=step_size,
                periodic_soc=False,
                SAND_MDF='SAND',
                time_scale=time_scale,
            ),
            name='battery_cell',
        )
        voltage = self.declare_variable('voltage',
                                        shape=(1, num_times * time_scale))

        # enforce charge/discharge constraint
        current = power[0, ::time_scale] / voltage[0, ::time_scale]
        self.register_output('current', current)
        # self.register_output('min_current', csdl.min(current, rho=10.))
        # self.register_output('max_current', csdl.max(current, rho=10.))
        # self.add_constraint('min_current', lower=max_discharge_rate)
        # self.add_constraint('max_current', upper=max_charge_rate)

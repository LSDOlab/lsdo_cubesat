from math import ceil

import numpy as np
from openmdao.api import (ExecComp, Group, IndepVarComp, LinearBlockGS,
                          NonlinearBlockGS)

from lsdo_battery.battery_model import BatteryModel
from lsdo_cubesat.aerodynamics.aerodynamics_group import AerodynamicsGroup
from lsdo_cubesat.attitude.attitude_group import AttitudeGroup
from lsdo_cubesat.attitude.new.attitude_group import \
    AttitudeGroup as NewAttitudeGroup
from lsdo_cubesat.communication.comm_group import CommGroup
# from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.orbit.orbit_angular_speed_group import OrbitAngularSpeedGroup
from lsdo_cubesat.orbit.orbit_group import OrbitGroup
from lsdo_cubesat.propulsion.propulsion_group import PropulsionGroup
from lsdo_cubesat.solar.solar_exposure import SolarExposure
from lsdo_cubesat.utils.ks_comp import KSComp
from lsdo_cubesat.utils.slice_comp import SliceComp
from lsdo_utils.api import (ArrayExpansionComp, BsplineComp,
                            LinearCombinationComp, PowerCombinationComp,
                            get_bspline_mtx, ScalarExpansionComp)
from lsdo_utils.comps.arithmetic_comps.elementwise_max_comp import \
    ElementwiseMaxComp


class CubesatGroup(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('cubesat')
        self.options.declare('mtx')
        self.options.declare('Ground_station')
        self.options.declare('add_battery', types=bool)
        self.options.declare('sm')
        self.options.declare('optimize_plant', types=bool)
        self.options.declare('new_attitude', types=bool)
        self.options.declare('fast_time_scale', types=float)
        self.options.declare('battery_time_scale', types=float)
        self.options.declare('attitude_time_scale', types=float)

    def setup(self):
        num_times = self.options['num_times']
        num_cp = self.options['num_cp']
        step_size = self.options['step_size']
        cubesat = self.options['cubesat']
        mtx = self.options['mtx']
        Ground_station = self.options['Ground_station']
        add_battery = self.options['add_battery']
        sm = self.options['sm']
        optimize_plant = self.options['optimize_plant']
        new_attitude = self.options['new_attitude']
        battery_time_scale = self.options['battery_time_scale']
        attitude_time_scale = self.options['attitude_time_scale']

        comp = IndepVarComp()
        comp.add_output('Initial_Data', val=np.zeros((1, )))
        self.add_subsystem('inputs_comp', comp, promotes=['*'])

        if new_attitude:
            step = max(1, ceil(step_size / attitude_time_scale))
            group = NewAttitudeGroup(
                num_times=num_times * step,
                num_cp=num_cp,
                cubesat=cubesat,
                mtx=mtx,
                step_size=attitude_time_scale,
            )
            self.add_subsystem('attitude_group', group, promotes=['*'])
            group = SliceComp(
                shape=(3, 3, num_times * step),
                step=step,
                slice_axis=2,
                in_name='rot_mtx_i_b_3x3xn_fast',
                out_name='rot_mtx_i_b_3x3xn',
            )
            self.add_subsystem('rot_mtx_slow_ts_comp', group, promotes=['*'])
        else:
            group = AttitudeGroup(
                num_times=num_times,
                num_cp=num_cp,
                cubesat=cubesat,
                mtx=mtx,
                step_size=step_size,
            )
            self.add_subsystem('attitude_group', group, promotes=['*'])

        if add_battery:
            comp = SolarExposure(num_times=num_times, sm=sm)
            self.add_subsystem('solar_exposure', comp, promotes=['*'])

            # From BCT
            # https://storage.googleapis.com/blue-canyon-tech-news/1/2020/06/BCT_DataSheet_Components_PowerSystems_06_2020.pdf
            # Solar panel area (3U): 0.12 m2
            # Power: 28-42W, choose 35W
            # Efficiency: 29.5%
            # 100% efficiency: 291.67 W/m2
            # 29.5% efficiency: 86.04 W/m2
            power_over_area = 86.04

            self.add_subsystem(
                'compute_solar_power',
                PowerCombinationComp(
                    shape=(num_times, ),
                    out_name='solar_power',
                    coeff=power_over_area,
                    powers_dict=dict(sunlit_area=1.0, ),
                ),
                promotes=['*'],
            )

            # comp = SolarPanelVoltage(num_times=num_times)
            # self.add_subsystem('solar_panel_voltage', comp, promotes=['*'])

        group = PropulsionGroup(
            num_times=num_times,
            num_cp=num_cp,
            step_size=step_size,
            cubesat=cubesat,
            mtx=mtx,
        )
        self.add_subsystem('propulsion_group', group, promotes=['*'])

        group = AerodynamicsGroup(
            num_times=num_times,
            num_cp=num_cp,
            step_size=step_size,
            cubesat=cubesat,
            mtx=mtx,
        )
        self.add_subsystem('aerodynamics_group', group, promotes=['*'])

        orbit_avionics = Group()

        group = OrbitGroup(
            num_times=num_times,
            num_cp=num_cp,
            step_size=step_size,
            cubesat=cubesat,
            mtx=mtx,
        )
        orbit_avionics.add_subsystem('orbit_group', group, promotes=['*'])

        # if new_attitude:
        # compute osculating orbit angular speed to feed into
        # attitude model
        # self.add_subsystem(
        #     'orbit_angular_speed_group',
        #     OrbitAngularSpeedGroup(num_times=num_times, ),
        #     promotes=['*'],
        # )

        for Ground_station in cubesat.children:
            name = Ground_station['name']

            group = CommGroup(
                num_times=num_times,
                num_cp=num_cp,
                step_size=step_size,
                Ground_station=Ground_station,
                mtx=mtx,
            )

            # self.connect('times', '{}_comm_group.times'.format(name))

            orbit_avionics.add_subsystem('{}_comm_group'.format(name), group)

        # name = cubesat['name']
        shape = (1, num_times)
        rho = 100.

        # cubesat_name = cubesat['name']

        comp = ElementwiseMaxComp(shape=shape,
                                  in_names=[
                                      'UCSD_comm_group_Download_rate',
                                      'UIUC_comm_group_Download_rate',
                                      'Georgia_comm_group_Download_rate',
                                      'Montana_comm_group_Download_rate',
                                  ],
                                  out_name='KS_Download_rate',
                                  rho=rho)
        orbit_avionics.add_subsystem('KS_Download_rate_comp',
                                     comp,
                                     promotes=['*'])

        if add_battery:
            comp = ElementwiseMaxComp(
                shape=shape,
                in_names=[
                    'UCSD_comm_group_P_comm',
                    'UIUC_comm_group_P_comm',
                    'Georgia_comm_group_P_comm',
                    'Montana_comm_group_P_comm',
                ],
                out_name='KS_P_comm',
                rho=rho,
            )
            orbit_avionics.add_subsystem('KS_P_comm_comp',
                                         comp,
                                         promotes=['*'])

        for Ground_station in cubesat.children:
            Ground_station_name = Ground_station['name']

            orbit_avionics.connect(
                '{}_comm_group.Download_rate'.format(Ground_station_name),
                '{}_comm_group_Download_rate'.format(Ground_station_name),
            )

            if add_battery:

                orbit_avionics.connect(
                    '{}_comm_group.P_comm'.format(Ground_station_name),
                    '{}_comm_group_P_comm'.format(Ground_station_name),
                )

        if add_battery:

            baseline_power = 6.3
            orbit_avionics.add_subsystem(
                'sum_power',
                LinearCombinationComp(
                    shape=(num_times, ),
                    in_names=[
                        'solar_power',
                        'KS_P_comm',
                    ],
                    out_name='battery_output_power_slow',
                    coeffs=[1, -1],
                    constant=-baseline_power,
                ),
                promotes=['*'],
            )
            comp = KSComp(
                in_name='battery_output_power_slow',
                out_name='min_battery_output_power_slow',
                shape=(1, ),
                constraint_size=num_times,
                lower_flag=True,
                rho=100.,
            )

            step = max(1, ceil(step_size / battery_time_scale))

            comp = BsplineComp(
                num_pt=num_times * step,
                num_cp=num_times,
                jac=get_bspline_mtx(num_times, num_times * step),
                in_name='battery_output_power_slow',
                out_name='battery_output_power',
            )
            orbit_avionics.add_subsystem(
                'power_spline',
                comp,
                promotes=['*'],
            )

            orbit_avionics.add_subsystem(
                'battery',
                BatteryModel(
                    num_times=num_times * step,
                    min_soc=0.05,
                    max_soc=0.95,
                    # periodic_soc=True,
                    optimize_plant=optimize_plant,
                    step_size=battery_time_scale,
                ),
                promotes=['*'],
            )

            orbit_avionics.add_subsystem(
                'expand_battery_mass',
                ScalarExpansionComp(
                    shape=(num_times, ),
                    in_name='battery_mass',
                    out_name='battery_mass_exp',
                ),
                promotes=['*'],
            )

            orbit_avionics.nonlinear_solver = NonlinearBlockGS(
                iprint=0,
                maxiter=40,
                atol=1e-14,
                rtol=1e-12,
            )
            orbit_avionics.linear_solver = LinearBlockGS(
                iprint=0,
                maxiter=40,
                atol=1e-14,
                rtol=1e-12,
            )

        self.add_subsystem('orbit_avionics', orbit_avionics, promotes=['*'])

        if add_battery:

            self.add_subsystem(
                'compute_battery_and_propellant_volume',
                LinearCombinationComp(
                    shape=(1, ),
                    in_names=[
                        'total_propellant_volume',
                        'battery_volume',
                    ],
                    out_name='battery_and_propellant_volume',
                    constant=0,
                ),
                promotes=['*'],
            )

            self.add_subsystem(
                'compute_battery_and_propellant_mass',
                LinearCombinationComp(
                    shape=(1, ),
                    in_names=[
                        'total_propellant_used',
                        'battery_mass',
                    ],
                    out_name='battery_and_propellant_mass',
                    constant=0,
                ),
                promotes=['*'],
            )

        self.add_constraint(
            'battery_and_propellant_mass',
            lower=0,
            # Don't bother with an upper limit because we are indirectly
            # minimizing mass, and we don't want to make the problem
            # infeasible
            # upper=1.33,
        )

        # 1U (10cm)**3
        u = 10**3 / 100**3
        self.add_constraint(
            'battery_and_propellant_volume',
            lower=0,
            upper=u,
        )

        comp = DataDownloadComp(
            num_times=num_times,
            step_size=step_size,
        )
        self.add_subsystem('Data_download_rk4_comp', comp, promotes=['*'])

        comp = ExecComp(
            'total_Data = Data[-1] - Data[0]',
            Data=np.empty(num_times),
        )
        self.add_subsystem('KS_total_Data_comp', comp, promotes=['*'])

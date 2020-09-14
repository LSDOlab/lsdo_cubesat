import numpy as np

from openmdao.api import Group, IndepVarComp, ExecComp

# from lsdo_cubesat.attitude.attitude_group import AttitudeGroup
from lsdo_cubesat.attitude.old.attitude_group import AttitudeGroup
from lsdo_cubesat.propulsion.propulsion_group import PropulsionGroup
from lsdo_cubesat.aerodynamics.aerodynamics_group import AerodynamicsGroup
from lsdo_cubesat.orbit.orbit_group import OrbitGroup
from lsdo_cubesat.communication.comm_group import CommGroup
# from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp

from lsdo_utils.comps.arithmetic_comps.elementwise_max_comp import ElementwiseMaxComp
from lsdo_cubesat.solar.solar_exposure import SolarExposure
from lsdo_cubesat.solar.solar_panel_voltage import SolarPanelVoltage
from smt.surrogate_models import RMTB
from lsdo_battery.battery_model import BatteryModel
from lsdo_utils.api import LinearCombinationComp, PowerCombinationComp
from lsdo_cubesat.orbit.mean_motion_group import MeanMotionGroup


class CubesatGroup(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('cubesat')
        self.options.declare('mtx')
        self.options.declare('Ground_station')
        self.options.declare('sm', types=RMTB)

    def setup(self):
        num_times = self.options['num_times']
        num_cp = self.options['num_cp']
        step_size = self.options['step_size']
        cubesat = self.options['cubesat']
        mtx = self.options['mtx']
        Ground_station = self.options['Ground_station']
        sm = self.options['sm']

        times = np.linspace(0., step_size * (num_times - 1), num_times)

        comp = IndepVarComp()

        comp.add_output('times', units='s', val=times)
        comp.add_output('Initial_Data', val=np.zeros((1, )))
        self.add_subsystem('inputs_comp', comp, promotes=['*'])

        group = AttitudeGroup(
            num_times=num_times,
            step_size=step_size,
            num_cp=num_cp,
            cubesat=cubesat,
            mtx=mtx,
        )
        self.add_subsystem('attitude_group', group, promotes=['*'])

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

        group = OrbitGroup(
            num_times=num_times,
            num_cp=num_cp,
            step_size=step_size,
            cubesat=cubesat,
            mtx=mtx,
        )
        self.add_subsystem('orbit_group', group, promotes=['*'])

        # self.add_subsystem('mean_motion_group',
        #                    MeanMotionGroup(num_times=num_times, ),
        #                    promotes=['*'])

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

            self.add_subsystem('{}_comm_group'.format(name), group)

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
        self.add_subsystem('KS_Download_rate_comp', comp, promotes=['*'])

        in_names = []
        for Ground_station in cubesat.children:
            Ground_station_name = Ground_station['name']

            self.connect(
                '{}_comm_group.Download_rate'.format(Ground_station_name),
                '{}_comm_group_Download_rate'.format(Ground_station_name),
            )

            in_names.append('{}_comm_group_P_comm'.format(Ground_station_name))

        self.add_subsystem(
            'sum_P_Comm',
            LinearCombinationComp(
                shape=(num_times, ),
                in_names=in_names,
                out_name='sum_P_comm',
                # coeffs=[cell_density, cell_density],
                constant=0,
            ),
            promotes=['*'],
        )

        for Ground_station in cubesat.children:
            Ground_station_name = Ground_station['name']
            self.connect(
                '{}_comm_group.P_comm'.format(Ground_station_name),
                '{}_comm_group_P_comm'.format(Ground_station_name),
            )

            # self.connect(
            #     '{}_comm_group.Download_rate'.format(Ground_station_name),
            #     '{}_comm_group_Download_rate'.format(Ground_station_name),
            # )

        baseline_power = 6.3
        self.add_subsystem(
            'sum_power',
            LinearCombinationComp(
                shape=(num_times, ),
                in_names=[
                    'solar_power',
                    'sum_P_comm',
                ],
                out_name='battery_output_power',
                coeffs=[-1, 1],
                constant=baseline_power,
            ),
            promotes=['*'],
        )
        self.add_constraint('battery_output_power', lower=baseline_power)

        self.add_subsystem(
            'battery',
            BatteryModel(num_times=num_times),
            promotes=['*'],
        )

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
            upper=1.33,
        )

        # 1U (m**3)
        u = 10**3 / 100**3
        self.add_constraint(
            'battery_and_propellant_volume',
            lower=0,
            upper=0.5 * u,
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

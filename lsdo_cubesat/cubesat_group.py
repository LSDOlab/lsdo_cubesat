from math import ceil

import numpy as np
from openmdao.api import (ExecComp, IndepVarComp, LinearBlockGS,
                          NonlinearBlockGS)
import omtools.api as ot
from omtools.api import Group
from lsdo_battery.battery_pack import BatteryPack
from lsdo_cubesat.aerodynamics.aerodynamics_group import AerodynamicsGroup
from lsdo_cubesat.attitude.attitude_group import AttitudeGroup
from lsdo_cubesat.attitude.attitude_ode_group import AttitudeOdeGroup
from lsdo_cubesat.communication.comm_group import CommGroup
from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.orbit.orbit_angular_speed_group import OrbitAngularSpeedGroup
from lsdo_cubesat.orbit.orbit_group import OrbitGroup
from lsdo_cubesat.propulsion.propulsion_group import PropulsionGroup
from lsdo_cubesat.solar.solar_illumination_comp import SolarIlluminationComp
from lsdo_cubesat.utils.api import (ArrayExpansionComp, BsplineComp,
                                    LinearCombinationComp,
                                    PowerCombinationComp, ScalarExpansionComp,
                                    get_bspline_mtx)
from lsdo_cubesat.utils.comps.arithmetic_comps.elementwise_max_comp import \
    ElementwiseMaxComp
from lsdo_cubesat.utils.ks_comp import KSComp
from lsdo_cubesat.utils.slice_comp import SliceComp


class CubesatGroup(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('cubesat')
        self.options.declare('mtx')
        self.options.declare('ground_station')
        self.options.declare('add_battery', types=bool)
        self.options.declare('sm')
        self.options.declare('optimize_plant', types=bool)
        self.options.declare('attitude_integrator', types=bool)
        self.options.declare('fast_time_scale', types=float)
        self.options.declare('battery_time_scale', types=float)
        self.options.declare('attitude_time_scale', types=float)

    def setup(self):
        num_times = self.options['num_times']
        num_cp = self.options['num_cp']
        step_size = self.options['step_size']
        cubesat = self.options['cubesat']
        mtx = self.options['mtx']
        ground_station = self.options['ground_station']
        add_battery = self.options['add_battery']
        sm = self.options['sm']
        optimize_plant = self.options['optimize_plant']
        attitude_integrator = self.options['attitude_integrator']
        battery_time_scale = self.options['battery_time_scale']
        attitude_time_scale = self.options['attitude_time_scale']

        comp = IndepVarComp()
        comp.add_output('Initial_Data', val=np.zeros((1, )))
        self.add_subsystem('inputs_comp', comp, promotes=['*'])

        # if attitude_integrator:
        #     step = max(1, ceil(step_size / attitude_time_scale))
        #     group = AttitudeOdeGroup(
        #         num_times=num_times * step,
        #         num_cp=num_cp,
        #         cubesat=cubesat,
        #         mtx=mtx,
        #         step_size=attitude_time_scale,
        #     )
        #     self.add_subsystem('attitude_group', group, promotes=['*'])
        #     group = SliceComp(
        #         shape=(3, 3, num_times * step),
        #         step=step,
        #         slice_axis=2,
        #         in_name='rot_mtx_i_b_3x3xn_fast',
        #         out_name='rot_mtx_i_b_3x3xn',
        #     )
        #     self.add_subsystem('rot_mtx_slow_ts_comp', group, promotes=['*'])
        # else:
        self.add_subsystem(
            'attitude_group',
            AttitudeGroup(
                num_times=num_times,
                num_cp=num_cp,
                cubesat=cubesat,
                mtx=mtx,
                step_size=step_size,
            ),
            promotes=['*'],
        )

        if add_battery:
            # From BCT
            # https://storage.googleapis.com/blue-canyon-tech-news/1/2020/06/BCT_DataSheet_Components_PowerSystems_06_2020.pdf
            # Solar panel area (3U): 0.12 m2
            # Power: 28-42W, choose 35W
            # Efficiency: 29.5%
            # 100% efficiency: 291.67 W/m2
            # 29.5% efficiency: 86.04 W/m2
            power_over_area = (35 / 0.12 * 0.295)  # 86.04

            # NOTE: CANNOT transform SMT model to om.Group!
            comp = SolarIlluminationComp(num_times=num_times, sm=sm)
            self.add_subsystem('solar_illumination', comp, promotes=['*'])
            sunlit_area = self.declare_input(
                'sunlit_area',
                shape=(num_times, ),
            )
            self.register_output('solar_power', power_over_area * sunlit_area)

            # comp = SolarPanelVoltage(num_times=num_times)
            # self.add_subsystem('solar_panel_voltage', comp, promotes=['*'])

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
        total_propellant_volume = self.declare_input('total_propellant_volume')
        total_propellant_used = self.declare_input('total_propellant_used')

        group = AerodynamicsGroup(
            num_times=num_times,
            num_cp=num_cp,
            step_size=step_size,
            cubesat=cubesat,
            mtx=mtx,
        )
        self.add_subsystem('aerodynamics_group', group, promotes=['*'])

        ones = self.create_indep_var(
            'ones',
            shape=(3, ),
            val=1,
        )
        with self.create_group('orbit_avionics') as orbit_avionics:
            orbit_avionics.add_subsystem(
                'orbit_group',
                OrbitGroup(
                    num_times=num_times,
                    num_cp=num_cp,
                    step_size=step_size,
                    cubesat=cubesat,
                    mtx=mtx,
                ),
                promotes=['*'],
            )

            # if attitude_integrator:
            # compute osculating orbit angular speed to feed into
            # attitude model
            # self.add_subsystem(
            #     'orbit_angular_speed_group',
            #     OrbitAngularSpeedGroup(num_times=num_times, ),
            #     promotes=['*'],
            # )

            # self.add_subsystem('mean_motion_group',
            #                    MeanMotionGroup(num_times=num_times, ),
            #                    promotes=['*'])

            for ground_station in cubesat.children:
                # self.connect('times', '{}_comm_group.times'.format(name))
                orbit_avionics.add_subsystem(
                    '{}_comm_group'.format(ground_station['name']),
                    CommGroup(
                        num_times=num_times,
                        num_cp=num_cp,
                        step_size=step_size,
                        ground_station=ground_station,
                        mtx=mtx,
                    ),
                )

            shape = (1, num_times)
            UCSD_comm_group_Download_rate = orbit_avionics.declare_input(
                'UCSD_comm_group_Download_rate',
                shape=shape,
            )
            UIUC_comm_group_Download_rate = orbit_avionics.declare_input(
                'UIUC_comm_group_Download_rate',
                shape=shape,
            )
            Georgia_comm_group_Download_rate = orbit_avionics.declare_input(
                'Georgia_comm_group_Download_rate',
                shape=shape,
            )
            Montana_comm_group_Download_rate = orbit_avionics.declare_input(
                'Montana_comm_group_Download_rate',
                shape=shape,
            )
            rho = 100.
            # orbit_avionics.register_output(
            #     'KS_Download_rate',
            #     ot.max(
            #         UCSD_comm_group_Download_rate,
            #         UIUC_comm_group_Download_rate,
            #         Georgia_comm_group_Download_rate,
            #         Montana_comm_group_Download_rate,
            #         rho=rho,
            #     ),
            # )
            # KS_P_comm = orbit_avionics.register_output(
            #     'KS_P_comm',
            #     ot.max(
            #         UCSD_comm_group_P_comm,
            #         UIUC_comm_group_P_comm,
            #         Georgia_comm_group_P_comm,
            #         Montana_comm_group_P_comm,
            #         rho=rho,
            #     ),
            # )

            for ground_station in cubesat.children:
                Ground_station_name = ground_station['name']

                orbit_avionics.connect(
                    '{}_comm_group.Download_rate'.format(Ground_station_name),
                    '{}_comm_group_Download_rate'.format(Ground_station_name),
                )

            if add_battery:
                baseline_power = 6.3
                solar_power = orbit_avionics.declare_input(
                    'solar_power',
                    shape=(num_times, ),
                )
                battery_output_power_slow = orbit_avionics.register_output(
                    'battery_output_power_slow',
                    solar_power - KS_P_comm - baseline_power,
                )
                min_battery_output_power_slow = self.register_output(
                    'min_battery_output_power_slow',
                    ot.min(
                        battery_output_power_slow,
                        rho=100,
                    ),
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
                    BatteryPack(
                        num_times=num_times * step,
                        min_soc=0.05,
                        max_soc=0.95,
                        # periodic_soc=True,
                        optimize_plant=optimize_plant,
                        step_size=battery_time_scale,
                    ),
                    promotes=['*'],
                )
                battery_mass = orbit_avionics.declare_input('battery_mass')
                battery_volume = orbit_avionics.declare_input('battery_volume')

                battery_mass_exp = orbit_avionics.register_output(
                    'battery_mass_exp',
                    ot.expand(
                        battery_mass,
                        shape=(num_times, ),
                    ),
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

            if add_battery:
                battery_and_propellant_volume = self.register_output(
                    'battery_and_propellant_volume',
                    total_propellant_volume + battery_volume,
                )
                battery_and_propellant_mass = self.register_output(
                    'battery_and_propellant_mass',
                    total_propellant_used + battery_mass,
                )
                self.add_constraint(
                    'battery_and_propellant_mass',
                    lower=0,
                )

                # 1U (10cm)**3
                u = 10**3 / 100**3
                self.add_constraint(
                    'battery_and_propellant_volume',
                    lower=0,
                    # upper=u,
                )

            self.add_subsystem(
                'Data_download_rk4_comp',
                DataDownloadComp(
                    num_times=num_times,
                    step_size=step_size,
                ),
                promotes=['*'],
            )
            Data = self.declare_input('Data', shape=(1, num_times))

            total_data = self.register_output(
                'total_data',
                Data[0, num_times - 1] - Data[0, 0],
            )


if __name__ == '__main__':

    from lsdo_cubesat.api import Cubesat
    from lsdo_cubesat.options.ground_station import GroundStation
    from openmdao.api import Problem, n2

    num_times = 20
    num_cp = 4
    step_size = 0.1
    initial_orbit_state_magnitude = np.array([1e-3] * 3 + [1e-3] * 3)
    attitude_time_scale = step_size
    battery_time_scale = step_size

    cubesat = Cubesat(
        name='sunshade',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
        approx_altitude_km=500.,
        specific_impulse=47.,
        apogee_altitude=500.001,
        perigee_altitude=499.99,
    )
    prob = Problem()
    prob.model = CubesatGroup(
        num_times=num_times,
        num_cp=num_cp,
        step_size=step_size,
        cubesat=cubesat,
        mtx=get_bspline_mtx(num_cp, num_times, order=4),
        ground_station=GroundStation(
            name='UCSD',
            lon=-117.1611,
            lat=32.7157,
            alt=0.4849,
        ),
        add_battery=False,
        sm=None,
        optimize_plant=False,
        attitude_integrator=False,
        attitude_time_scale=attitude_time_scale,
        battery_time_scale=battery_time_scale,
    )
    prob.setup()
    prob.run_model()
    n2(prob)

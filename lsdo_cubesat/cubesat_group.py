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
from lsdo_cubesat.solar.solar_illumination import SolarIllumination
from lsdo_cubesat.utils.api import (ArrayExpansionComp, BsplineComp,
                                    LinearCombinationComp,
                                    PowerCombinationComp, ScalarExpansionComp,
                                    get_bspline_mtx)
from lsdo_cubesat.utils.comps.arithmetic_comps.elementwise_max_comp import \
    ElementwiseMaxComp
from lsdo_cubesat.utils.ks_comp import KSComp
from lsdo_cubesat.utils.slice_comp import SliceComp

rho = 100.


class OrbitAvionics(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('cubesat')
        self.options.declare('mtx')
        self.options.declare('add_battery', types=bool)
        self.options.declare('optimize_plant', types=bool)

    def setup(self):
        num_times = self.options['num_times']
        num_cp = self.options['num_cp']
        step_size = self.options['step_size']
        cubesat = self.options['cubesat']
        mtx = self.options['mtx']
        add_battery = self.options['add_battery']
        optimize_plant = self.options['optimize_plant']

        self.nonlinear_solver = NonlinearBlockGS(
            iprint=0,
            maxiter=40,
            atol=1e-14,
            rtol=1e-12,
        )
        self.linear_solver = LinearBlockGS(
            iprint=0,
            maxiter=40,
            atol=1e-14,
            rtol=1e-12,
        )

        if add_battery:
            KS_P_comm = self.create_output('KS_P_comm', shape=(num_times, ))
        if add_battery:
            solar_power = self.declare_input(
                'solar_power',
                shape=(num_times, ),
            )
            baseline_power = 6.3
            battery_output_power = self.register_output(
                'battery_output_power',
                solar_power - KS_P_comm - baseline_power,
            )
            self.add_subsystem(
                'battery',
                BatteryPack(
                    num_times=num_times,
                    min_soc=0.05,
                    max_soc=0.95,
                    # periodic_soc=True,
                    optimize_plant=optimize_plant,
                    step_size=step_size,
                ),
                promotes=['*'],
            )
            battery_mass = self.declare_input('battery_mass')
            battery_volume = self.declare_input('battery_volume')
            total_propellant_volume = self.declare_input(
                'total_propellant_volume')
            total_propellant_used = self.declare_input('total_propellant_used')
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

            battery_mass = self.declare_input('battery_mass')

            battery_mass_exp = self.register_output(
                'battery_mass_exp',
                ot.expand(
                    battery_mass,
                    shape=(num_times, ),
                ),
            )

        self.add_subsystem(
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

        # # compute osculating orbit angular speed to feed into
        # # attitude model
        # self.add_subsystem(
        #     'orbit_angular_speed_group',
        #     OrbitAngularSpeedGroup(num_times=num_times, ),
        #     promotes=['*'],
        # )

        for ground_station in cubesat.children:
            self.add_subsystem(
                '{}_comm_group'.format(ground_station['name']),
                CommGroup(
                    num_times=num_times,
                    num_cp=num_cp,
                    step_size=step_size,
                    ground_station=ground_station,
                    mtx=mtx,
                ),
            )
        comm_powers = [
            self.declare_input(cs['name'] + '_comm_group_P_comm',
                               shape=(1, num_times)) for cs in cubesat.children
        ]
        if add_battery:
            KS_P_comm.define(ot.max(
                *comm_powers,
                rho=rho,
            ), )
        else:
            KS_P_comm = self.register_output(
                'KS_P_comm',
                ot.max(
                    *comm_powers,
                    rho=rho,
                ),
            )


class CubesatGroup(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('cubesat')
        self.options.declare('mtx')
        self.options.declare('add_battery', types=bool)
        self.options.declare('sm')
        self.options.declare('optimize_plant', types=bool)

    def setup(self):
        num_times = self.options['num_times']
        num_cp = self.options['num_cp']
        step_size = self.options['step_size']
        cubesat = self.options['cubesat']
        mtx = self.options['mtx']
        add_battery = self.options['add_battery']
        sm = self.options['sm']
        optimize_plant = self.options['optimize_plant']

        initial_data = self.create_indep_var(
            'initial_data',
            val=np.zeros((1, )),
        )

        # # LEAVE THIS HERE (COMMENTED OUT)
        # group = AttitudeOdeGroup(
        #     num_times=num_times,
        #     num_cp=num_cp,
        #     cubesat=cubesat,
        #     mtx=mtx,
        #     step_size=step_size,
        # )
        # self.add_subsystem('attitude_group', group, promotes=['*'])
        # group = SliceComp(
        #     shape=(3, 3, num_times * step),
        #     step=step,
        #     slice_axis=2,
        #     in_name='rot_mtx_i_b_3x3xn_fast',
        #     out_name='rot_mtx_i_b_3x3xn',
        # )
        # self.add_subsystem('rot_mtx_slow_ts_comp', group,
        # promotes=['*'])

        # TODO: add actuators here
        # output external_torques_x, external_torques_y,
        # external_torques_z

        self.add_subsystem(
            'attitude_group',
            AttitudeOdeGroup(
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
            comp = SolarIllumination(num_times=num_times, sm=sm)
            self.add_subsystem('solar_illumination', comp, promotes=['*'])
            sunlit_area = self.declare_input(
                'sunlit_area',
                shape=(num_times, ),
            )
            solar_power = self.register_output('solar_power',
                                               power_over_area * sunlit_area)

            # comp = SolarPanelVoltage(num_times=num_times)
            # self.add_subsystem('solar_panel_voltage', comp, promotes=['*'])

        # TODO: connect thrust in propulsion gorup to force in orbit group
        self.add_subsystem(
            'propulsion_group',
            PropulsionGroup(
                num_times=num_times,
                num_cp=num_cp,
                step_size=step_size,
                cubesat=cubesat,
                mtx=mtx,
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'aerodynamics_group',
            AerodynamicsGroup(
                num_times=num_times,
                num_cp=num_cp,
                step_size=step_size,
                cubesat=cubesat,
                mtx=mtx,
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'orbit_avionics',
            OrbitAvionics(
                num_times=num_times,
                num_cp=num_cp,
                step_size=step_size,
                cubesat=cubesat,
                mtx=mtx,
                add_battery=add_battery,
                optimize_plant=optimize_plant,
            ),
            promotes=['*'],
        )

        if add_battery:
            battery_output_power = self.declare_input('battery_output_power',
                                                      shape=(num_times, ))
            min_battery_output_power = self.register_output(
                'min_battery_output_power',
                ot.min(
                    battery_output_power,
                    rho=rho,
                ),
            )

        shape = (1, num_times)

        download_rates = [
            self.declare_input(cs['name'] + '_comm_group_Download_rate',
                               shape=(1, num_times)) for cs in cubesat.children
        ]
        self.register_output(
            'KS_Download_rate',
            ot.max(
                *download_rates,
                rho=rho,
            ),
        )

        for ground_station in cubesat.children:
            ground_station_name = ground_station['name']

            self.connect(
                '{}_comm_group.Download_rate'.format(ground_station_name),
                '{}_comm_group_Download_rate'.format(ground_station_name),
            )

            self.connect(
                '{}_comm_group.P_comm'.format(ground_station_name),
                '{}_comm_group_P_comm'.format(ground_station_name),
            )

            self.connect(
                'orbit_state_km',
                '{}_comm_group.orbit_state_km'.format(ground_station_name),
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
    from lsdo_cubesat.examples.data.cubesat_xdata import cubesat_xdata as az
    from lsdo_cubesat.examples.data.cubesat_ydata import cubesat_ydata as el
    from lsdo_cubesat.examples.data.cubesat_zdata import cubesat_zdata as yt
    from lsdo_cubesat.solar.smt_exposure import smt_exposure

    num_times = 20
    num_cp = 4
    step_size = 0.1
    initial_orbit_state_magnitude = np.array([1e-3] * 3 + [1e-3] * 3)

    cubesat = Cubesat(
        name='detector',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
        approx_altitude_km=500.,
        specific_impulse=47.,
        apogee_altitude=500.001,
        perigee_altitude=499.99,
    )

    cubesat.add(
        GroundStation(
            name='UCSD',
            lon=-117.1611,
            lat=32.7157,
            alt=0.4849,
        ))
    cubesat.add(
        GroundStation(
            name='UIUC',
            lon=-88.2272,
            lat=32.8801,
            alt=0.2329,
        ))
    cubesat.add(
        GroundStation(
            name='Georgia',
            lon=-84.3963,
            lat=33.7756,
            alt=0.2969,
        ))
    cubesat.add(
        GroundStation(
            name='Montana',
            lon=-109.5337,
            lat=33.7756,
            alt=1.04,
        ))

    prob = Problem()
    prob.model = CubesatGroup(
        num_times=num_times,
        num_cp=num_cp,
        step_size=step_size,
        cubesat=cubesat,
        mtx=get_bspline_mtx(num_cp, num_times, order=4),
        add_battery=False,
        sm=smt_exposure(20, az, el, yt),
        optimize_plant=False,
    )
    prob.setup()
    prob.run_model()
    n2(prob)

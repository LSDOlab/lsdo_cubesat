from csdl.core.output import Output
import numpy as np

from csdl import Model
import csdl

from lsdo_cubesat.parameters.cubesat import CubesatParams
# from lsdo_cubesat.communication.comm_group import CommGroup
# from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.dynamics.vehicle_dynamics import VehicleDynamics
from lsdo_cubesat.operations.sun_los import SunLOS
from lsdo_cubesat.eps.electrical_power_system import ElectricalPowerSystem
from lsdo_cubesat.solar.solar_exposure import SolarExposure


class Cubesat(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('num_cp', types=int)
        self.parameters.declare('step_size', types=float)
        # CADRE
        # self.parameters.declare('rw_voltage', default=4, types=float)
        # BCTRWP015 (10-14V)
        self.parameters.declare('rw_voltage', default=10., types=float)
        self.parameters.declare('cubesat', types=CubesatParams)
        self.parameters.declare('comm', types=bool, default=False)

    def define(self):
        num_times = self.parameters['num_times']
        num_cp = self.parameters['num_cp']
        step_size = self.parameters['step_size']
        cubesat = self.parameters['cubesat']
        comm = self.parameters['comm']
        rw_voltage = self.parameters['rw_voltage']

        self.add(
            VehicleDynamics(
                num_times=num_times,
                num_cp=num_cp,
                step_size=step_size,
                cubesat=cubesat,
            ),
            name='vehicle_dynamics',
        )
        position_km = self.declare_variable(
            'position_km',
            shape=(3, num_times),
        )
        rw_torque = self.declare_variable(
            'rw_torque',
            shape=(3, num_times),
        )
        rw_speed = self.declare_variable(
            'rw_speed',
            shape=(3, num_times),
        )

        # three currents running in parallel
        rw_current = csdl.sum(
            (4.9e-4 * rw_speed + 4.5e2 * rw_torque)**2 + 0.017, axes=(0, ))
        rw_power = rw_current * rw_voltage
        self.register_output('rw_power', rw_power)

        # if comm is True:
        #     self.add_comm()

        sun_direction = self.declare_variable('sun_direction',
                                              shape=(3, num_times))

        # check if s/c is in Earth's shadow
        # Compute direction of sun in body coordinate frame, use to
        # compute line of sight to sun and solar exposure
        sun_LOS = csdl.custom(
            position_km,
            sun_direction,
            op=SunLOS(num_times=num_times),
        )
        self.register_output('sun_LOS', sun_LOS)

        # component of sun direction normal to solar panels in body frame
        B_from_ECI = self.declare_variable('B_from_ECI',
                                           shape=(3, 3, num_times))
        sun_direction_body = csdl.einsum(
            B_from_ECI,
            sun_direction,
            subscripts='ijk,jk->ik',
            partial_format='sparse',
        )
        sun_component = sun_direction_body[2, :]
        self.register_output(
            'sun_component',
            sun_component,
        )

        percent_facing_sun = csdl.custom(
            sun_component,
            op=SolarExposure(num_times=num_times),
        )
        self.register_output(
            'percent_exposed_area',
            percent_facing_sun * sun_LOS,
        )

        self.add(
            ElectricalPowerSystem(
                num_times=num_times,
                step_size=step_size,
                comm=comm,
            ),
            name='EPS',
        )

        total_propellant_volume = self.declare_variable(
            'total_propellant_volume', )
        battery_volume = self.declare_variable('battery_volume')
        battery_and_propellant_volume = total_propellant_volume + battery_volume
        self.register_output('battery_and_propellant_volume',
                             battery_and_propellant_volume)

        # volume of half of 1U
        self.add_constraint('battery_and_propellant_volume',
                            upper=10**3 / (10**3) / 2)

        if comm is True:
            self.add_download_rate_model()

    # def add_comm(self):
    #     num_times = self.parameters['num_times']
    #     num_cp = self.parameters['num_cp']
    #     step_size = self.parameters['step_size']
    #     cubesat = self.parameters['cubesat']

    #     comm_powers = []
    #     for ground_station in cubesat.children:
    #         name = ground_station['name']

    #         self.add(
    #             CommGroup(
    #                 num_times=num_times,
    #                 num_cp=num_cp,
    #                 step_size=step_size,
    #                 ground_station=ground_station,
    #             ),
    #             name='{}_comm_group'.format(name),
    #             promotes=['orbit_state_km'],
    #         )
    #         comm_powers.append(
    #             self.declare_variable(
    #                 '{}_P_comm'.format(name),
    #                 shape=(num_times, ),
    #             ))
    #         self.connect(
    #             '{}_comm_group.P_comm'.format(name),
    #             '{}_P_comm'.format(name),
    #         )
    #     comm_power = csdl.sum(*comm_powers)
    #     self.register_output('comm_power', comm_power)

    def add_download_rate_model(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        cubesat = self.parameters['cubesat']

        dl_rates = []
        for name in [
                'UCSD_comm_group_Download_rate',
                'UIUC_comm_group_Download_rate',
                'Georgia_comm_group_Download_rate',
                'Montana_comm_group_Download_rate',
        ]:
            dl_rates.append(self.declare_variable(name, shape=(num_times, )))
        KS_Download_rate = self.register_output(
            'KS_Download_rate',
            csdl.expand(
                csdl.max(*dl_rates, rho=100.),
                (1, num_times),
                indices='i->ji',
            ),
        )

        for Ground_station in cubesat.children:
            Ground_station_name = Ground_station['name']

            self.connect(
                '{}_comm_group.Download_rate'.format(Ground_station_name),
                '{}_comm_group_Download_rate'.format(Ground_station_name),
            )

        Data = csdl.custom(
            KS_Download_rate,
            self.create_input('Initial_Data', val=0),
            op=DataDownloadComp(
                num_times=num_times,
                step_size=step_size,
            ),
        )

        self.register_output('Data', Data)

        total_data = Data[0, -1] - Data[0, 0]
        self.register_output('total_data', total_data)


if __name__ == "__main__":
    from csdl_om import Simulator
    from lsdo_cubesat.parameters.cubesat import CubesatParams
    initial_orbit_state_magnitude = np.array([1e-3] * 3 + [1e-3] * 3)
    sim = Simulator(
        Cubesat(num_times=10,
                num_cp=3,
                step_size=0.1,
                cubesat=CubesatParams(
                    name='detector',
                    dry_mass=1.3,
                    initial_orbit_state=initial_orbit_state_magnitude *
                    np.random.rand(6),
                    specific_impulse=47.,
                    perigee_altitude=500.002,
                    apogee_altitude=499.98,
                )))
    sim.visualize_implementation()

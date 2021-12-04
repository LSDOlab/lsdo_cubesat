from csdl.core.output import Output
import numpy as np

from csdl import Model
import csdl

from lsdo_cubesat.api import CubesatParams
from lsdo_cubesat.communication.comm_group import CommGroup
# from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.dynamics.vehicle_dynamics import VehicleDynamics
from lsdo_cubesat.solar.sun_los import SunLOS
from lsdo_cubesat.eps.electrical_power_system import ElectricalPowerSystem
from lsdo_cubesat.solar.solar_exposure import SolarExposure


class Cubesat(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('num_cp', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('cubesat', types=CubesatParams)
        self.parameters.declare('comm', types=bool, default=False)

    def define(self):
        num_times = self.parameters['num_times']
        num_cp = self.parameters['num_cp']
        step_size = self.parameters['step_size']
        cubesat = self.parameters['cubesat']
        comm = self.parameters['comm']

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
        rw_power = rw_torque * rw_speed
        self.register_output('rw_power', rw_power)

        if comm is True:
            self.add_comm()

        # check if s/c is in Earth's shadow
        sun_direction = self.compute_sun_direction()
        sun_LOS = csdl.custom(
            position_km,
            sun_direction,
            op=SunLOS(num_times=num_times),
        )
        self.register_output('sun_LOS', sun_LOS)

        sun_component = -sun_direction[2, :]
        self.register_output(
            'sun_component',
            sun_component,
        )

        percent_exposed_area = csdl.custom(
            sun_component,
            op=SolarExposure(num_times=num_times),
        )
        self.register_output(
            'percent_exposed_area',
            percent_exposed_area,
        )

        self.add(
            ElectricalPowerSystem(
                num_times=num_times,
                step_size=step_size,
                comm=comm,
            ),
            name='EPS',
        )

        if comm is True:
            self.add_download_rate_model()

    def compute_sun_direction(self) -> Output:
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']

        # TODO: get spin rate right
        earth_spin_rate = 2 * np.pi / 24 / 3600  # rad/s
        t = self.declare_variable('t', val=np.arange(num_times) * step_size)

        # Earth Centered Inertial to Earth Sun Frame
        v = np.zeros((3, 3, num_times))
        # ECI and ESF z axes are parallel
        v[2, 2, :] = 1
        ECI_to_ESF = self.create_output('ECI_to_ESF', val=v)
        et = -earth_spin_rate * t
        ECI_to_ESF[0, 0, :] = csdl.cos(csdl.reshape(et, (1, 1, num_times)))
        ECI_to_ESF[0, 1, :] = csdl.sin(csdl.reshape(et, (1, 1, num_times)))
        ECI_to_ESF[1, 0, :] = csdl.cos(csdl.reshape(et, (1, 1, num_times)))
        ECI_to_ESF[1, 1, :] = -csdl.sin(csdl.reshape(et, (1, 1, num_times)))

        # Body to Earth Sun Frame
        ECI_to_B = self.declare_variable('ECI_to_B', shape=(3, 3, num_times))
        B_to_ECI = csdl.einsum(ECI_to_B, subscripts='ijk->jik')
        B_to_ESF = csdl.einsum(B_to_ECI, ECI_to_ESF, subscripts='ijm,klm->ilm')

        # Earth Sun Frame to Body
        ESF_to_B = csdl.einsum(B_to_ESF, subscripts='ijk->jik')
        sun_direction = csdl.reshape(ESF_to_B[:, 0, :], (3, num_times))
        return self.register_output('sun_direction', sun_direction)

    def add_comm(self):
        num_times = self.parameters['num_times']
        num_cp = self.parameters['num_cp']
        step_size = self.parameters['step_size']
        cubesat = self.parameters['cubesat']

        comm_powers = []
        for ground_station in cubesat.children:
            name = ground_station['name']

            self.add(
                CommGroup(
                    num_times=num_times,
                    num_cp=num_cp,
                    step_size=step_size,
                    ground_station=ground_station,
                ),
                name='{}_comm_group'.format(name),
                promotes=['orbit_state_km'],
            )
            comm_powers.append(
                self.declare_variable(
                    '{}_P_comm'.format(name),
                    shape=(num_times, ),
                ))
            self.connect(
                '{}_comm_group.P_comm'.format(name),
                '{}_P_comm'.format(name),
            )
        comm_power = csdl.sum(*comm_powers)
        self.register_output('comm_power', comm_power)

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
    from lsdo_cubesat.api import CubesatParams
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
                    approx_altitude_km=500.,
                    specific_impulse=47.,
                    perigee_altitude=500.002,
                    apogee_altitude=499.98,
                )))
    sim.visualize_implementation()

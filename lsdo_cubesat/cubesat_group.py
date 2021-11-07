import numpy as np

from csdl import Model
import csdl

from lsdo_cubesat.communication.comm_group import CommGroup
# from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.dynamics.vehicle_dynamics import VehicleDynamics


class Cubesat(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('num_cp', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('cubesat')

    def define(self):
        num_times = self.parameters['num_times']
        num_cp = self.parameters['num_cp']
        step_size = self.parameters['step_size']
        cubesat = self.parameters['cubesat']

        times = np.linspace(0., step_size * (num_times - 1), num_times)
        # TODO: does this need to be an input?
        times = self.create_input('times', units='s', val=times)

        self.add(
            VehicleDynamics(
                num_times=num_times,
                num_cp=num_cp,
                step_size=step_size,
                cubesat=cubesat,
            ),
            name='vehicle_dynamics',
        )

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
                promotes=['orbit_state_km', 'radius'],
            )

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

import numpy as np
from openmdao.api import ExecComp, Group, IndepVarComp, Problem

from lsdo_cubesat.attitude.rot_mtx_b_i_comp import RotMtxBIComp
from lsdo_cubesat.communication.Antenna_rot_mtx import AntennaRotationMtx
from lsdo_cubesat.communication.Antenna_rotation import AntRotationComp
from lsdo_cubesat.communication.Comm_Bitrate import BitRateComp
from lsdo_cubesat.communication.Comm_distance import \
    StationSatelliteDistanceComp
from lsdo_cubesat.communication.Comm_LOS import CommLOSComp
from lsdo_cubesat.communication.Comm_vector_antenna import AntennaBodyComp
from lsdo_cubesat.communication.Comm_VectorBody import VectorBodyComp
from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.communication.Earth_spin_comp import EarthSpinComp
from lsdo_cubesat.communication.Earthspin_rot_mtx import EarthspinRotationMtx
from lsdo_cubesat.communication.GSposition_ECEF_comp import GS_ECEF_Comp
from lsdo_cubesat.communication.GSposition_ECI_comp import GS_ECI_Comp
# from lsdo_cubesat.communication.rot_mtx_ECI_EF_comp import RotMtxECIEFComp
from lsdo_cubesat.communication.Vec_satellite_GS_ECI import Comm_VectorECI
from lsdo_cubesat.utils.api import (ArrayExpansionComp, BsplineComp,
                                    LinearCombinationComp,
                                    PowerCombinationComp, get_bspline_mtx)

# from lsdo_cubesat.cubesat_group import CubesatGroup


class GSGroup(Group):
    def initialize(self):
        self.options.declare('ground_station')
        self.options.declare('num_times', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('step_size', types=float)

        self.options.declare('mtx')

    def setup(self):
        ground_station = self.options['ground_station']
        num_times = self.options['num_times']
        num_cp = self.options['num_cp']
        step_size = self.options['step_size']

        mtx = self.options['mtx']

        comp = IndepVarComp()
        # comp.add_output('P_comm_cp', val=0.25 * np.ones(num_cp), units='W')
        comp.add_output('P_comm_cp', val=np.zeros(num_cp), units='W')
        comp.add_design_var('P_comm_cp', lower=0., upper=100.)
        comp.add_output('gain', val=16.0 * np.ones(num_times))
        comp.add_output('Initial_Data', val=0.0)
        for var_name in [
                'lon',
                'lat',
                'alt',
                'antenna_angle',
        ]:
            comp.add_output(var_name, val=ground_station[var_name])

        self.add_subsystem('inputs_comp', comp, promotes=['*'])

        comp = EarthSpinComp(num_times=num_times)
        self.add_subsystem('q_E', comp, promotes=['*'])

        comp = EarthspinRotationMtx(num_times=num_times)
        self.add_subsystem('Rot_ECI_EF', comp, promotes=['*'])

        comp = GS_ECEF_Comp(num_times=num_times)
        self.add_subsystem('r_e2g_E', comp, promotes=['*'])

        comp = GS_ECI_Comp(num_times=num_times)
        self.add_subsystem('r_e2g_I', comp, promotes=['*'])

        comp = Comm_VectorECI(num_times=num_times)
        self.add_subsystem('r_b2g_I', comp, promotes=['*'])

        comp = CommLOSComp(num_times=num_times)
        self.add_subsystem('CommLOS', comp, promotes=['*'])

        comp = VectorBodyComp(num_times=num_times)
        self.add_subsystem('r_b2g_B', comp, promotes=['*'])

        comp = AntRotationComp(num_times=num_times)
        self.add_subsystem('antenna_orientation', comp, promotes=['*'])

        comp = AntennaRotationMtx(num_times=num_times)
        self.add_subsystem('Rot_AB', comp, promotes=['*'])

        comp = AntennaBodyComp(num_times=num_times)
        self.add_subsystem('r_b2g_A', comp, promotes=['*'])

        comp = StationSatelliteDistanceComp(num_times=num_times)
        self.add_subsystem('Gsdist', comp, promotes=['*'])

        comp = BsplineComp(
            num_pt=num_times,
            num_cp=num_cp,
            jac=mtx,
            in_name='P_comm_cp',
            out_name='P_comm',
        )
        self.add_subsystem('P_comm_comp', comp, promotes=['*'])

        comp = BitRateComp(num_times=num_times)
        self.add_subsystem('Download_rate', comp, promotes=['*'])

        comp = DataDownloadComp(
            num_times=num_times,
            step_size=step_size,
        )
        self.add_subsystem('Data_download_rk4_comp', comp, promotes=['*'])

        comp = ExecComp(
            'total_data_downloaded= Data[-1] - Data[0]',
            Data=np.empty(num_times),
        )
        self.add_subsystem('total_data_downloaded_comp', comp, promotes=['*'])

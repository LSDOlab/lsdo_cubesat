import numpy as np
from openmdao.api import Problem, IndepVarComp, Group, ExecComp

from lsdo_cubesat.attitude.rot_mtx_b_i_comp import RotMtxBIComp
from lsdo_cubesat.communication.Antenna_rot_mtx import AntennaRotationMtx
from lsdo_cubesat.communication.Antenna_rotation import AntRotationComp
from lsdo_cubesat.communication.Comm_Bitrate import BitRateComp
from lsdo_cubesat.communication.Comm_distance import StationSatelliteDistanceComp
from lsdo_cubesat.communication.Comm_LOS import CommLOSComp
from lsdo_cubesat.communication.Comm_VectorBody import VectorBodyComp
from lsdo_cubesat.communication.GSposition_ECEF_comp import GS_ECEF_Comp
from lsdo_cubesat.communication.GSposition_ECI_comp import GS_ECI_Comp
# from lsdo_cubesat.communication.rot_mtx_ECI_EF_comp import RotMtxECIEFComp
from lsdo_cubesat.communication.Vec_satellite_GS_ECI import Comm_VectorECI
from lsdo_cubesat.communication.Comm_vector_antenna import AntennaBodyComp
from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.communication.Earth_spin_comp import EarthSpinComp
from lsdo_cubesat.communication.Earthspin_rot_mtx import EarthspinRotationMtx
# from lsdo_cubesat.communication.Ground_comm import Groundcomm

class CommGroup(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('cubesat')
        self.options.declare('mtx')
        # self.options.declare('ground_station')

    def setup(self):
        num_times = self.options['num_times']
        num_cp = self.options['num_cp']
        step_size = self.options['step_size']
        cubesat = self.options['cubesat']
        mtx = self.options['mtx']
        # ground_station = self.options['ground_station']

        comp = IndepVarComp()

        comp.add_output('lon', val=32.8563)
        comp.add_output('lat', val=-117.2500)
        comp.add_output('alt', val=0.4849368)
        # comp.add_output('t', val=np.zeros(num_times))
        # comp.add_output('rot_mtx_i_b_3x3xn', val=np.zeros((3, 3, num_times)))
        # comp.add_output('orbit_state_km', val=np.zeros((6, num_times)))
        comp.add_output('antAngle', val=0.0)
        comp.add_output('P_comm', val=np.zeros(num_times))
        comp.add_output('Gain', val=np.zeros(num_times))
        comp.add_output('Initial_Data', val=0.0)

        comp.add_design_var('antAngle')

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
        self.add_subsystem('q_A', comp, promotes=['*'])

        comp = AntennaRotationMtx(num_times=num_times)
        self.add_subsystem('Rot_AB',comp, promotes=['*'])

        comp = AntennaBodyComp(num_times=num_times)
        self.add_subsystem('r_b2g_A',comp, promotes=['*'])

        comp = StationSatelliteDistanceComp(num_times=num_times)
        self.add_subsystem('Gsdist',comp, promotes=['*'])

        comp = BitRateComp(num_times=num_times)
        self.add_subsystem('Download_rate',comp, promotes=['*'])

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



# if __name__ == '__main__':
#     import numpy as np
#     import openmdao.api as om

#     from openmdao.api import Problem, IndepVarComp, Group

#     "Data from Goldestone Deep Space Communication Complex"

#     num_times = 3

#     prob = Problem()
#     prob.CommGroup = CommGroup

#     prob.model.add_subsystem('lon', om.IndepVarComp('lon', -116.8873))
#     prob.model.add_subsystem('lat', om.IndepVarComp('lat', 35.4227))
#     prob.model.add_subsystem('alt', om.IndepVarComp('alt', 0.9))
#     prob.model.add_subsystem('t', om.IndepVarComp('t', np.arange(1, num_times + 1, 1)))
#     prob.model.add_subsystem('Rot_b_i', om.IndepVarComp('Rot_b_i', np.random.random((3,3,num_times))))
#     prob.model.add_subsystem('r_e2b_i', om.IndepVarComp('r_e2b_i', np.random.random((6, num_times))))
#     prob.model.add_subsystem('antAngle', om.IndepVarComp('antAngle', 1.5))
#     prob.model.add_subsystem('P_comm', om.IndepVarComp('P_comm', 10 * np.ones(num_times)))
#     prob.model.add_subsystem('gain', om.IndepVarComp('gain', np.random.randint(1.76,40,size=num_times)))
#     prob.model.add_subsystem('Initial_Data', om.IndepVarComp('Initial_Data', 0.0))

#     comm_group = CommGroup(num_times=num_times)
#     prob.model.add_subsystem('comm_group', comm_group, promotes=['*'])

#     prob.setup(check=True)
#     prob.run()

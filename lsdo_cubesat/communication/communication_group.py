import numpy as np

from openmdao.api import Group, IndepVarComp, NonlinearBlockGS, LinearBlockGS
from lsdo_utils.api import ArrayReorderComp, LinearCombinationComp, PowerCombinationComp, ScalarContractionComp

from lsdo_cubesat.communication.Data_download_comp import DataDownloadComp


from lsdo_cubesat.communication.Comm_antenna_rotation import AntennaRotationComp
from lsdo_cubesat.communication.Comm_earth_spin import EarthQuaternionComp
from lsdo_cubesat.communication.Comm_EarthSpinMtx import EarthsSpinMtxComp
from lsdo_cubesat.communication.Comm_GSposEarth import GroundStationComp
from lsdo_cubesat.communication.Comm_inertial_station import GSposECIComp
from lsdo_cubesat.communication.Comm_antenna_Mtx import AntennaRotationMtx
from lsdo_cubesat.communication.Comm_BitRate import BitRateComp
from lsdo_cubesat.communication.Comm_LOS import CommLOSComp
from lsdo_cubesat.communication.Comm_TransGain import TransGainComp
from lsdo_cubesat.communication.Comm_distance import SationSatelliteDistanceComp


class CommunicationGroup(Group):
    
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('cubesat')
        self.options.declare('mtx')
        
    def setup(self):
        num_times = self.options['num_times']
        step_size = self.options['step_size']
        
        cubesat = self.options['cubesat']
        mtx = self.options['mtx']
        
        
        # Design Parameters
        
        comp = IndepVarComp()
        comp.add_output('Data0',val= )
        comp.add_output('antAngle',val= )
        comp.add_output('lon',val= )
        comp.add_output('lat',val= )
        comp.add_output('alt',val= )
        comp.add_output('r_e2b_I',val= )
        comp.add_output('O_BI',val= )
        comp.add_output('t',val= )
        self.add_subsystem('inputs_comp', comp, promotes=['*'])
        
        # Antenna angle to body frame
        comp = AntennaRotationComp(
            num_times = num_times,
            antAngle_name = 'antAngle'
        )
        self.add_subsystem('q_A_comp',comp,promotes=['*'])
        
        comp = AntennaRotationMtx(
            num_times = num_times,
        )
        self.add_subsystem('Rot_AB_comp',comp,promotes=['*'])
        
        #Ground Station Position
        comp = GroundStationComp(
            num_times = num_times,
        )
        self.add_subsystem('r_e2g_E_comp',comp,promotes=['*'])
        
        comp = EarthQuaternionComp(
            num_times = num_times,
        )
        self.add_subsystem('q_E_comp',comp,promotes=['*'])
        
        #Earth Spin returns earth quarternion
        comp = EarthsSpinMtxComp(
            num_times = num_times,
        )
        self.add_subsystem('O_IE_comp',comp,promotes=['*'])
        
        #returns inertial frame ground station
        comp = GSposECIComp(
            num_times = num_times,
        )
        self.add_subsystem('r_e2g_I_comp',comp,promotes=['*'])
        
        comp =(
        )
        self.add_subsystem('r_b2g_I_comp',comp,promotes=['*'])
        
        comp =
        
        self.add_subsystem('r_b2g_B_comp',comp,promotes=['*'])
        
        comp =
        
        self.add_subsystem('r_b2g_A_comp',comp,promotes=['*'])
        
        comp =
        
        self.add_subsystem('azimuthGS_comp',comp,promotes=['*'])
        self.add_subsystem('elevationGS_comp',comp,promotes=['*'])
        
        comp =  TransGainComp(
            num_times = num_times,
        )
        
        self.add_subsystem('gain',comp,promotes=['*'])
        
        comp = SationSatelliteDistanceComp(
            num_times = num_times,
        )
        
        self.add_subsystem('GSdist',comp,promotes=['*'])
        
        comp = CommLOSComp(
            num_times = num_times,
        )
        self.add_subsystem('CommLOS_comp',comp,promotes=['*'])
  
        comp = BitRateComp(
            num_times = num_times,
        )
        self.add_subsystem('Download_rate_comp',comp,promotes=['*'])
        
        comp = DataDownloadComp(
            num_times = num_times,
            step_size = step_size,
        )
        self.add_subsystem('Data_download_rk4_comp',comp, promotes=['*'])
        
        
        
        
        
        
